import mmseg
from mmseg.apis import inference_model, init_model, show_result_pyplot, inference_model_1
import mmcv
import cv2
import glob
import os
import time
from os.path import join, isdir
from os import listdir, rmdir
from shutil import move, rmtree, make_archive
import pickle
print(mmseg.__version__)
# Example output: 0.24.1


import torch
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import os
from typing import Callable, Dict, List, Tuple, Union
from torch import nn, optim, utils
import torch.nn.functional as F
from itertools import combinations

s1 = torch.cuda.Stream('cuda:1')
s2 = torch.cuda.Stream('cuda:2')
s3 = torch.cuda.Stream('cuda:3')

def _get_relu(name: str) -> nn.Sequential:
    container = nn.Sequential()
    relu = nn.ReLU()
    container.add_module(f'{name}_relu', relu)

    return container


def _max_pool2D(name: str) -> nn.Sequential:
    container = nn.Sequential()
    pool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=1)
    container.add_module(f'{name}_maxpool_2d', pool1)
    container.add_module(f'{name}_maxpool_2d_pad_1', nn.ConstantPad2d(1, 1))
    return container


def _avg_pool2D(name: str) -> nn.Sequential:
    container = nn.Sequential()
    pool1 = nn.AvgPool2d(kernel_size=(3, 3), stride=1)
    container.add_module(f'{name}_maxpool_2d', pool1)
    container.add_module(f'{name}_maxpool_2d_pad_1', nn.ConstantPad2d(1, 1))
    return container




class PhysicsFormer(nn.Module):

  def __init__(self, num_classes):

    super(PhysicsFormer,self).__init__()
    self.num_classes = num_classes
    self.relu = _get_relu('relu')
    self.maxpool_1 = _max_pool2D("maxpool_1")
    self.avgpool_1 = _avg_pool2D("avgpool_1")
    self.pad_1 = nn.ConstantPad2d(1,1)
    self.pad_0  = nn.ConstantPad2d(1,0)
    self.T=8

  def opening(self, x):

    relu = x
    for iteration in range(self.T):
        x1 = self.maxpool_1(x)
        x = torch.matmul(x1, relu)

    return x

  def _rectification(self, original, dilated):

    offset = torch.sub(dilated, original, alpha=1)
    offset_mean = torch.mean(offset, dim=2, keepdim=True)
    offset_diff = torch.sub(offset, offset_mean, alpha=1)
    offset_relu = self.relu(offset_diff)
    final_dilation = torch.add(original, offset_relu, alpha=1)
    return final_dilation

  def selective_dilation(self, x):

    relu = x
    for iteration in range(self.T):
        x1 = self.avgpool_1(x)
        x2 = torch.matmul(x1, relu)
        x = self._rectification(x, x2)
    return x

  def final_operation(self, original, mode='opening'):

    if mode == 'opening':
        final_concatenated_opened = torch.mul(original, -1)
        final_concatenated_opening = self.pad_1(final_concatenated_opened)
        operated = self.opening(final_concatenated_opening)
        operated_normalized = F.normalize(operated)
        x = torch.matmul(operated_normalized, final_concatenated_opening)
        subtracted = torch.sub(final_concatenated_opening, x, alpha=1)
    else:
        final_concatenated_dilation = self.pad_1(original)
        operated = self.selective_dilation(final_concatenated_dilation)
        operated_normalized = F.normalize(operated)
        x = torch.matmul(operated_normalized, final_concatenated_dilation)
        subtracted = torch.sub(final_concatenated_dilation, x, alpha=1)

    l1_norm = torch.norm(subtracted, p=1)
    return l1_norm

  def forward(self, input):

      logits_upscaled = nn.functional.interpolate(
            input,
            size=(512,512),
            mode='bilinear',
            align_corners=False)
      
      upscaled_softmax = logits_upscaled.softmax(dim=1)
      tensor_list = []
      perm = combinations(range(self.num_classes), 2)
      
      for i in perm:
          concatenated_tensor = torch.cat(
                (upscaled_softmax[:, i[0]:i[0] + 1, ::], upscaled_softmax[:, i[1]:i[1] + 1, ::]), dim=1)
          logits_mean = torch.mean(concatenated_tensor, dim=1, keepdim=True)
          logits_sub = torch.sub(concatenated_tensor, logits_mean, alpha=1)       
          concat_relu = self.relu(logits_sub)
          tensor_list.append(concat_relu)
          

      final_concatenated = torch.cat(tensor_list, dim=1)
      split_tensor = torch.split(final_concatenated,[171,171],dim=1)
     

      with torch.cuda.stream(s1):

          norm_opened_1 = self.final_operation(split_tensor[0])

      with torch.cuda.stream(s3):

          norm_opened_2 = self.final_operation(split_tensor[1].to('cuda:3'))
 
      with torch.cuda.stream(s2):

          norm_dilated = self.final_operation(final_concatenated.to('cuda:2'), mode='dilation')
      
      torch.cuda.synchronize('cuda:1')
      torch.cuda.synchronize('cuda:2')
      torch.cuda.synchronize('cuda:3')
      final_norm = torch.abs((norm_opened_1+ norm_opened_2.to('cuda:1')) - norm_dilated.to('cuda:1'))
      return final_norm
