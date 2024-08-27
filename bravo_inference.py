from mmseg.apis import init_model, inference_model
from tqdm import tqdm
import os
import mmcv
from PIL import Image
import torch
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
from torch import nn, optim, utils
import torch.nn.functional as F

config_file = '/cluster/work/cvl/shbasu/phyfeaSegformer/data/segformer_mit-b5_8xb1-160k_cityscapes-1024x1024.py'  #/content/mmsegmentation/ocrnet_hr18_4xb2-40k_cityscapes-512x1024.py'
checkpoint_file =  '/cluster/work/cvl/shbasu/phyfeaSegformer/data/segformer_mit-b5_8x1_1024x1024_160k_cityscapes_20211206_072934-87a052ec.pth'

model = init_model(config_file, checkpoint_file, device="cuda:0")

splits = ['ACDC', 'SMIYC', 'outofcontext', 'synflare', 'synobjs', 'synrain']
modified_suffix_from = ['.png', '.jpg', '.png', '.png', '.png', '.png']


imgs = []
for split in splits:

  output = f'/cluster/work/cvl/shbasu/phyfeaSegformer/data/output/bravo_{split}'
  input = f'/cluster/work/cvl/shbasu/phyfeaSegformer/data/input/bravo_{split}'
  if not os.path.exists(output):
    os.makedirs(output)


  for (dirpath, dirnames, filenames) in os.walk(input):

    for filename in filenames:

      if filename.endswith('.png') or filename.endswith('.jpg'):
        imgs.append(dirpath + '/' + filename)

  split_idx = splits.index(split)
  img_suffix = modified_suffix_from[split_idx]

  for i, file in enumerate(tqdm(imgs)):

    filepath = file[len(input):]
    if filepath[0] == '/':
      filepath = filepath[1:]
      filename = filepath.split('/')[-1]
      filepath = filepath.removesuffix(filename)


    destfile = os.path.join(output, filepath)

    image_read = mmcv.imread(file)
    result = inference_model(model,image_read)
    pred = result.pred_sem_seg.data.squeeze().to(torch.uint8).to('cpu').numpy()
    conf = (result.seg_logits.data.softmax(dim=0).max(dim=0,keepdim=True).values.squeeze()* 65535).to(torch.uint16).cpu().numpy()


    pred = Image.fromarray(pred)
    conf = Image.fromarray(conf)

    if not os.path.exists(destfile):
      os.makedirs(destfile)


    pred.save(destfile+filename.replace(img_suffix, '_pred.png'))
    conf.save(destfile+filename.replace(img_suffix, '_conf.png'))



