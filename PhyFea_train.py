from tqdm import tqdm
#from torchmetrics.classification import MulticlassJaccardIndex
import sys
sys.path.append('/cluster/home/shbasu/valeo/mmseg')
import gc
from PhyFea import PhysicsFormer
import mmseg
from mmseg.apis import inference_model, init_model, show_result_pyplot, inference_model_1
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import os
from torch.utils.data import DataLoader
import torch
from cityscapesscripts.helpers import labels
import torch.nn.functional as F
import torch.nn as nn
import warnings
import wandb
warnings.filterwarnings('ignore')

import random
import numpy as np
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from torchvision.datasets import Cityscapes
from torch.optim import optimizer,AdamW
from torchvision.transforms import v2
#import albumentations as A
#from albumentations.pytorch import ToTensorV2


# def set_device(set_device_to : str):
#
#     device = None
#     if torch.cuda.is_available and set_device_to=='gpu':
#         print('All good, a Gpu is available')
#         device = torch.device("cuda:0")
#     elif torch.cuda.is_available and set_device_to=='cpu':
#         print('cpu is selected as device so setting to cpu')
#         device = torch.device("cpu")
#     else:
#         print('Please set GPU via Edit -> Notebook Settings.')
#
#     return device
#
# device = 'gpu'
# device = set_device(device)

def fix_random(seed: int) -> None:

    """Fix all the possible sources of randomness.

    Args:
        seed: the seed to use.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def encode_segmap(mask):

    mask_copy = mask.copy()
    for label in labels.labels:
        mask_copy[mask==label.id] = label.trainId
        mask_copy[mask_copy==255] = 19
    return mask_copy


transform=v2.Compose(
[
    v2.RandomResizedCrop(size=(1024, 1024), antialias=True),
    #A.HorizontalFlip(),
    #A.Normalize(mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375)),
    v2.ToDtype(torch.float32, scale=True),
]
)

transform_mask = v2.Compose(
[
    #A.Resize(512,512),
    #A.HorizontalFlip(),
    #A.Normalize(mean=(0.385, 0.356, 0.306), std=(0.329, 0.324, 0.325)),
    v2.ToDtype(torch.long),
    
]
)
class MyClass(Cityscapes):
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image = Image.open(self.images[index]).convert('RGB')
        image = np.array(image).transpose()
        image = torch.from_numpy(image)


        targets: Any = []
        for i, t in enumerate(self.target_type):
            if t == 'polygon':
                target = self._load_json(self.targets[index][i])
            else:
                target = Image.open(self.targets[index][i])
                target = np.array(target)
                target = encode_segmap(target)
                target = torch.from_numpy(target)
            targets.append(target)
        target = tuple(targets) if len(targets) > 1 else targets[0]

        if self.transforms is not None:
            image=transform(image)
            target = transform_mask(target)
            

        # transformed['mask'] = nn.functional.interpolate(transformed['mask'].unsqueeze(0).unsqueeze(0),size= (64,64),mode='bilinear',align_corners=False)
        # transformed['mask'] = transformed['mask'].squeeze(0).squeeze(0)
        return image, target

data_train= MyClass('/cluster/work/cvl/shbasu/phyfeaSegformer/data/Cityscapes/', split='train', mode='fine',
                     target_type='semantic',transforms=transform) #coarse

data_val = MyClass('/cluster/work/cvl/shbasu/phyfeaSegformer/data/Cityscapes/', split='val', mode='fine',
                     target_type='semantic',transforms=transform) #coarse

num_workers = os.cpu_count()
size_batch_train =1
size_batch_val = 1 #2 * size_batch_train

loader_train = torch.utils.data.DataLoader(data_train, batch_size=size_batch_train,
                                           shuffle=True,
                                           pin_memory=True,
                                           num_workers=num_workers)
loader_val = torch.utils.data.DataLoader(data_val, batch_size=size_batch_val,
                                         shuffle=False,
                                         num_workers=num_workers)



def train(model,phyfea,train_loader,optimizer,criterion) :
    """Train loop to train a neural network for one epoch.

    Args:
        model: the model to train.
        train_loader: the data loader containing the training data.
        device: the device to use to train the model.
        optimizer: the optimizer to use to train the model.
        criterion: the loss to optimize.
        epoch: the number of the current epoch

    Returns:
        the L1 Loss value on the training data,
        the accuracy on the training data.
    """

    samples_train = 0
    loss_train = 0
    correct = 0
    train_ce_loss = 0
    #metric_train = 0
    #size_ds_train = len(train_loader.dataset)
    #metric = MulticlassJaccardIndex(num_classes=151)

    eps = 1e-10
    L1_values = 0
    
    # IMPORTANT: from now on, since we will introduce batch norm, we have to tell PyTorch if we are training or evaluating our model
    model.train()
    phyfea.train()
    
    for idx_batch, (image, labels) in tqdm(enumerate(train_loader), total=len(train_loader), desc='Sample'): #595

      
      
      labels =  labels.to('cuda:0')

      optimizer.zero_grad()
      final_loss = 0
      

      scores = inference_model_1(model, image.to('cuda:0'))
      final_norm = phyfea(scores.to('cuda:1'))
      scores = nn.functional.interpolate(scores, size=(labels.shape[1], labels.shape[2]), mode='bilinear', align_corners=False)
      ce_loss = criterion(scores, labels) #.unsqueeze(dim=0))
      final_norm = final_norm.to('cuda:0')
      final_loss += (ce_loss + eps*final_norm)
      loss_train += final_loss.item()*len(image)
      train_ce_loss += ce_loss.item()*len(image)
      L1_values += final_norm.item()*len(image)
      samples_train += len(image)
      

      final_loss.backward()
      optimizer.step()
      

      



    loss_train /= samples_train
    train_ce_loss /= samples_train
    L1_values /= samples_train
    
    print(f'loss--->{loss_train}')
    print(f'ce_loss-->{train_ce_loss}')
    print(f'final_norm-->{L1_values}')
    
    wandb.log({
            'batch': idx_batch,
            'memory_allocated': torch.cuda.memory_allocated(),
            'memory_reserved': torch.cuda.memory_reserved(),
            'max_memory_allocated': torch.cuda.max_memory_allocated(),
            'max_memory_reserved': torch.cuda.max_memory_reserved(),
            'final_loss': loss_train,
            'CE_loss' : train_ce_loss,
            'Final_norm' : L1_values
        })
    torch.cuda.empty_cache()
    gc.collect()

    return loss_train, L1_values, train_ce_loss #, log_values


def training_loop(num_epochs: int,
                  optimizer: torch.optim,
                  scheduler: torch.optim,
                  model: nn.Module,
                  phyfea:nn.Module,
                  loader_train: object)->Dict:

    """Executes the training loop.

        Args:
            name_exp: the name for the experiment.
            num_epochs: the number of epochs.
            optimizer: the optimizer to use.
            model: the mode to train.
            loader_train: the data loader containing the training data.
            loader_val: the data loader containing the validation data.


        Returns:
            A dictionary with the statistics computed during the train:
            the values for the train loss for each epoch
            the values for the train accuracy for each epoch

        """
    criterion =  nn.CrossEntropyLoss(ignore_index=19)
    #criterion_acdc = nn.CrossEntropyLoss(ignore_index=255)


    losses_values = []
    #train_acc_values = []
    l1_values = []
    train_ce = []
    
    for epoch in range(1, num_epochs+1):
        print(f'Epoch----{epoch}/{num_epochs}')
        loss_train, L1_values, train_ce_loss = train(model,phyfea, loader_train,optimizer,criterion)   #accuracy_train
        #loss_val, accuracy_val = validate(model, loader_val, device, criterion)

        # if loss_train <= 0.1:
        #   break

        losses_values.append(loss_train)
        l1_values.append(L1_values)
        train_ce.append(train_ce_loss)
       
        scheduler.step()
       
        torch.save(model.state_dict(), f'/cluster/work/cvl/shbasu/phyfeaSegformer/data/physicsFormer_segformer_cityscape_{epoch}.pth')
        #wandb.log({'epoch': epoch})
        
                  

  
    wandb.finish()
    return {'loss_values': losses_values,
            'L1_values': l1_values,
            'train_ce_loss': train_ce}
            






def main():
    wandb.init(project="PhyFea")
    lr =  0.00006
    fix_random(42)

    weight_decay=0.01
    momentum = 0.9
    num_epochs = 5
    num_classes = 19

    config_file = '/cluster/work/cvl/shbasu/phyfeaSegformer/data/segformer_mit-b3_8xb1-160k_cityscapes-1024x1024.py'  #/content/mmsegmentation/ocrnet_hr18_4xb2-40k_cityscapes-512x1024.py'
    checkpoint_file =  '/cluster/work/cvl/shbasu/phyfeaSegformer/data/segformer_mit-b3_8x1_1024x1024_160k_cityscapes_20211206_224823-a8f8a177.pth' #'/content/mmsegmentation/ocrnet_hr18_512x1024_40k_cityscapes_20200601_033320-401c5bdd.pth'
    # build the model from a config file and a checkpoint file
    model = init_model(config_file, checkpoint_file, device='cuda:0')



    
    physicsFormer = PhysicsFormer(num_classes).to('cuda:1')

    if os.path.exists('/cluster/work/cvl/shbasu/phyfeaSegformer/data/physicsFormer_segformer_cityscape_5.pth'):
        model.load_state_dict(torch.load('/cluster/work/cvl/shbasu/phyfeaSegformer/data/physicsFormer_segformer_cityscape_5.pth'))
        print(f'******model_loaded***********')


    #*** IMPORTANT: remember to create the optimizer AFTER you have moved the model to the GPU, the tensors storing the parameter are not the same used by the original model ***
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer,total_iters=10, power=1.0, last_epoch=- 1, verbose=False)


    statistics = training_loop(num_epochs, optimizer,scheduler,model,physicsFormer, loader_train)

    loss_value = statistics['loss_values']
    #acc_value =  statistics['train_acc_values']
    L1_values =  statistics['L1_values']
    train_ce_loss = statistics['train_ce_loss']
    #mean_max_values = statistics['mean_max_values']


    print(f'loss value: {loss_value}')
    
    print(f'L1 loss values: {L1_values}')
    print(f'Cross_entropy loss values: {train_ce_loss}')
    
    


if __name__ == "__main__":
    main()
