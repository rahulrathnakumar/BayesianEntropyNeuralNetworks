import torch
import torch.nn as nn

import numpy as np

import os
import datetime

from tqdm import tqdm
from torchvision.utils import save_image, make_grid
import torch.nn.functional as F

import matplotlib.pyplot as plt

from network import BaseCVAE
from dataset import AnisoDataset
from torch.utils.data import DataLoader

from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms

from config import configDict

from torch.utils.tensorboard import SummaryWriter

from torch_tpc import two_point_correlation, porosity

# config
gpu_id = configDict['gpu_id']
num_training_samples = configDict['num_training_samples']
is_binary = configDict['is_binary']
batch_size = configDict['batch_size']
num_epochs = configDict['num_epochs']
optimizer = configDict['optimizer']
scheduler = configDict['scheduler']
lr = configDict['lr']
lr_step_size = configDict['lr_step_size']
lr_gamma = configDict['lr_gamma']
latent_dim = configDict['latent_dim']
root_dir = configDict['root_dir']
save_dir = configDict['save_dir']

device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

if device == 'cpu':
    print('Using CPU')


def save_checkpoint(state, is_best, checkpoint_dir, best_dir):
    filename = os.path.join(checkpoint_dir, 'checkpoint.pth')
    torch.save(state, filename)
    if is_best:
        torch.save(state, os.path.join(best_dir, 'best_model.pth'))


# directories
# create models directory
if not os.path.exists('models'):
    os.makedirs('models')

# create save directory inside models directory
if not os.path.exists(save_dir):
    # create save directory with name date_time_save_dir
    save_dir = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_' + save_dir
    save_dir = os.path.join('models', save_dir)
    os.makedirs(save_dir)

# create checkpoints directory in save directory
if not os.path.exists(os.path.join(save_dir, 'checkpoints')):
    os.makedirs(os.path.join(save_dir, 'checkpoints'))

# create best directory in save directory
if not os.path.exists(os.path.join(save_dir, 'best')):
    os.makedirs(os.path.join(save_dir, 'best'))

writer = SummaryWriter(os.path.join(save_dir, 'logs'))

# write config file to save directory 
with open(os.path.join(save_dir, 'config.txt'), 'w') as f:
    for key, value in configDict.items():
        f.write(f'{key}: {value}\n')


# dataset
if is_binary:
    transform = {
        'train': transforms.Compose([
        transforms.Resize((256, 256), transforms.InterpolationMode.NEAREST),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),                                        
        transforms.ToTensor(),
    ]),
        'val': transforms.Compose([
        transforms.Resize((256, 256), transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])
    }
else:
    transform = {
        'train': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),                                        
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.732], std=[0.442])
    ]),
        'val': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.732], std=[0.442])
    ])
    }

dataset = {
    'train': AnisoDataset(root_dir, 'train', transforms=transform['train'], 
                          num_training_samples=num_training_samples),
    'val': AnisoDataset(root_dir, 'val', transforms=transform['val'],
                        num_training_samples=num_training_samples),
}
dataloader = {
    'train': DataLoader(dataset['train'], batch_size=batch_size, shuffle=True),
    'val': DataLoader(dataset['val'], batch_size=batch_size, shuffle=False)
}

# model 
model = BaseCVAE(latent_size = 256)
model = model.to(device)

# optimizer
if optimizer == 'Adam':
    optimizer = Adam(model.parameters(), lr=lr)
else:
    ValueError('Optimizer not supported')

# scheduler
if scheduler == 'StepLR':
    scheduler = StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)
else:
    ValueError('Scheduler not supported')

def binarize(x):
    return (x > 0.5).float()

def normalize(x):
    '''
    x: Tensor of shape (N, C, H, W)
    This function normalizes each sample in the batch between 0 and 1
    '''
    min_vals = torch.min(x.view(x.shape[0], -1), dim=1, keepdim=True)[0].view(x.shape[0], 1, 1, 1)
    max_vals = torch.max(x.view(x.shape[0], -1), dim=1, keepdim=True)[0].view(x.shape[0], 1, 1, 1)

    return (x - min_vals) / (max_vals - min_vals)

    
def diff_binarize(x, threshold = 0.5, steepness = 100):
    x_ = steepness * (x - threshold)
    return torch.sigmoid(x_)

def s2r(x, gpu_id):
    '''
    2-point spatial correlation function for a distance array r
    '''
    # shape 
    # x = x.view(-1, 256, 256) # TODO: generalize to arbitrary image size
    # bin_x = (x > 0.5)
    diff_bin_x = diff_binarize(x, threshold=0.5, steepness=100)
    res = two_point_correlation(diff_bin_x, ddp = False, rank = gpu_id)
    tpc = res.probability
    dist = res.distance
    return tpc, dist

# do the above, but this time compute porosity
def compute_porosity(x, gpu_id):
    '''
    Compute porosity of a binary image
    '''
    diff_bin_x = diff_binarize(x, threshold=0.5, steepness=100)
    # assert that diff_bin_x is not on cpu
    assert(diff_bin_x.device.index == gpu_id)
    return porosity(diff_bin_x)

# for i in range(len(dataset['val'])):
#     assert(len(dataset['val'][i].unique()) == 2)

# loss function
def loss_function(recon_x, x, mu, logvar):
    if len(x.unique()) == 2:
        reconstruction_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    else:
        # continuous
        reconstruction_loss = F.mse_loss(recon_x, x, reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return reconstruction_loss + KLD_loss, reconstruction_loss, KLD_loss

# training loop
# Compute the average of the tpc of the training data for constraint
# load training data using dataloader
for batch in dataloader['train']:
    batch = batch.to(device)
    batch = batch.view(batch.shape[0], 256, 256)
    tpc_gt, _ = s2r(batch, gpu_id)
    porosity_gt = compute_porosity(batch, gpu_id)
    y_test = [tpc_gt, porosity_gt]
    break
constraint_functions = [s2r, compute_porosity] # not using it for training
for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0
    recon_loss_train = 0
    KLD_loss_train = 0

    for batch in dataloader['train']:
        batch = batch.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(batch)
        loss, recon_loss, KLD_loss = loss_function(recon_batch, batch, mu, logvar)
        loss.backward()
        optimizer.step()
            
        train_loss += loss.item()
        recon_loss_train += recon_loss.item()
        KLD_loss_train += KLD_loss.item()


    # Validation
    model.eval()
    val_loss = 0
    best_val_loss = np.inf
    with torch.no_grad():
        for batch in tqdm(dataloader['val']):
            batch = batch.to(device)
            recon_batch, mu, logvar = model(batch)
            loss, recon_loss, KLD_loss = loss_function(recon_batch, batch, mu, logvar)
            val_loss += loss.item()
    
    # Logging
    print("Logging to TensorBoard...")
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/train/recon', recon_loss_train, epoch)
    writer.add_scalar('Loss/train/KLD', KLD_loss_train, epoch)

    writer.add_scalar('Loss/val', val_loss, epoch)
    
    is_best = val_loss < best_val_loss
    if is_best:
        best_val_loss = val_loss
        print("Saving best model...")

    # Create checkpoint dict
    checkpoint = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_val_loss': best_val_loss,
        # 'optimizer': optimizer.state_dict(),
    }

    
    # Save model
    save_checkpoint(checkpoint, is_best, os.path.join(save_dir, 'checkpoints'), os.path.join(save_dir, 'best'))
            
    # display some generated images on tensorboard every 10 epochs
    if epoch % 2 == 0:
        with torch.no_grad():
            sample = torch.randn(8, 256).to(device)
            sample = model.decode(sample)
            # sample = sample.cpu()
            img_grid = make_grid(binarize(normalize(sample.view(8, 1, 256, 256))))
            writer.add_image('Generated Images', img_grid, epoch)

            sample = sample.view(8, 256, 256)
            
            for i, func in enumerate(constraint_functions):
                result = func(sample, gpu_id)
                if isinstance(result, tuple):
                    g = result[0] # for the s2r function
                else:
                    g = result # for the porosity function

                l1 = F.smooth_l1_loss(g, y_test[i], reduction='sum') # for displaying averaged vals
                if func == s2r:
                    writer.add_scalar('TPC/L1_summed', l1, epoch)
                else:
                    writer.add_scalar('Porosity/L1_summed', l1, epoch)
                        
            # save_image(sample, os.path.join(save_dir, f'{epoch}.png'))
            # display some original images on tensorboard
            sample = next(iter(dataloader['val']))
            # take first 8 images
            sample = sample[:8]
            sample = sample.to(device)
            img_grid = make_grid(sample)
            writer.add_image('Original Images', img_grid)
            
        
    
            
    # Update scheduler
    scheduler.step()
    
    # Print loss
    print(f'Epoch: {epoch} Train Loss: {train_loss} Val Loss: {val_loss}')
