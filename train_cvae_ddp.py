import torch
import torch.nn as nn

import numpy as np

import os
import datetime

from tqdm import tqdm
from torchvision.utils import save_image, make_grid
import torch.nn.functional as F

import matplotlib.pyplot as plt

from network import CVAE
from dataset import AnisoDataset
from torch.utils.data import DataLoader

from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms

from config import configDict

from torch.utils.tensorboard import SummaryWriter

# DDP imports
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, get_rank, get_world_size

from torch.optim import Adam
from torch_tpc import two_point_correlation


######################################################################
def ddp_setup(rank, world_size):
    '''
    Args:
        rank (int): ID of the process
        world_size (int): Number of processes participating in the DDP training
    '''
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    init_process_group("nccl", rank=rank, world_size=world_size)

def save_checkpoint(state, is_best, checkpoint_dir, best_dir):
    filename = os.path.join(checkpoint_dir, 'checkpoint.pth')
    torch.save(state, filename)
    if is_best:
        torch.save(state, os.path.join(best_dir, 'best_model.pth'))

def main(rank, args):
    ddp_setup(rank, args.world_size)
    torch.cuda.set_device(rank)

    # config
    gpu_id = configDict['gpu_id']
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
    load_model = configDict['load_model']
    
    
    model = CVAE(latent_size = 256).to(rank)
    model = DDP(model, device_ids=[rank])
    
    # load model from checkpoint
    if load_model:
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        checkpoint = torch.load(configDict['load_model_path'], map_location=map_location)
        model.load_state_dict(checkpoint['state_dict'])
        print('Loaded model from checkpoint')
    
    

    if rank == 0:
        # directories
        # create models directory
        if not os.path.exists('models'):
            os.makedirs('models')
        
        save_dir = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_' + save_dir
        save_dir = os.path.join('models', save_dir)

        # create save directory inside models directory
        if not os.path.exists(save_dir):
            # create save directory with name date_time_save_dir
            os.makedirs(save_dir)

        # create checkpoints directory in save directory
        if not os.path.exists(os.path.join(save_dir, 'checkpoints')):
            os.makedirs(os.path.join(save_dir, 'checkpoints'))

        # create best directory in save directory
        if not os.path.exists(os.path.join(save_dir, 'best')):
            os.makedirs(os.path.join(save_dir, 'best'))

        # create checkpoints directory in save directory
        if not os.path.exists(os.path.join(save_dir, 'checkpoints')):
            os.makedirs(os.path.join(save_dir, 'checkpoints'))

        # create best directory in save directory
        if not os.path.exists(os.path.join(save_dir, 'best')):
            os.makedirs(os.path.join(save_dir, 'best'))

        # write config file to save directory 
        with open(os.path.join(save_dir, 'config.txt'), 'w') as f:
            for key, value in configDict.items():
                f.write(f'{key}: {value}\n')

    # Initialize TensorBoard writer only on the master process
    if rank == 0:
        writer = SummaryWriter(os.path.join(save_dir, 'logs'))
    # dataset
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
        'train': AnisoDataset(root_dir, 'train', transforms=transform['train']),
        'val': AnisoDataset(root_dir, 'val', transforms=transform['val']),
    }

    # DataLoaders with DistributedSampler
    train_sampler = DistributedSampler(dataset['train'])
    val_sampler = DistributedSampler(dataset['val'])

    dataloader = {
        'train': DataLoader(dataset['train'], batch_size=batch_size, sampler = train_sampler, shuffle=False),
        'val': DataLoader(dataset['val'], batch_size=batch_size, sampler = val_sampler, shuffle=False)
    }

    # model 
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
        
    # loss function
    def loss_function(recon_x, x, mu, logvar):
        reconstruction_loss = F.mse_loss(recon_x, x, reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return reconstruction_loss + KLD_loss, reconstruction_loss, KLD_loss

    # training loop
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        recon_loss_train = 0
        KLD_loss_train = 0

        for batch in dataloader['train']:
            batch = batch.to(rank)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(batch)
            loss, recon_loss, KLD_loss = loss_function(recon_batch, batch, mu, logvar)
            recon_loss_train += recon_loss.item()
            KLD_loss_train += KLD_loss.item()
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        
        # Validation
        model.eval()
        val_loss = 0
        best_val_loss = np.inf
        with torch.no_grad():
            for batch in tqdm(dataloader['val']):
                batch = batch.to(rank)
                recon_batch, mu, logvar = model(batch)
                loss, recon_loss, KLD_loss = loss_function(recon_batch, batch, mu, logvar)
                val_loss += loss.item()
        
        # Logging
        if rank == 0:
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
                'state_dict': model.module.state_dict(),
                'best_val_loss': best_val_loss,
                # 'optimizer': optimizer.state_dict(),
            }

            # Save checkpoint
            print(save_dir)
            save_checkpoint(checkpoint, is_best, os.path.join(save_dir, 'checkpoints'), os.path.join(save_dir, 'best'))
            
            # display some generated images on tensorboard every 10 epochs
            if epoch % 5 == 0:
                with torch.no_grad():
                    sample = torch.randn(8, 256).to(rank)
                    sample = model.module.decode(sample).cpu()
                    img_grid = make_grid(sample)
                    writer.add_image('Generated Images', img_grid, epoch)
                    # save_image(sample, os.path.join(save_dir, f'{epoch}.png'))
                    # display some original images on tensorboard
                    sample = next(iter(dataloader['val']))
                    # take first 8 images
                    sample = sample[:8]
                    sample = sample.to(rank)
                    img_grid = make_grid(sample)
                    writer.add_image('Original Images', img_grid)
                    
    
                
        # Update scheduler
        scheduler.step()
        
        # Print loss
        print(f'Epoch: {epoch} Train Loss: {train_loss} Val Loss: {val_loss}')

            
    if rank == 0:
        writer.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', type=int)
    args = parser.parse_args()

    # config
    gpu_id = configDict['gpu_id']
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
    load_model = configDict['load_model']

    # device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

    # if device == 'cpu':
    #     print('Using CPU')


    try:
        torch.multiprocessing.spawn(main, args=(args,), nprocs=args.world_size, join=True)
    except KeyboardInterrupt:
        print('Keyboard Interrupted')
        try:
            dist.destroy_process_group()
        except:
            os.system("kill $(ps aux | grep multiprocessing.spawn | grep -v grep | awk '{print $2}') ")
