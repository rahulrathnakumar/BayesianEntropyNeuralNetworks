import torch
import torch.nn as nn

import numpy as np

import os
import datetime

from tqdm import tqdm
from torchvision.utils import save_image, make_grid
import torch.nn.functional as F

import matplotlib.pyplot as plt

from network import MLP, BaseCVAE
from dataset import AnisoDataset
from torch.utils.data import DataLoader

from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR, CyclicLR
from torchvision import transforms

from config import configDict

from torch.utils.tensorboard import SummaryWriter

# DDP imports
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, get_rank, get_world_size

from torch.optim import Adam
from torch_tpc import two_point_correlation, porosity


def save_checkpoint(state, is_best, checkpoint_dir, best_dir):
    filename = os.path.join(checkpoint_dir, 'checkpoint.pth')
    torch.save(state, filename)
    if is_best:
        torch.save(state, os.path.join(best_dir, 'best_model.pth'))


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
load_model = configDict['load_model']

N = 3

device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

model = BaseCVAE(latent_size = 256).to(gpu_id)

# load model from checkpoint
if load_model:
    map_location = {'cuda:%d' % 0: 'cuda:%d' % gpu_id}
    checkpoint = torch.load(configDict['load_model_path'], map_location=map_location)
    model.load_state_dict(checkpoint['state_dict'])
    print('Loaded model from checkpoint')



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
writer = SummaryWriter(os.path.join(save_dir, 'logs'))
# dataset
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

# DataLoaders 
dataloader = {
    'train': DataLoader(dataset['train'], batch_size=batch_size, shuffle=True),
    'val': DataLoader(dataset['val'], batch_size=batch_size, shuffle=False)
}

# optimizer for the main model
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
    # reconstruction_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    # determine if input is binary or continuous
    if len(x.unique()) == 2:
        reconstruction_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    else:
        # continuous
        reconstruction_loss = F.mse_loss(recon_x, x, reduction='sum')
    
    # TODO: write a test for above if-else statement 
    

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return reconstruction_loss + KLD_loss, reconstruction_loss, KLD_loss

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

def plot_histogram(x, bins=100):
    plt.hist(x, bins=bins)
    plt.savefig('histogram.png')
    plt.close()

smooth_l1_loss = nn.SmoothL1Loss(reduction='sum', 
                                 beta = 0.2
                                )

# do the above, but this time compute porosity
def compute_porosity(x, gpu_id):
    '''
    Compute porosity of a binary image
    '''
    diff_bin_x = diff_binarize(x, threshold=0.5, steepness=100)
    # assert that diff_bin_x is not on cpu
    assert(diff_bin_x.device.index == gpu_id)
    return porosity(diff_bin_x)

def s2r(x, gpu_id):
    '''
    2-point spatial correlation function for a distance array r
    TODO: r not needed - remove it.
    '''
    diff_bin_x = diff_binarize(x, threshold=0.5, steepness=1000)
    res = two_point_correlation(diff_bin_x, ddp = False, rank = gpu_id)
    tpc = res.probability
    dist = res.distance
    return tpc, dist

def value_constraint_loss(eta, N, model, gpu_id, y_test, funcs, *args, **kwargs):
    '''
    Given a list of functions, compute the constraint loss
    '''
    # CHANGE to generalize to n funcs made on 01/19/2024
    # removed N evaluations of funcs
    cons_loss = 0.0
    sample = torch.randn(8, 256).to(device)
    sample = model.decode(sample)
    # reshape from [1,1,256,256] to [256,256]
    sample = sample.view(8, 256, 256)
    for i, func in enumerate(funcs):
        result = func(sample, gpu_id)
        if isinstance(result, tuple):
            g = result[0] # for the s2r function
        else:
            g = result # for the porosity function
        # I1 = g*torch.exp(eta[i]*g)
        # I2 = torch.exp(eta[i]*g)
        # y_hat = I1/I2
        cons_loss += smooth_l1_loss(g, y_test[i])
    return cons_loss


# eta = torch.tensor(0.9, requires_grad = True, device = device)

# Compute the average of the tpc of the training data for constraint
# load training data using dataloader
for batch in dataloader['train']:
    batch = batch.to(device)
    batch = batch.view(batch.shape[0], 256, 256)
    tpc_gt, _ = s2r(batch, gpu_id)
    porosity_gt = compute_porosity(batch, gpu_id)
    y_test = [tpc_gt, porosity_gt]
    break

# training loop
# lam needs to cycle from 1 to 0 every n epochs
ramp_length = 10
lam = 1.0
epoch__ = 0
def one_cycle_ramp(epoch, k = 1, ramp_length = 50):
    '''
    Ramp function for one cycle
    '''
    if epoch < ramp_length or torch.sigmoid(torch.tensor(k*(1 - epoch/ramp_length))) > 0.2:
        return torch.sigmoid(torch.tensor(k*(1 - epoch/ramp_length)))
    else:
        return torch.tensor(0.2)

def cyclic_ramp(epoch, ramp_length = 50):
    '''
    Ramp function for one cycle - resets to 1 after ramp_length epochs
    '''
    if epoch < ramp_length:
        return torch.sigmoid(torch.tensor(1 - epoch/ramp_length))

constraint_functions = [s2r, compute_porosity]

mlp = MLP(num_constraints = len(constraint_functions)).to(gpu_id)

# eta = torch.tensor(1.0, requires_grad = True, device = gpu_id)
# initial input to mlp 
eta = torch.tensor([1.0]*len(constraint_functions), device = gpu_id)
assert len(eta) == len(constraint_functions)

# optimizer and scheduler for mlp - SGD w/ one cycle lr
optimizer_eta = SGD(mlp.parameters(), lr=1e-2, momentum=0.9)
# optimizer_eta = SGD([eta], lr=1e-2, momentum=0.9) # eta is a list of tensors
scheduler_eta = CyclicLR(
    optimizer_eta, base_lr = 1e-2, max_lr = 1e-1, step_size_up = 30,
    step_size_down = 30, mode = 'triangular2'
)


for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0
    recon_loss_train = 0
    KLD_loss_train = 0
    constraint_loss_train = 0
    for batch in dataloader['train']:
        batch = batch.to(device)
        
        # zero out gradients for both optimizers
        optimizer.zero_grad()
        optimizer_eta.zero_grad()
        
        # recon_batch, mu, logvar, eta = model(batch)
        recon_batch, mu, logvar = model(batch)
        eta_ = mlp(eta)
        eta = eta_.detach().clone()
        # eta = torch.clamp(torch.abs(torch.mean(eta)), 0, 0.5)
        loss, recon_loss, KLD_loss = loss_function(recon_batch, batch, mu, logvar)
        constraint_loss = value_constraint_loss(eta_, N, model, gpu_id, y_test, funcs = constraint_functions)
        total_loss = KLD_loss + ((1-lam)*recon_loss + (lam)*constraint_loss)
        
        recon_loss_train += recon_loss.item()
        KLD_loss_train += KLD_loss.item()
        constraint_loss_train += constraint_loss.item()
        
        total_loss.backward()
        train_loss += total_loss.item()
        
        # step both optimizers
        optimizer.step()
        optimizer_eta.step()
        
        # step scheduler for eta
        scheduler_eta.step()
        
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
    
    # average constraint loss over all batches 
    
    print("Logging to TensorBoard...")
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/train/recon', recon_loss_train, epoch)
    writer.add_scalar('Loss/train/KLD', KLD_loss_train, epoch)
    
    # writer.add_scalar('eta', eta, epoch)
    for i, e in enumerate(eta):
        # eta_{:} where {:} is the name of the constraint_function
        writer.add_scalar(f'eta_{constraint_functions[i].__name__}', e, epoch)
        
    writer.add_scalar('Loss/constraint', constraint_loss_train, epoch)
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

    # Save checkpoint
    print(save_dir)
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
            # save the generated images with name {epoch}.png

            
    # Update scheduler for vae model
    scheduler.step()
    # check for reset of lam
    if epoch % ramp_length == 0:
        lam = 1.0
        epoch__ = 0
        
    lam = cyclic_ramp(epoch__, ramp_length=ramp_length)
    epoch__ += 1 # epoch__ is a dummy variable to keep track of the number of epochs since lam was reset
    # Print loss
    print(f'Epoch: {epoch} Train Loss: {train_loss} Constraint Loss: {constraint_loss_train} Val Loss: {val_loss} Eta {eta} Lam {lam} Ramp_epoch {epoch__}')