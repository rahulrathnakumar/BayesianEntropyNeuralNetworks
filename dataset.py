import torch
import torch.nn as nn

import numpy as np
from PIL import Image

from torch.utils.data import Dataset, DataLoader

import os

class AnisoDataset(Dataset):
    def __init__(self, root_dir, image_set, num_training_samples, transforms):
        super(AnisoDataset, self).__init__()
        self.root_dir = root_dir
        self.image_set = image_set
        self.transform = transforms
        self.read_image_list(root_dir, num_training_samples = num_training_samples)
    
    def read_image_list(self, root_dir, num_training_samples = None):
        if num_training_samples is None:
            with open(os.path.join(root_dir, self.image_set + '.txt')) as f:
                self.image_list = f.read().splitlines()
        else:
            with open(os.path.join(root_dir, self.image_set + '_{:d}_' 'TrainSamples.txt'.format(num_training_samples))) as f:
                self.image_list = f.read().splitlines()

        
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_list[idx])
        image = Image.open(img_name)
        image = image.convert('L')
        
        if self.transform:
            image = self.transform(image)
        
        return image


if __name__ == '__main__':
    dataset = AnisoDataset('data/aniso_data', 'train', transfoms = 
                           torch.Compose
                           ([   torch.Normalize(mean=[0.8306], std=[0.375]),
                                torch.ToTensor()])
                            )
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    for batch in dataloader:
        print(batch.shape)
        break