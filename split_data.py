import os
import numpy as np

def split_data(root_dir, train_ratio=0.7, val_ratio=0.2):
    # Get all file names
    filenames = os.listdir(root_dir)
    filenames = [name for name in filenames if name.endswith('.png')]

    # Shuffle filenames
    np.random.shuffle(filenames)

    # Calculate split sizes
    total_files = len(filenames)
    train_size = int(total_files * train_ratio)
    val_size = int(total_files * val_ratio)

    # Split data
    train_files = filenames[:train_size]
    val_files = filenames[train_size:train_size + val_size]
    test_files = filenames[train_size + val_size:]

    return train_files, val_files, test_files

def write_to_file(filenames, filename):
    with open(filename, 'w') as f:
        for name in filenames:
            f.write(name + '\n')

root_dir = 'data/isotropic_true/'

train_ratio = 0.05
val_ratio = 0.05

train_files, val_files, test_files = split_data(root_dir, train_ratio=train_ratio, val_ratio=val_ratio)

write_to_file(train_files, root_dir + 'train_{:d}_TrainSamples.txt'.format(len(train_files)))
write_to_file(val_files, root_dir + 'val_{:d}_TrainSamples.txt'.format(len(train_files)))
write_to_file(test_files, root_dir + 'test_{:d}_TrainSamples.txt'.format(len(train_files)))