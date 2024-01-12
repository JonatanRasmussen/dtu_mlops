# Filename: data.py
import os
import torch

def mnist():
    # Assuming the 'data' directory is in 'dtu_mlops',
    # and you are running the script from 'dtu_mlops' directory
    base_dir = os.path.join('data', 'corruptmnist')
    train_images = []
    train_targets = []
    print(os.path.join(base_dir, 'test_images.pt'))
    for i in range(6):
        # Adjust the file names if they are different
        image_file = f'train_images_{i}.pt'
        target_file = f'train_target_{i}.pt'
        train_images.append(torch.load(os.path.join(base_dir, image_file)))
        train_targets.append(torch.load(os.path.join(base_dir, target_file)))
    train_images = torch.cat(train_images)
    train_targets = torch.cat(train_targets)

# Combine all parts of the test data and targets
    test_images = torch.cat([torch.load(os.path.join(base_dir, 'test_images.pt'))])
    test_targets = torch.cat([torch.load(os.path.join(base_dir, 'test_target.pt'))])

    return (train_images, train_targets), (test_images, test_targets)
