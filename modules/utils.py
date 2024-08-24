import os
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import pandas as pd
from torch.utils.data import TensorDataset
import torchvision.transforms as transforms
import random
import numpy as np 
from torchvision.transforms import ToPILImage
from pathlib import Path

def plot_images(images):
    plt.figure(figsize=(32, 32))
    if images.shape[1]>1:
        plt.imshow(torch.cat([
            torch.cat([i for i in images.cpu()], dim=-1),
        ], dim=-2).permute(1, 2, 0).cpu())
    elif images.shape[1]==1:
        plt.imshow(torch.cat([
            torch.cat([i.squeeze(0) for i in images.cpu()], dim=-1),
        ], dim=-2).cpu(),
        cmap='gray'
        )
    else:
        print('Check inputs. Size should be: n x chan x height x width')
    plt.show()


def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)


def get_data(args):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(args.image_size),  # args.image_size + 1/4 *args.image_size
        #torchvision.transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = torchvision.datasets.ImageFolder(args.dataset_path, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader, dataset


def get_data_MNIST(args):
    # Load the MNIST small dataset from CSV
    data = pd.read_csv(args.dataset_path)

    # Separate features and labels
    labels = torch.tensor(data.iloc[:, 0].values, dtype=torch.long)
    features = torch.tensor(data.iloc[:, 1:].values / 255.0, dtype=torch.float32).view(-1, 28, 28)
    # Add a channel dimension (C=1) to the features
    features = features.unsqueeze(1)  # Now size is [batch_size, 1, 28, 28]
    # Define the transforms
    transform = transforms.Compose([
        # Resizes images
        transforms.Resize(32),  # args.image_size + 1/4 *args.image_size
        # Randomly crops images to args.image_size, with a scaling factor between 0.8 and 1.0.
        #transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
        transforms.Normalize((0.5,), (0.5,))  # Normalize with mean and std deviation for MNIST (single channel)
    ])

    # Apply transforms to the dataset
    features = torch.stack([transform(image) for image in features])

    # Create a TensorDataset
    dataset = TensorDataset(features, labels)

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    return dataloader, dataset

def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)

def plot_loss(loss_all):
    plt.figure(figsize=(6, 6))
    epochs_all=np.arange(1,len(loss_all)+1,1)
    plt.plot(epochs_all,loss_all,marker="",label='loss')
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.show()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_dataset_MNIST(path_str,dataset):
    save_dir = Path(path_str)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Initialize transform to convert tensors to PIL images
    to_pil = ToPILImage()

    # Iterate over the dataset and save each image
    for i, (image, label) in enumerate(dataset):
        # Convert tensor to PIL image
        pil_image = to_pil(image)
        
        # Define the image filename
        filename = save_dir / f"image_{i}.png"
        
        # Save the image
        pil_image.save(filename, format="PNG")

        # Optionally, print out the filename to confirm
        print(f"Saved: {filename}")

def save_gen_images(path_str,data,fileno):
    save_dir = Path(path_str)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Initialize transform to convert tensors to PIL images
    to_pil = ToPILImage()

    # Iterate over the dataset and save each image
    for i in range(data.shape[0]):
        if data.shape[1]==1: #gray images
            image=data[i,:,:,:].squeeze(0).cpu()#.numpy()
        else: # RGB images
            image=data[i,:,:,:].cpu()
        # Convert tensor to PIL image
        pil_image = to_pil(image)
        
        # Define the image filename
        filename = save_dir / f"image_{fileno[i]}.png"
        
        # Save the image
        pil_image.save(filename, format="PNG")

        # Optionally, print out the filename to confirm
        print(f"Saved: {filename}")

def image_data(x):
        if x.shape[1]>1:
            return x.squeeze().permute(1, 2, 0).cpu()#.numpy()
        elif x.shape[1]==1:
            return x.squeeze().cpu()#.numpy()
        else:
            return None