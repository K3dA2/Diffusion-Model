import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader,Subset
from tqdm import tqdm
import datetime
import os
import torch.nn.utils as utils
from unet import Unet  
from util_funcs import forward_cosine_noise, reverse_diffusion, count_parameters
import random


def get_data_loader(path, batch_size, num_samples=None, shuffle=True):
    # Define your transforms
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.7002, 0.6099, 0.6036), (0.2195, 0.2234, 0.2097))  # Adjust these values if you have RGB images
    ])
    
    # Load the full dataset
    full_dataset = datasets.ImageFolder(root=path, transform=transform)
    
    # If num_samples is not specified, use the entire dataset
    if num_samples is None or num_samples > len(full_dataset):
        num_samples = len(full_dataset)
    print("data length: ",len(full_dataset))
    # Generate a list of indices to sample from (ensure dataset size is not exceeded)
    if shuffle:
        indices = random.sample(range(len(full_dataset)), num_samples)
    else:
        indices = list(range(num_samples))
    
    # Create a subset of the full dataset using the specified indices
    subset_dataset = Subset(full_dataset, indices)
    
    # Create a DataLoader for the subset
    data_loader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=shuffle)
    
    return data_loader


def training_loop(n_epochs, optimizer, model, loss_fn, device, data_loader, max_grad_norm=1.0, timesteps=200, epoch_start = 0):
    model.train()
    for epoch in range(epoch_start,n_epochs):
        loss_train = 0.0

        progress_bar = tqdm(data_loader, desc=f'Epoch {epoch}', unit=' batch')
        for imgs, _ in progress_bar:
            imgs = imgs.to(device)

            # Generate timestamps
            t = torch.randint(0, timesteps, (imgs.size(0),), dtype=torch.float32).to(device) / timesteps
            t = t.view(-1, 1)
            
            imgs, noise = forward_cosine_noise(None, imgs, t,device='mps')
            
            outputs = model(imgs, t)
            loss = loss_fn(outputs, noise)
            
            optimizer.zero_grad()
            loss.backward()
            utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            loss_train += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        # Save model checkpoint with the current epoch in the filename
        model_filename = f'waifu-diffusion-cts_epoch_{epoch}.pth'
        model_path = os.path.join('/Users/ayanfe/Documents/Code/Diffusion-Model/weights', model_filename)
        
        with open("waifu-diffusion-loss.txt", "a") as file:
            file.write(f"{loss_train / len(data_loader)}\n")
        
        print('{} Epoch {}, Training loss {}'.format(datetime.datetime.now(), epoch, loss_train / len(data_loader)))
        if epoch % 5 == 0:
            reverse_diffusion(model,30,size=(64,64))
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, model_path)


if __name__ == "__main__":
    timesteps = 1000
    path = '/Users/ayanfe/Documents/Datasets/Waifus'
    model_path = '/Users/ayanfe/Documents/Code/Diffusion-Model/weights/waifu-diffusion-cts_epoch_80.pth'
    
    device = torch.device("mps")
    model = Unet()  # Assuming Unet is correctly imported and defined
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    #loss_fn = nn.L1Loss().to(device)
    loss_fn = nn.MSELoss().to(device)
    print(count_parameters(model))
    data_loader = get_data_loader(path, batch_size=16,num_samples=24_000)

    # Optionally load model weights if needed
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    reverse_diffusion(model,30,size=(64,64))
    
    
    training_loop(
        n_epochs=1000,
        optimizer=optimizer,
        model=model,
        loss_fn=loss_fn,
        device=device,
        data_loader=data_loader,
        timesteps=timesteps,
        epoch_start= epoch+1
    )
    
