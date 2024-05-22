import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader,Subset,Dataset
from PIL import Image
from tqdm import tqdm
import datetime
import os
import torch.nn.utils as utils
from unet import UnetConditional 
from util_funcs import forward_cosine_noise, reverse_diffusion_cfg, count_parameters,reverse_diffusion
import random


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None, subset_size=None):
        self.root_dir = root_dir
        self.transform = transform
        # Filter out non-image files
        self.image_files = [file for file in os.listdir(root_dir) if file.endswith('.jpg')]
        
        # If subset_size is specified and less than the total number of images, sample a subset
        if subset_size is not None and subset_size < len(self.image_files):
            self.image_files = random.sample(self.image_files, subset_size)
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        try:
            image = Image.open(img_name).convert('RGB')  # Ensure images are loaded with 3 channels
        except Exception as e:
            #print(f"Error loading image '{img_name}': {e}")
            return self.__getitem__((idx + 1) % len(self))  # Return next item
        
        # Check if image loading failed
        if image is None:
            return self.__getitem__((idx + 1) % len(self))  # Return next item
        
        # Load associated text file
        txt_name = img_name.replace('.jpg', '.txt')
        try:
            with open(txt_name, 'r') as f:
                text_data = int(f.read())  # Convert text data to integer
        except Exception as e:
            #print(f"Error loading text '{txt_name}': {e}")
            return self.__getitem__((idx + 1) % len(self))  # Return next item
        
        if self.transform:
            image = self.transform(image)
        
        return image, text_data


def training_loop(n_epochs, optimizer, model, loss_fn, device, data_loader, max_grad_norm=1.0, timesteps=200, epoch_start = 0):
    model.train()
    for epoch in range(epoch_start,n_epochs+epoch_start):
        loss_train = 0.0

        progress_bar = tqdm(data_loader, desc=f'Epoch {epoch}', unit=' batch')
        for imgs, labels in progress_bar:
            #print(labels)
            imgs = imgs.to(device)
            labels = labels.to(device)
            # Generate timestamps
            t = torch.randint(0, timesteps, (imgs.size(0),), dtype=torch.float32).to(device) / timesteps
            t = t.view(-1, 1)
            
            imgs, noise = forward_cosine_noise(None, imgs, t,device='mps')
            
            '''
            if np.random.random() <= 0.1:
                outputs = model(imgs, t)
            else:
                outputs = model(imgs, t,labels)
            '''
            outputs = model(imgs, t)
            loss = loss_fn(outputs, noise)
            
            optimizer.zero_grad()
            loss.backward()
            utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            loss_train += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        # Save model checkpoint with the current epoch in the filename
        model_filename = f'number-diffusion-cts_epoch_{epoch}_cfg.pth'
        model_path = os.path.join('/Users/ayanfe/Documents/Code/Diffusion-Model/weights', model_filename)
        
        with open("number-diffusion-loss.txt", "a") as file:
            file.write(f"{loss_train / len(data_loader)}\n")
        
        print('{} Epoch {}, Training loss {}'.format(datetime.datetime.now(), epoch, loss_train / len(data_loader)))
        if epoch % 5 == 0:
            #reverse_diffusion_cfg(model,30,torch.tensor([[4]],dtype=torch.int32),5,size=(32,32))
            reverse_diffusion(model,30)
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, model_path)


if __name__ == "__main__":
    timesteps = 1000

    device = torch.device("mps")
    model = UnetConditional()  # Assuming Unet is correctly imported and defined
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    #loss_fn = nn.L1Loss().to(device)
    loss_fn = nn.MSELoss().to(device)
    print(count_parameters(model))
    
    transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
    ])
    dataset = CustomDataset(root_dir='/Users/ayanfe/Documents/Datasets/MNIST Labled', transform=transform, subset_size= 30_000)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, pin_memory=True)

    # Optionally load model weights if needed
    #model_path = "/Users/ayanfe/Documents/Code/Diffusion-Model/weights/number-diffusion-cts_epoch_30_cfg.pth"
    #checkpoint = torch.load(model_path)
    #model.load_state_dict(checkpoint['model_state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #epoch = checkpoint['epoch']
    reverse_diffusion_cfg(model,30,torch.tensor([[0]],dtype=torch.int32),3,size=(32,32))
    
    training_loop(
        n_epochs=1000,
        optimizer=optimizer,
        model=model,
        loss_fn=loss_fn,
        device=device,
        data_loader=dataloader,
        timesteps=timesteps,
        epoch_start= 1
    )
    
