import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import datetime
import random
import torch.nn.utils as utils
from util_funcs import get_data, reshape_img, generate_timestamp,forward_cosine_noise, reverse_diffusion,count_parameters
from unet import Unet

def training_loop(n_epochs, optimizer, model, loss_fn, device, accumulation_steps=1, 
                  epoch_start = 0, batch_size = 64, max_grad_norm=1.0, repeats = 5, timesteps = 200):
    start = 1
    img_length = len(image_names)
    data_length = 4200
    data_idx = list(range(0,data_length+32))
    random.shuffle(data_idx)
    with torch.autograd.set_detect_anomaly(True):
        for epoch in range(epoch_start, n_epochs + 1):
            loss_train = 0.0
            accumulated_loss = 0.0
            loss_mae = 0.0
            random.shuffle(data_idx)
            # Use tqdm function for the progress bar
            with tqdm(range(start, (repeats*data_length)//batch_size), desc=f'Epoch {epoch}', unit=' steps') as pbar:
                st = 0
                for x in pbar:
                    # Training loop code
                    sp = st + batch_size
                    if sp > data_length:
                        st = 0
                        sp = st + batch_size
                    img_arr = []
                    for i in range(st,sp):
                        
                        img = plt.imread(path + '/' + image_names[data_idx[i]])
                        img = reshape_img(img,size=(32,32),greyscale=False)
                        img = np.expand_dims(img, 0)
                        img_arr.append(img)
                    st+= batch_size
                    
                    t = generate_timestamp(0, batch_size)
                    t = torch.reshape(t, (-1, 1)).type(torch.float32)
                    imgs, noise = forward_cosine_noise(None, np.squeeze(np.array(img_arr)), t)
                    
                    if torch.is_tensor(imgs):
                        imgs = imgs.type(torch.float32).to(device)
                    else:
                        imgs = torch.from_numpy(imgs).type(torch.float32).to(device)
                    noise = torch.from_numpy(noise).type(torch.float32).to(device)
                    t = t.to(device)
                    t /= timesteps

                    outputs = model(imgs, t)
                    
                    loss = loss_fn(outputs, noise)
                    
                    # Perform gradient accumulation
                    accumulated_loss += loss / accumulation_steps
                    
                    if x % accumulation_steps == 0:
                        accumulated_loss.backward()

                        # Clip gradients
                        utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                        
                        optimizer.step()
                        optimizer.zero_grad()
                        accumulated_loss = 0.0  # Reset accumulated loss
                    
                    loss_train += loss.item()
                    outputs.detach_()
                    pbar.set_postfix(loss=loss.item())
                
            avg_loss_epoch = loss_train / ((repeats*data_length)//batch_size)
            with open("mnist-diffusion-n-cts_1000-4000-loss.txt", "a") as file:
                file.write(f"{avg_loss_epoch}\n")
            
            print('{} Epoch {}, Training loss {}'.format(
                datetime.datetime.now(), epoch,
                loss_train / ((repeats*data_length)//batch_size)))
            #torch.save(model.state_dict(), path1)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, model_path)
            #inference(model, device)
            if epoch % 5 == 0:
                #inference(model, device)
                reverse_diffusion(model,50)
            '''
            if epoch % 20 == 0 and timesteps <= 1000:
                timesteps += 100
            '''
                


if __name__ == "__main__":
    timesteps = 1000
    path = '/Users/ayanfe/Documents/Datasets/MNIST Upscaled/upscayl_png_realesrgan-x4plus_4x'
    model_path = '/Users/ayanfe/Documents/Code/Diffusion-Model/weights/mnist-diffusion-n-cts_1000-4000.pth'
    image_names = get_data(path)
    print("Image Length: ",len(image_names))

    model = Unet(greyscale=False)
    #model.load_state_dict(torch.load(model_two))
    device = torch.device("mps")
    model.to(device)
    optimizer = optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)  #  <3>
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print("Total parameters: ",count_parameters(model))
    #loss_fn = nn.MSELoss()  #  <4>
    loss_fn = nn.L1Loss()

    training_loop(  # <5>
        n_epochs = 1000,
        optimizer = optimizer,
        model = model,
        loss_fn = loss_fn,
        device = device,
        batch_size = 32,
        epoch_start = epoch,
        timesteps = timesteps
    )