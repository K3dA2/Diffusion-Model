import numpy as np
import torch
import matplotlib.pyplot as plt
import uuid
import os
import cv2

timesteps = 1000
def save_img(img):
    plt.imshow(np.squeeze(img[-1]))
    plt.axis('off')  # If you want to hide the axes
    # Generate a random filename
    random_filename = str(uuid.uuid4()) + '.png'

    # Specify the directory where you want to save the image
    save_directory = 'Noise/'

    # Create the full path including the directory and filename
    full_path = os.path.join(save_directory, random_filename)
    # Save the image with the random filename
    plt.savefig(full_path, bbox_inches='tight', pad_inches=0)

def cosine(t):
    min_signal_rate = 0.02
    max_signal_rate = 0.95
    if torch.is_tensor(t):
        t /= timesteps
    else:
        t = t.astype(float)  # Convert t to float
        t /= timesteps
    start_angle = np.arccos(max_signal_rate)
    end_angle = np.arccos(min_signal_rate)
    diffusion_angles = start_angle + t * (end_angle - start_angle)
    signal_rates = np.cos(diffusion_angles)
    noise_rates = np.sin(diffusion_angles)
    return noise_rates, signal_rates

def set_key(key):
    np.random.seed(key)

def generate_timestamp(key, num):
    set_key(key)
    return torch.randint(0, timesteps,(num,), dtype=torch.int32)

def forward_cosine_noise(key, x_0, t):
    set_key(key)
    noise = np.random.normal(size=x_0.shape)
    noise_rates,signal_rates = cosine(t)
    reshaped_noise_rates = np.reshape(noise_rates, (-1, 1, 1, 1))
    reshaped_signal_rates = np.reshape(signal_rates, (-1, 1, 1, 1))
    noisy_image = reshaped_signal_rates  * ((x_0 - 127.5)/127.5) + reshaped_noise_rates * noise
    return noisy_image, noise

def reverse_diffusion(noisy_im,noise, diffusion_steps, device='mps'):
    step_size = 1.0 / diffusion_steps
    current_images = noisy_im[0]
    
    for step in range(diffusion_steps):
        diffusion_times = torch.ones((1, 1)) - step * step_size 
        indices = int(diffusion_times*timesteps)-1
        #print(diffusion_times)
        
        # Ensure model and other  operations are also moved to the device
        #print(indices)
        pred_noises = noise[indices]
        noise_rates, signal_rates = cosine((diffusion_times * 1000))
        
        pred_noises = pred_noises  
        noise_rates = noise_rates  
        signal_rates = signal_rates  
        
        pred_images = (current_images - noise_rates * pred_noises) / signal_rates
        next_diffusion_times = diffusion_times - step_size
        next_noise_rates, next_signal_rates = cosine(next_diffusion_times*1000)
        
        next_noise_rates = next_noise_rates  
        next_signal_rates = next_signal_rates  
        
        current_images = (next_signal_rates * pred_images + next_noise_rates * pred_noises)

    pred_images = (pred_images.clamp(-1, 1) + 1) / 2
    
    # Detach the tensor before converting it to a NumPy array
    plt.imshow(pred_images[-1].cpu().numpy())
    plt.show()

if __name__ == "__main__":
    test_img_path = "120b40c149fcd761cfc4f5ef4225c9aa.jpg"
    sample_data = plt.imread(test_img_path)
    sample_data = cv2.resize(sample_data,(64,64))
    noisy_im = []
    noise = []
    for i in range(timesteps):
        img, noi = forward_cosine_noise(0,np.expand_dims(sample_data, 0),np.array([i,]))
        noisy_im.append(img)
        noise.append(noi)
        #save_img(noise)

    noisy_im = list(reversed(noisy_im))
    noise = list(reversed(noise))

    noisy_im = torch.from_numpy(np.array(noisy_im))
    noise = torch.from_numpy(np.array(noise))

    reverse_diffusion(noisy_im,noise,30)
    
