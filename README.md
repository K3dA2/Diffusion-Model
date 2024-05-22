# Diffusion Model Repository (Unofficially called Waifu Diffusion 🙃)

Welcome to the Diffusion Model Repository! This repository contains implementations of various small diffusion models developed by Ayanfe. Currently, the repository is undergoing training and experimentation, and it serves as a store for implementations of diffusion models.

## About the Diffusion Model

Diffusion models are a class of generative models used for image synthesis and manipulation. They operate by iteratively applying noise to an input image and gradually refining it to generate realistic samples. These models have shown promising results in various computer vision tasks, including image generation, inpainting, and super-resolution.

## Training Status

The diffusion model in this repository is currently undergoing training. The training process is being conducted on a 16' M1 Max MacBook Pro, with occasional usage of Google Colab when feasible (due to limitations of the free tier). As of now, the model is on its 12th epoch of training.

## Repository Structure

The repository is organized as follows:

models/: Where I plan on storing all implemented models eventually.
inference_images/: Images showcasing inference results from the diffusion model.
utils/: Utility functions and scripts used in model training.

## Usage

To utilize the diffusion model implementations in this repository, follow these steps:

Clone the repository to your local machine:

`git clone https://github.com/K3dA2/Diffusion-Model.git`

Install the required dependencies. Ensure that you have Python and the necessary libraries installed:

## Inference Results
Here are some of my favorite results from the old base model :) (Remeber its still training)

![Epoch 4](https://github.com/K3dA2/Diffusion-Model/assets/112480809/ac2894b2-4131-4e84-adc2-8eb34946be1f)
![Epoch 5](https://github.com/K3dA2/Diffusion-Model/assets/112480809/4c4b3687-90ed-49d6-844c-8d51de601b1d)

![Epoch 8](https://github.com/K3dA2/Diffusion-Model/assets/112480809/c29d51d5-5894-4b2b-96ca-4c1b6605d68a)
![Epoch 9](https://github.com/K3dA2/Diffusion-Model/assets/112480809/edb76dae-6a9e-43b1-93eb-ea569153baba)

![Epoch 10](https://github.com/K3dA2/Diffusion-Model/assets/112480809/6edd5735-8562-45e1-ad16-fae71151b0c3)
![Epoch 12](https://github.com/K3dA2/Diffusion-Model/assets/112480809/4a6d08c6-a756-4857-ae05-c13c609bb539)

![Epoch 12](https://github.com/K3dA2/Diffusion-Model/assets/112480809/161e3fa6-b17f-4539-be98-443a7b500283)
![Epoch 12](https://github.com/K3dA2/Diffusion-Model/assets/112480809/7c71469b-b316-4e37-90d6-468dbb1b839d)

## Images From New Model

![fbe4725d-63c9-45c7-8ecd-b979926cc658](https://github.com/K3dA2/Diffusion-Model/assets/112480809/2188003a-a7a5-4476-abf3-806a65f7bb27)
![215f043b-c4ca-4ec2-bf6c-248e453cb4a8](https://github.com/K3dA2/Diffusion-Model/assets/112480809/dafc1dfc-5aab-4b00-a9eb-2588ceac5891)

![7916e7a5-eb58-49b8-b2ce-4ec74b64f4cc](https://github.com/K3dA2/Diffusion-Model/assets/112480809/8c7e90b4-9561-4a5c-8c21-bb96b8f29b33)
![f38b5054-552b-448e-ba46-193a0ffbff8b](https://github.com/K3dA2/Diffusion-Model/assets/112480809/fc2c662d-5ff6-49e0-87dc-bf51399e4de1)


## Dataset
I'm currently using a subset (30k Images) of [this](https://www.kaggle.com/datasets/scribbless/another-anime-face-dataset) Anime Face Dataset. 


## Future Plans

In the future, I plan to continue expanding this repository with implementations of various small diffusion models. However, please note that the models included may not exceed 100 million parameters, considering the compute resources available for training.

## Feedback and Contributions

Feedback, suggestions, and contributions to this repository are highly welcome! If you encounter any issues, have ideas for improvement, or wish to contribute your own implementations of diffusion models, please feel free to open an issue or submit a pull request.

Thank you for visiting the Diffusion Model Repository! Happy modeling!
