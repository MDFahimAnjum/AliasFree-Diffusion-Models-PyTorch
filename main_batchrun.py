
#%% 
from ddpm_tasks import ddpm_run
import numpy as np
import os

#%%
current_directory = os.getcwd() 

# MNIST
#---------
datapath = os.path.join(current_directory,"data\MNIST\mnist_train_small.csv")
#datapath = os.path.join(current_directory,"data\MNIST-M-6000")
#datapath = os.path.join(current_directory,"data\cifar10-32 selected\CIFAR 10-32-10k")
#datapath = os.path.join(current_directory,"data\CelebA 64X64\local_train")

params={
    'unet_v': 0,
    'epochs': 100,
    'batchsize': 16,
    'image_size': 32,
    'image_channels': 1,
    'device': "cuda",
    'lr': 3e-4,
    'noise_steps': 1000,
    'image_gen_per_epoch': 8,
    'f_kernel': None,
    'f_beta': None,
    'f_down': None,
    'f_up': None,
    'save_trining': False,
    'gen_per_batch': 200,
    'gen_total': 2000,
    'seed': 42,
    'collage_n_per_image': 400,
    'collage_n': 2000,
    'dataset': "MNIST",
    'dataset_dir':  datapath
}

ddpm_run(params)

#%%
params['f_kernel']=3
params['f_beta']=2
params['f_down']=np.pi/2
params['f_up']=np.pi/2

unet_v_all=[1,2,3]#[1,2,3,4]

for unet_v in unet_v_all:
    params['unet_v']=unet_v
    ddpm_run(params)

# %%
