
#%% 
from ddpm_run import ddpm_run
import numpy as np
import os

#%%
current_directory = os.getcwd() 

# MNIST
#---------
datapath = os.path.join(current_directory,"data\MNIST\mnist_train_small.csv")

params={
    'unet_v': 0,
    'epochs': 2,
    'batchsize': 32,
    'image_size': 32,
    'image_channels': 1,
    'device': "cuda",
    'lr': 3e-4,
    'noise_steps': 1000,
    'image_gen_per_epoch': 8,
    'f_kernel': None,#3,
    'f_beta': None,#8,
    'f_down': None,#np.pi/4,
    'f_up': None,#np .pi,
    'save_trining': False,
    'gen_per_batch': 25,#200,
    'gen_total': 50,#3000,
    'seed': 42,
    'collage_n_per_image': 25,#400,
    'collage_n': 50,#2000,
    'dataset': "MNIST",
    'dataset_dir':  datapath
}

ddpm_run(params)

params['f_kernel']=3
params['f_beta']=8
params['f_down']=np.pi/2
params['f_up']=np.pi

unet_v_all=[1,2,3]

for unet_v in unet_v_all:
    params['unet_v']=unet_v
    ddpm_run(params)

# MNIST-M
#---------
datapath = os.path.join(current_directory,"data\MNIST-M-6000")
#datapath = os.path.join(current_directory,"data\Linnaeus 5 64X64")
params={
    'unet_v': 0,
    'epochs': 2,
    'batchsize': 32,
    'image_size': 32,
    'image_channels': 3,
    'device': "cuda",
    'lr': 3e-4,
    'noise_steps': 1000,
    'image_gen_per_epoch': 8,
    'f_kernel': None,#3,
    'f_beta': None,#8,
    'f_down': None,#np.pi/4,
    'f_up': None,#np .pi,
    'save_trining': False,
    'gen_per_batch': 25,
    'gen_total': 50,
    'seed': 42,
    'collage_n_per_image': 25,
    'collage_n': 50,
    'dataset': "MNIST-M",
    'dataset_dir':  datapath
}

ddpm_run(params)

params['f_kernel']=3
params['f_beta']=8
params['f_down']=np.pi/2
params['f_up']=np.pi

unet_v_all=[1,2,3]

for unet_v in unet_v_all:
    params['unet_v']=unet_v
    ddpm_run(params)


# %%
