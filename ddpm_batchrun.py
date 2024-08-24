
from ddpm_run import ddpm_run
import numpy as np

params={
    'unet_v': 0,
    'epochs': 100,
    'batchsize': 32,
    'image_size': 32,
    'image_channels': 1,
    'device': "cuda",
    'lr': 3e-4,
    'noise_steps': 1000,
    'image_gen_per_epoch': 8,
    'f_kernel':3,
    'f_beta': 8,
    'f_down': np.pi/4,
    'f_up': np .pi,
    'save_trining': False,
    'gen_per_batch': 200,
    'gen_total': 3000,
    'seed': 42 

}

ddpm_run(params)



