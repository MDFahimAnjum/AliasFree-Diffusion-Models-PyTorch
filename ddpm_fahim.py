
#%% imports
import os
import torch
from modules.utils import *
from modules.ddpm_utils import *
from modules.ddpm_models import *

import logging
#from torch.utils.tensorboard import SummaryWriter
from modules.utils import get_data
import numpy as np 

#%% for random seed
random_seed=42
set_seed(random_seed)


#%% path
logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")
current_directory = os.getcwd() #parameters
#datapath = os.path.join(current_directory,"data\Linnaeus 5 64X64")
datapath = os.path.join(current_directory,"data\MNIST\mnist_train_small.csv")
modelpath= os.path.join(current_directory,"models\DDPM_Uncondtional_MNIST\ckpt_MNIST.pt")
print(f' Dataset path: {datapath}')
print(f' Model save path: {modelpath}')

#%% test model 

if torch.cuda.is_available():
    print("CUDA is available. Device:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available.")

args = argument()
args.image_size = 64
args.image_channels=1 #3

net = UNet(c_in=args.image_channels, c_out=args.image_channels,image_size=args.image_size,device="cpu")
#net = UNet_conditional(num_classes=10, device="cpu")
print(sum([p.numel() for p in net.parameters()]))
x = torch.randn(2, args.image_channels, args.image_size, args.image_size)
t = x.new_tensor([500] * x.shape[0]).long()
y = x.new_tensor([1] * x.shape[0]).long()
#print(net(x, t, y).shape) # conditional case
print(net(x, t).shape)
#%% test noise
args = argument()
args.batch_size = 1
args.image_size = 64
args.image_channels=1 #3
args.device = "cuda"
args.lr = 3e-4
args.dataset_path = datapath
args.noise_steps=1000

set_seed(random_seed)

#dataloader = get_data(args)
dataloader = get_data_MNIST(args)
image = next(iter(dataloader))[0]
image = image.to(args.device)
t = torch.Tensor(np.round(np.linspace(0,args.noise_steps-1,9))).long().to(args.device)
diffusion = Diffusion(noise_steps=args.noise_steps,img_size=args.image_size, device=args.device)
noised_image, _ = diffusion.noise_images(image, t)
noised_image = (noised_image.clamp(-1, 1) + 1) / 2
noised_image = (noised_image * 255).type(torch.uint8)
plot_images(noised_image)

#%% train model
args = argument()
args.run_name = "DDPM_Uncondtional_MNIST"
args.epochs = 100
args.batch_size = 64 #4  #6  #12
args.image_size = 32 #64
args.image_channels=1 #3
args.dataset_path = datapath
args.device = "cuda"
args.lr = 3e-4
args.noise_steps=1000
args.image_gen_n=8

set_seed(random_seed)
# dataloader = get_data(args)
dataloader = get_data_MNIST(args)
model = UNet(c_in=args.image_channels, c_out=args.image_channels,image_size=args.image_size,device=args.device).to(args.device)
diffusion = Diffusion(noise_steps=args.noise_steps,img_size=args.image_size, device=args.device)
loss_all=train(args,model_path=modelpath,dataloader=dataloader,model=model,diffusion=diffusion)

#%% inspect training loss
plot_loss(loss_all)

#%% sample images
set_seed(random_seed)
model = UNet(c_in=args.image_channels, c_out=args.image_channels,image_size=args.image_size).to(args.device)
ckpt = torch.load(modelpath)
model.load_state_dict(ckpt)
x = diffusion.sample(model, n=6,image_channels=args.image_channels)
plot_images(x)

#%% denoise image
set_seed(random_seed)
denoise_img = diffusion.revert(model, n=1,image_channels=args.image_channels)
plot_images(denoise_img)
denoise_img.shape
# %%
