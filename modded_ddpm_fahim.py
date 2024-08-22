
# dataset: http://chaladze.com/l5/

#%% imports
import os
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from modules.utils import *
import logging
#from torch.utils.tensorboard import SummaryWriter
from modules.utils import get_data
from modules.filtrs import *
from modules.utils import *
from modules.ddpm_utils import *
from modules.ddpm_models import *

#%% for random seed
random_seed=42
set_seed(random_seed)

#%% Filters

filter_size=7
beta=2
omega_c = np.pi/2  # Cutoff frequency in radians <= pi

filters=[]
filters.append( jinc_filter_2d(filter_size, beta))
filters.append(circularLowpassKernel(omega_c, filter_size))
filters.append(circularLowpassKernel(omega_c, filter_size,beta=beta))

for filter in filters:  
    plot_filter_and_response(filter)


#%% path
logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")
current_directory = os.getcwd() #parameters
datapath = os.path.join(current_directory,"data\Linnaeus 5 64X64")
modelpath= os.path.join(current_directory,"models\DDPM_Uncondtional_F\ckpt2.pt")
print(f' Dataset path: {datapath}')
print(f' Model save path: {modelpath}')

#%% test model 

if torch.cuda.is_available():
    print("CUDA is available. Device:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available.")

args = argument()
args.image_size = 64
args.image_channels=3

f_settings={}
f_settings['kernel_size']=7
f_settings['kaiser_beta']=4
f_settings['omega_c_down'] = np.pi/2
f_settings['omega_c_up'] = np.pi

net = UNet(c_in=args.image_channels, c_out=args.image_channels,
           image_size=args.image_size,f_settings=f_settings,device="cpu")
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
args.image_channels=3
args.device = "cuda"
args.lr = 3e-4
args.dataset_path = datapath
args.noise_steps=1000

set_seed(random_seed)

dataloader = get_data(args)
image = next(iter(dataloader))[0]
image = image.to(args.device)
t = torch.Tensor(np.round(np.linspace(0,args.noise_steps-1,9))).long().to(args.device)
diffusion = Diffusion(noise_steps=args.noise_steps,img_size=args.image_size, device=args.device)
noised_image, _ = diffusion.noise_images(image, t)
noised_image = (noised_image.clamp(-1, 1) + 1) / 2
noised_image = (noised_image * 255).type(torch.uint8)
plot_images(noised_image)


#%% test filter
# load an image
set_seed(random_seed)
dataloader = get_data(args)
image = next(iter(dataloader))[0]
x=image

images=[]
images.append(x.squeeze().permute(1, 2, 0).cpu().numpy()) # original

#filter params
omega_c_down=np.pi/2
omega_c_up=np.pi
filter_size=7
beta=1

#downsample
jinc_filter = circularLowpassKernel(omega_c=omega_c_down,N=filter_size, beta=beta)
jinc_filter = jinc_filter.repeat(x.size(1), 1, 1, 1)  # Match number of channels
x = F.conv2d(x, jinc_filter, padding='same', groups=x.size(1))
images.append(x.squeeze().permute(1, 2, 0).cpu().numpy()) # down filtered
x = F.max_pool2d(x, 2)
images.append(x.squeeze().permute(1, 2, 0).cpu().numpy()) # downsampled

#upsample
x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
images.append(x.squeeze().permute(1, 2, 0).cpu().numpy()) # upsampled
sinc_filter = circularLowpassKernel(omega_c=omega_c_up,N=filter_size, beta=beta)
sinc_filter = sinc_filter.repeat(x.size(1), 1, 1, 1)  # Match number of channels
x = F.conv2d(x, sinc_filter, padding='same', groups=x.size(1))
images.append(x.squeeze().permute(1, 2, 0).cpu().numpy()) # up filtered


titles=[
    'original',
    'downfilter',
    'downsample',
    'upsample',
    'upfilter'
]

fig, axs = plt.subplots(1, len(images), figsize=(3*len(images), 3))

for i,img in enumerate(images):
    axs[i].imshow(img)
    axs[i].set_title(titles[i])
    axs[i].axis('off')

plt.tight_layout()
plt.show()

#%% test no filter
# load an image
set_seed(random_seed)
dataloader = get_data(args)
image = next(iter(dataloader))[0]
x=image

images=[]
images.append(x.squeeze().permute(1, 2, 0).cpu().numpy()) # original


#downsample
x = F.max_pool2d(x, 2)
images.append(x.squeeze().permute(1, 2, 0).cpu().numpy()) # downsampled

#upsample
x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
images.append(x.squeeze().permute(1, 2, 0).cpu().numpy()) # upsampled


titles=[
    'original',
    'downsample',
    'upsample'
]

fig, axs = plt.subplots(1, len(images), figsize=(3*len(images), 3))

for i,img in enumerate(images):
    axs[i].imshow(img)
    axs[i].set_title(titles[i])
    axs[i].axis('off')

plt.tight_layout()
plt.show()



#%% train model
args = argument()
args.run_name = "DDPM_Uncondtional_F"
args.epochs = 2
args.batch_size = 4  #6  #12
args.image_size = 64
args.image_channels=3
args.dataset_path = datapath
args.device = "cuda"
args.lr = 3e-4
args.noise_steps=1000

f_settings={}
f_settings['kernel_size']=7
f_settings['kaiser_beta']=4
f_settings['omega_c_down'] = np.pi/2
f_settings['omega_c_up'] = np.pi

set_seed(random_seed)
dataloader = get_data(args)
model = UNet(c_in=args.image_channels, c_out=args.image_channels,
             image_size=args.image_size,f_settings=f_settings,device=args.device).to(args.device)
diffusion = Diffusion(noise_steps=args.noise_steps,img_size=args.image_size, device=args.device)
train(args,model_path=modelpath,dataloader=dataloader,model=model,diffusion=diffusion)

#%% sample images
set_seed(random_seed)
model = UNet(c_in=args.image_channels, c_out=args.image_channels,
             image_size=args.image_size,f_settings=f_settings,device=args.device).to(args.device)
ckpt = torch.load(modelpath)
model.load_state_dict(ckpt)
x = diffusion.sample(model, n=6)
plot_images(x)

#%% denoise image
set_seed(random_seed)
denoise_img = diffusion.revert(model, n=1)
plot_images(denoise_img)
denoise_img.shape
# %%
