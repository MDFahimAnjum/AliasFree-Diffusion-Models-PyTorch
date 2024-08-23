
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

filter_size=3
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
#datapath = os.path.join(current_directory,"data\Linnaeus 5 64X64")
datapath = os.path.join(current_directory,"data\MNIST\mnist_train_small.csv")
modelpath= os.path.join(current_directory,"models\DDPM_Uncondtional_F_MNIST\ckpt_F_MNIST.pt")
print(f' Dataset path: {datapath}')
print(f' Model save path: {modelpath}')

#%% test model 

if torch.cuda.is_available():
    print("CUDA is available. Device:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available.")

args = argument()
args.image_size = 32
args.image_channels=1 #3

f_settings={}
f_settings['kernel_size']=3
f_settings['kaiser_beta']=8
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
args.image_size = 32
args.image_channels= 1 #3
args.device = "cuda"
args.lr = 3e-4
args.dataset_path = datapath
args.noise_steps=1000

set_seed(random_seed)

#dataloader, dataset = get_data(args)
dataloader, dataset = get_data_MNIST(args)
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
#dataloader, dataset = get_data(args)
dataloader, dataset = get_data_MNIST(args)
image = next(iter(dataloader))[0]
x=image

def image_data(x):
    if x.shape[1]>1:
        return x.squeeze().permute(1, 2, 0).cpu().numpy()
    elif x.shape[1]==1:
        return x.squeeze().cpu().numpy()
    else:
        return None

images=[]
images.append(image_data(x)) # original

#filter params
omega_c_down=np.pi/2
omega_c_up=np.pi
filter_size=3
beta=8

#downsample
jinc_filter = circularLowpassKernel(omega_c=omega_c_down,N=filter_size, beta=beta)
jinc_filter = jinc_filter.repeat(x.size(1), 1, 1, 1)  # Match number of channels
x = F.conv2d(x, jinc_filter, padding='same', groups=x.size(1))
images.append(image_data(x)) # down filtered
x = F.max_pool2d(x, 2)
images.append(image_data(x)) # downsampled

#upsample
x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
images.append(image_data(x)) # upsampled
sinc_filter = circularLowpassKernel(omega_c=omega_c_up,N=filter_size, beta=beta)
sinc_filter = sinc_filter.repeat(x.size(1), 1, 1, 1)  # Match number of channels
x = F.conv2d(x, sinc_filter, padding='same', groups=x.size(1))
images.append(image_data(x)) # up filtered


titles=[
    'original',
    'downfilter',
    'downsample',
    'upsample',
    'upfilter'
]

fig, axs = plt.subplots(1, len(images), figsize=(3*len(images), 3))

for i,img in enumerate(images):
    axs[i].imshow(img,
                  cmap='gray'
                  )
    axs[i].set_title(titles[i])
    axs[i].axis('off')

plt.tight_layout()
plt.show()

#%% test no filter
# load an image
set_seed(random_seed)
#dataloader, dataset = get_data(args)
dataloader, dataset = get_data_MNIST(args)
image = next(iter(dataloader))[0]
x=image

images=[]
images.append(image_data(x)) # original


#downsample
x = F.max_pool2d(x, 2)
images.append(image_data(x)) # downsampled

#upsample
x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
images.append(image_data(x)) # upsampled


titles=[
    'original',
    'downsample',
    'upsample'
]

fig, axs = plt.subplots(1, len(images), figsize=(3*len(images), 3))

for i,img in enumerate(images):
    axs[i].imshow(img,
                  cmap='gray'
                  )
    axs[i].set_title(titles[i])
    axs[i].axis('off')

plt.tight_layout()
plt.show()



#%% model parameters
args = argument()
args.run_name = "DDPM_Uncondtional_F_MNIST"
args.epochs = 100
args.batch_size = 64  #6  #12
args.image_size = 32 #64
args.image_channels=1 #3
args.dataset_path = datapath
args.device = "cuda"
args.lr = 3e-4
args.noise_steps=1000
args.image_gen_n=8

f_settings={}
f_settings['kernel_size']=3
f_settings['kaiser_beta']=8
f_settings['omega_c_down'] = np.pi/2
f_settings['omega_c_up'] = np.pi

#%% train model
set_seed(random_seed)
# dataloader, dataset = get_data(args)
dataloader, dataset = get_data_MNIST(args)
model = UNet(c_in=args.image_channels, c_out=args.image_channels,
             image_size=args.image_size,f_settings=f_settings,device=args.device).to(args.device)
diffusion = Diffusion(noise_steps=args.noise_steps,img_size=args.image_size, device=args.device)
loss_all=train(args,model_path=modelpath,dataloader=dataloader,model=model,diffusion=diffusion)

#%% inspect training loss
plot_loss(loss_all)

#%% load model
set_seed(random_seed)
model = UNet(c_in=args.image_channels, c_out=args.image_channels,
             image_size=args.image_size,f_settings=f_settings,device=args.device).to(args.device)
ckpt = torch.load(modelpath)
model.load_state_dict(ckpt)
diffusion = Diffusion(noise_steps=args.noise_steps,img_size=args.image_size, device=args.device)

#%% sample images
x = diffusion.sample(model, n=6,image_channels=args.image_channels)
plot_images(x)

#%% denoise image
set_seed(random_seed)
denoise_img = diffusion.revert(model, n=1,image_channels=args.image_channels)
plot_images(denoise_img)
denoise_img.shape

#%% load training images
args = argument()
args.run_name = "DDPM_Uncondtional_F_MNIST"
args.epochs = 100
args.batch_size = 64  #6  #12
args.image_size = 32 #64
args.image_channels=1 #3
args.dataset_path = datapath
args.device = "cuda"
args.lr = 3e-4
args.noise_steps=1000
args.image_gen_n=8
_, dataset = get_data_MNIST(args)

#%% save original dataset
current_directory = os.getcwd() #parameters
savepath = os.path.join(current_directory,"images\original\MNIST")
save_dataset_MNIST(savepath,dataset)

# %% generate images and save
current_directory = os.getcwd() #parameters
savepath = os.path.join(current_directory,"images\generated\MNIST_F")
per_batch=250
total_gen=3000
fileno_start=np.arange(0,total_gen,per_batch)
for start_no in fileno_start:
    fileno=np.arange(start_no,start_no+per_batch,1)
    x = diffusion.sample(model, n=per_batch,image_channels=args.image_channels)
    save_gen_images(savepath,x,fileno)

# %%
