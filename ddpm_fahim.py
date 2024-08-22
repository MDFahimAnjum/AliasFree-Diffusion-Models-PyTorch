
#%% imports
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
from utils import *
import logging
#from torch.utils.tensorboard import SummaryWriter
from utils import get_data
from torchvision.utils import save_image
import numpy as np 
import random

#%% for random seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
random_seed=42
set_seed(random_seed)
#%% param class

class argument:
    def __init__(self,run_name=None,epochs=None,batch_size=None,image_size=None,image_channels=3,dataset_path=None,device=None,lr=None,noise_steps=None):
        super().__init__()
        self.run_name = run_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.image_size = image_size
        self.image_channels=image_channels
        self.dataset_path = dataset_path
        self.device = device
        self.lr = lr
        self.noise_steps=noise_steps


#%% Modules for the model


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb

"""
| Layer              | Input Size (Channels x H x W)     | Output Size (Channels x H x W)        | Notes                                                                                                                                      |
|--------------------|-----------------------------------|---------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------|
| `self.inc(x)`      | `c_in x H x W`                    | `32 x H x W`                          | Initial double convolution, `c_in` is the number of input channels                                                                         |
| `self.down1(x1)`   | `32 x H x W`                      | `64 x H/2 x W/2`                      | Downsampling, image size halved, channels doubled                                                                                          |
| `self.sa1(x2)`     | `64 x H/2 x W/2`                  | `64 x H/2 x W/2`                      | Self-attention, maintains the same size                                                                                                    |
| `self.down2(x2)`   | `64 x H/2 x W/2`                  | `128 x H/4 x W/4`                     | Downsampling, image size halved, channels doubled                                                                                          |
| `self.sa2(x3)`     | `128 x H/4 x W/4`                 | `128 x H/4 x W/4`                     | Self-attention, maintains the same size                                                                                                    |
| `self.down3(x3)`   | `128 x H/4 x W/4`                 | `128 x H/8 x W/8`                     | Downsampling, image size halved, channels stay the same                                                                                    |
| `self.sa3(x4)`     | `128 x H/8 x W/8`                 | `128 x H/8 x W/8`                     | Self-attention, maintains the same size                                                                                                    |
| `self.bot1(x4)`    | `128 x H/8 x W/8`                 | `256 x H/8 x W/8`                     | Bottleneck double convolution, channels doubled                                                                                            |
| `self.bot2(x4)`    | `256 x H/8 x W/8`                 | `256 x H/8 x W/8`                     | Bottleneck double convolution, channels stay the same                                                                                      |
| `self.bot3(x4)`    | `256 x H/8 x W/8`                 | `128 x H/8 x W/8`                     | Bottleneck double convolution, channels halved                                                                                             |
| `self.up1(x4, x3)` | `128 x H/8 x W/8` & `128 x H/4 x W/4` | `64 x H/4 x W/4`                       | Upsampling, image size doubled, channels halved + skip connection                                                                          |
| `self.sa4(x)`      | `64 x H/4 x W/4`                  | `64 x H/4 x W/4`                      | Self-attention, maintains the same size                                                                                                    |
| `self.up2(x, x2)`  | `64 x H/4 x W/4` & `64 x H/2 x W/2`   | `32 x H/2 x W/2`                       | Upsampling, image size doubled, channels halved + skip connection                                                                          |
| `self.sa5(x)`      | `32 x H/2 x W/2`                  | `32 x H/2 x W/2`                      | Self-attention, maintains the same size                                                                                                    |
| `self.up3(x, x1)`  | `32 x H/2 x W/2` & `32 x H x W`       | `32 x H x W`                           | Upsampling, image size doubled, channels stay the same + skip connection                                                                   |
| `self.sa6(x)`      | `32 x H x W`                      | `32 x H x W`                          | Self-attention, maintains the same size                                                                                                    |
| `self.outc(x)`     | `32 x H x W`                      | `c_out x H x W`                       | Final convolution, output channels to `c_out` (e.g., for classification, segmentation, or image generation) |


### Key Points:
- **Downsampling (`Down`) layers** reduce the height (`H`) and width (`W`) of the image by a factor of 2, while typically doubling the number of channels.
- **Upsampling (`Up`) layers** increase the height (`H`) and width (`W`) by a factor of 2, while reducing the number of channels. They also incorporate skip connections from the corresponding downsampling layers.
- **Self-Attention (`SelfAttention`) layers** maintain the spatial dimensions but enhance the feature representation by focusing on different parts of the feature maps.
- **Bottleneck layers (`bot1`, `bot2`, `bot3`)** work at the smallest resolution, focusing on deeper feature extraction.
- The final output size is determined by the `outc` layer, which typically adjusts the number of channels to `c_out` depending on the specific task.
"""
class UNet(nn.Module):
    def __init__(self, c_in=3, c_out=3,image_size=64, time_dim=256, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.image_size=image_size
        self.inc = DoubleConv(in_channels=c_in, out_channels=int(self.image_size) )
        self.down1 = Down(in_channels=int(self.image_size),out_channels= int(2*self.image_size))
        self.sa1 = SelfAttention(channels=int(2*self.image_size), size=int(self.image_size/2))
        self.down2 = Down(in_channels=int(2*self.image_size),out_channels= int(4*self.image_size))
        self.sa2 = SelfAttention(channels=int(4*self.image_size), size=int(self.image_size/4))
        self.down3 = Down(in_channels=int(4*self.image_size), out_channels=int(4*self.image_size))
        self.sa3 = SelfAttention(channels=int(4*self.image_size), size=int(self.image_size/8))

        self.bot1 = DoubleConv(in_channels=int(4*self.image_size),out_channels=int(8*self.image_size))
        self.bot2 = DoubleConv(in_channels=int(8*self.image_size), out_channels=int(8*self.image_size))
        self.bot3 = DoubleConv(in_channels=int(8*self.image_size),out_channels=int(4*self.image_size))

        self.up1 = Up(in_channels=int(8*self.image_size), out_channels=int(2*self.image_size))
        self.sa4 = SelfAttention(channels=int(2*self.image_size), size=int(self.image_size/4))
        self.up2 = Up(in_channels=int(4*self.image_size),out_channels= int(self.image_size))
        self.sa5 = SelfAttention(channels=int(self.image_size), size=int(self.image_size/2))
        self.up3 = Up(in_channels=int(2*self.image_size), out_channels=int(self.image_size))
        self.sa6 = SelfAttention(channels=int(self.image_size), size=int(self.image_size))
        self.outc = nn.Conv2d(in_channels=int(self.image_size), out_channels=c_out, kernel_size=1)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output


class UNet_conditional(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=256, num_classes=None, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128, 32)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256, 16)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256, 8)

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128, 16)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64, 32)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64, 64)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_dim)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t, y):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        if y is not None:
            t += self.label_emb(y)

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output


#%% diffusion model
logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def revert(self,model,n):
        logging.info(f"Sampling {n} new images....")
        result = []
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
                if(i % 100 == 0):
                    result.append(x)
        model.train()
        result.append(x)
        result = torch.cat(result)
        result = (result.clamp(-1, 1) + 1) / 2
        result = (result * 255).type(torch.uint8)
        return result

    def sample(self, model, n):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x

#%% training def
def train(args,model_path=None,dataloader=None):
    setup_logging(args.run_name)
    device = args.device
    dataloader = get_data(args)
    model = UNet(c_in=args.image_channels, c_out=args.image_channels,image_size=args.image_size,device=device).to(device)
    #if(model_path):
    #    ckpt = torch.load(model_path)
    #    model.load_state_dict(ckpt)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(noise_steps=args.noise_steps,img_size=args.image_size, device=device)
    #logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, (images, _) in enumerate(pbar):
            images = images.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            predicted_noise = model(x_t, t)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())
            #logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        sampled_images = diffusion.sample(model, n=images.shape[0])
        save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
        torch.save(model.state_dict(), model_path)

#%% path
current_directory = os.getcwd() #parameters
datapath = os.path.join(current_directory,"data\Linnaeus 5 64X64")
modelpath= os.path.join(current_directory,"models\DDPM_Uncondtional\ckpt.pt")
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

#%% train model
args = argument()
args.run_name = "DDPM_Uncondtional"
args.epochs = 2
args.batch_size = 4  #6  #12
args.image_size = 64
args.image_channels=3
args.dataset_path = datapath
args.device = "cuda"
args.lr = 3e-4
args.noise_steps=10

set_seed(random_seed)
dataloader = get_data(args)
train(args,model_path=modelpath,dataloader=dataloader)

#%% sample images
set_seed(random_seed)
model = UNet(c_in=args.image_channels, c_out=args.image_channels,image_size=args.image_size).to(args.device)
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
