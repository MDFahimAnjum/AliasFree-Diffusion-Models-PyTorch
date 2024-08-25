
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.ddpm_utils import *
import logging
from tqdm import tqdm


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
    def __init__(self, c_in=3, c_out=3,image_size=64, time_dim=256, device="cuda", f_settings=None,num_classes=None,variant=0):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.image_size=image_size
        self.f_settings=f_settings

        if variant==0:
            print(f'Variant {variant} Original UNet')
            self.inc = DoubleConv(in_channels=c_in, 
                                out_channels=int(self.image_size) )
            self.down1 = Down(in_channels=int(self.image_size),
                            out_channels= int(2*self.image_size))
            self.sa1 = SelfAttention(channels=int(2*self.image_size), 
                                    size=int(self.image_size/2))
            self.down2 = Down(in_channels=int(2*self.image_size),
                            out_channels= int(4*self.image_size))
            self.sa2 = SelfAttention(channels=int(4*self.image_size), 
                                    size=int(self.image_size/4))
            self.down3 = Down(in_channels=int(4*self.image_size), 
                            out_channels=int(4*self.image_size))
            self.sa3 = SelfAttention(channels=int(4*self.image_size), 
                                    size=int(self.image_size/8))

            self.bot1 = DoubleConv(in_channels=int(4*self.image_size),
                                out_channels=int(8*self.image_size))
            self.bot2 = DoubleConv(in_channels=int(8*self.image_size), 
                                out_channels=int(8*self.image_size))
            self.bot3 = DoubleConv(in_channels=int(8*self.image_size),
                                out_channels=int(4*self.image_size))

            self.up1 = Up(in_channels=int(8*self.image_size), 
                        out_channels=int(2*self.image_size))
            self.sa4 = SelfAttention(channels=int(2*self.image_size), 
                                    size=int(self.image_size/4))
            self.up2 = Up(in_channels=int(4*self.image_size),
                        out_channels= int(self.image_size))
            self.sa5 = SelfAttention(channels=int(self.image_size), 
                                    size=int(self.image_size/2))
            self.up3 = Up(in_channels=int(2*self.image_size), 
                        out_channels=int(self.image_size))
            self.sa6 = SelfAttention(channels=int(self.image_size), 
                                    size=int(self.image_size))
            self.outc = nn.Conv2d(in_channels=int(self.image_size), 
                                out_channels=c_out, kernel_size=1)
        elif variant==1:
            if self.f_settings is None:
                raise ValueError("f_settings is empty")
            
            print(f'Variant {variant} Modified UNet: aliasing filters in up and downsampling')
            self.inc = DoubleConv(in_channels=c_in, 
                              out_channels=int(self.image_size) )
            self.down1 = Down_FF(in_channels=int(self.image_size),
                            out_channels= int(2*self.image_size),f_settings=self.f_settings)
            self.sa1 = SelfAttention(channels=int(2*self.image_size), 
                                    size=int(self.image_size/2))
            self.down2 = Down_FF(in_channels=int(2*self.image_size),
                            out_channels= int(4*self.image_size),f_settings=self.f_settings)
            self.sa2 = SelfAttention(channels=int(4*self.image_size), 
                                    size=int(self.image_size/4))
            self.down3 = Down_FF(in_channels=int(4*self.image_size), 
                            out_channels=int(4*self.image_size),f_settings=self.f_settings)
            self.sa3 = SelfAttention(channels=int(4*self.image_size), 
                                    size=int(self.image_size/8))

            self.bot1 = DoubleConv(in_channels=int(4*self.image_size),
                                out_channels=int(8*self.image_size))
            self.bot2 = DoubleConv(in_channels=int(8*self.image_size), 
                                out_channels=int(8*self.image_size))
            self.bot3 = DoubleConv(in_channels=int(8*self.image_size),
                                out_channels=int(4*self.image_size))

            self.up1 = Up_FF(in_channels=int(8*self.image_size), 
                        out_channels=int(2*self.image_size),f_settings=self.f_settings)
            self.sa4 = SelfAttention(channels=int(2*self.image_size), 
                                    size=int(self.image_size/4))
            self.up2 = Up_FF(in_channels=int(4*self.image_size),
                        out_channels= int(self.image_size),f_settings=self.f_settings)
            self.sa5 = SelfAttention(channels=int(self.image_size), 
                                    size=int(self.image_size/2))
            self.up3 = Up_FF(in_channels=int(2*self.image_size), 
                        out_channels=int(self.image_size),f_settings=self.f_settings)
            self.sa6 = SelfAttention(channels=int(self.image_size), 
                                    size=int(self.image_size))
            self.outc = nn.Conv2d(in_channels=int(self.image_size), 
                                out_channels=c_out, kernel_size=1)
        elif variant==2:
            if self.f_settings is None:
                raise ValueError("f_settings is empty")
            
            print(f'Variant {variant} Modified UNet: filters around gelu but no filters in up or downsampling')
            self.inc = DoubleConv_F(in_channels=c_in, 
                              out_channels=int(self.image_size),f_settings=self.f_settings )
            self.down1 = Down_F(in_channels=int(self.image_size),
                            out_channels= int(2*self.image_size),f_settings=self.f_settings)
            self.sa1 = SelfAttention(channels=int(2*self.image_size), 
                                    size=int(self.image_size/2))
            self.down2 = Down_F(in_channels=int(2*self.image_size),
                            out_channels= int(4*self.image_size),f_settings=self.f_settings)
            self.sa2 = SelfAttention(channels=int(4*self.image_size), 
                                    size=int(self.image_size/4))
            self.down3 = Down_F(in_channels=int(4*self.image_size), 
                            out_channels=int(4*self.image_size),f_settings=self.f_settings)
            self.sa3 = SelfAttention(channels=int(4*self.image_size), 
                                    size=int(self.image_size/8))

            self.bot1 = DoubleConv_F(in_channels=int(4*self.image_size),
                                out_channels=int(8*self.image_size),f_settings=self.f_settings)
            self.bot2 = DoubleConv_F(in_channels=int(8*self.image_size), 
                                out_channels=int(8*self.image_size),f_settings=self.f_settings)
            self.bot3 = DoubleConv_F(in_channels=int(8*self.image_size),
                                out_channels=int(4*self.image_size),f_settings=self.f_settings)

            self.up1 = Up_F(in_channels=int(8*self.image_size), 
                        out_channels=int(2*self.image_size),f_settings=self.f_settings)
            self.sa4 = SelfAttention(channels=int(2*self.image_size), 
                                    size=int(self.image_size/4))
            self.up2 = Up_F(in_channels=int(4*self.image_size),
                        out_channels= int(self.image_size),f_settings=self.f_settings)
            self.sa5 = SelfAttention(channels=int(self.image_size), 
                                    size=int(self.image_size/2))
            self.up3 = Up_F(in_channels=int(2*self.image_size), 
                        out_channels=int(self.image_size),f_settings=self.f_settings)
            self.sa6 = SelfAttention(channels=int(self.image_size), 
                                    size=int(self.image_size))
            self.outc = nn.Conv2d(in_channels=int(self.image_size), 
                                out_channels=c_out, kernel_size=1)
        elif variant==3:
            if self.f_settings is None:
                raise ValueError("f_settings is empty")
            
            print(f'Variant {variant} Modified UNet: filters around gelu + filters in up or downsampling')
            self.inc = DoubleConv_F(in_channels=c_in, 
                              out_channels=int(self.image_size),f_settings=self.f_settings )
            self.down1 = Down_FFF(in_channels=int(self.image_size),
                            out_channels= int(2*self.image_size),f_settings=self.f_settings)
            self.sa1 = SelfAttention(channels=int(2*self.image_size), 
                                    size=int(self.image_size/2))
            self.down2 = Down_FFF(in_channels=int(2*self.image_size),
                            out_channels= int(4*self.image_size),f_settings=self.f_settings)
            self.sa2 = SelfAttention(channels=int(4*self.image_size), 
                                    size=int(self.image_size/4))
            self.down3 = Down_FFF(in_channels=int(4*self.image_size), 
                            out_channels=int(4*self.image_size),f_settings=self.f_settings)
            self.sa3 = SelfAttention(channels=int(4*self.image_size), 
                                    size=int(self.image_size/8))

            self.bot1 = DoubleConv_F(in_channels=int(4*self.image_size),
                                out_channels=int(8*self.image_size),f_settings=self.f_settings)
            self.bot2 = DoubleConv_F(in_channels=int(8*self.image_size), 
                                out_channels=int(8*self.image_size),f_settings=self.f_settings)
            self.bot3 = DoubleConv_F(in_channels=int(8*self.image_size),
                                out_channels=int(4*self.image_size),f_settings=self.f_settings)

            self.up1 = Up_FFF(in_channels=int(8*self.image_size), 
                        out_channels=int(2*self.image_size),f_settings=self.f_settings)
            self.sa4 = SelfAttention(channels=int(2*self.image_size), 
                                    size=int(self.image_size/4))
            self.up2 = Up_FFF(in_channels=int(4*self.image_size),
                        out_channels= int(self.image_size),f_settings=self.f_settings)
            self.sa5 = SelfAttention(channels=int(self.image_size), 
                                    size=int(self.image_size/2))
            self.up3 = Up_FFF(in_channels=int(2*self.image_size), 
                        out_channels=int(self.image_size),f_settings=self.f_settings)
            self.sa6 = SelfAttention(channels=int(self.image_size), 
                                    size=int(self.image_size))
            self.outc = nn.Conv2d(in_channels=int(self.image_size), 
                                out_channels=c_out, kernel_size=1)
        elif variant==4:
            if self.f_settings is None:
                raise ValueError("f_settings is empty")
            
            print(f'Variant {variant} Modified UNet: filters around gelu + filters in up or downsampling + groupnorm')
            self.inc = DoubleConv_F4(in_channels=c_in, 
                              out_channels=int(self.image_size),f_settings=self.f_settings )
            self.down1 = Down_F4(in_channels=int(self.image_size),
                            out_channels= int(2*self.image_size),f_settings=self.f_settings)
            self.sa1 = SelfAttention(channels=int(2*self.image_size), 
                                    size=int(self.image_size/2))
            self.down2 = Down_F4(in_channels=int(2*self.image_size),
                            out_channels= int(4*self.image_size),f_settings=self.f_settings)
            self.sa2 = SelfAttention(channels=int(4*self.image_size), 
                                    size=int(self.image_size/4))
            self.down3 = Down_F4(in_channels=int(4*self.image_size), 
                            out_channels=int(4*self.image_size),f_settings=self.f_settings)
            self.sa3 = SelfAttention(channels=int(4*self.image_size), 
                                    size=int(self.image_size/8))

            self.bot1 = DoubleConv_F4(in_channels=int(4*self.image_size),
                                out_channels=int(8*self.image_size),f_settings=self.f_settings)
            self.bot2 = DoubleConv_F4(in_channels=int(8*self.image_size), 
                                out_channels=int(8*self.image_size),f_settings=self.f_settings)
            self.bot3 = DoubleConv_F4(in_channels=int(8*self.image_size),
                                out_channels=int(4*self.image_size),f_settings=self.f_settings)

            self.up1 = Up_F4(in_channels=int(8*self.image_size), 
                        out_channels=int(2*self.image_size),f_settings=self.f_settings)
            self.sa4 = SelfAttention(channels=int(2*self.image_size), 
                                    size=int(self.image_size/4))
            self.up2 = Up_F4(in_channels=int(4*self.image_size),
                        out_channels= int(self.image_size),f_settings=self.f_settings)
            self.sa5 = SelfAttention(channels=int(self.image_size), 
                                    size=int(self.image_size/2))
            self.up3 = Up_F4(in_channels=int(2*self.image_size), 
                        out_channels=int(self.image_size),f_settings=self.f_settings)
            self.sa6 = SelfAttention(channels=int(self.image_size), 
                                    size=int(self.image_size))
            self.outc = nn.Conv2d(in_channels=int(self.image_size), 
                                out_channels=c_out, kernel_size=1)
        else:
            raise ValueError("variant value must be between 0 and 4")
        
        # for conditional Unet
        if num_classes is not None:
            print("Conditional UNet")
            self.label_emb = nn.Embedding(num_classes, time_dim)
        else:
            print("Unconditional UNet")
        

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t, y=None):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        # for conditional
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

# diffusion model
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

    def revert(self,model,n,image_channels):
        logging.info(f"Sampling {n} new images....")
        result = []
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, image_channels, self.img_size, self.img_size)).to(self.device)
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

    def sample(self, model, n,image_channels):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, image_channels, self.img_size, self.img_size)).to(self.device)
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


