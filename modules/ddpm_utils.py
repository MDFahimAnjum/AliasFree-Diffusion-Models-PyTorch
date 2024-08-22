import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.utils import *
from modules.filtrs import *
import logging
from tqdm import tqdm
from torch import optim

# param class
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


class Down_F(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256, filter_size=6, beta=14,omega_c=np.pi/2):
        super().__init__()

        # Generate the 2D Jinc filter with Kaiser window
        #self.jinc_filter = jinc_filter_2d(filter_size, beta)
        self.jinc_filter = circularLowpassKernel(omega_c=omega_c,N=filter_size, beta=beta)

        self.conv = nn.Sequential(
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
        # Downsample using the custom Jinc-based filter
        x = self.custom_downsample(x, self.jinc_filter)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb

    def custom_downsample(self, x, jinc_filter):
        # Apply the Jinc filter before downsampling
        jinc_filter = jinc_filter[None, None, :, :].to(x.device)  # Shape (1, 1, filter_size, filter_size)
        jinc_filter = jinc_filter.repeat(x.size(1), 1, 1, 1)  # Match number of channels
        x = F.conv2d(x, jinc_filter, padding='same', groups=x.size(1))
        x = F.max_pool2d(x, 2)
        return x


class Up_F(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256, filter_size=6, beta=14,omega_c=np.pi):
        super().__init__()

        # Generate the 2D sinc filter with Kaiser window
        #self.sinc_filter = jinc_filter_2d(filter_size, beta)
        self.sinc_filter = circularLowpassKernel(omega_c=omega_c,N=filter_size, beta=beta)

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
        # Upsample using the custom filter
        x = self.custom_upsample(x, self.sinc_filter)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb

    def custom_upsample(self, x, sinc_filter):
        # Upsample using F.interpolate, followed by applying the sinc filter
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        # Apply the sinc filter (low-pass filter)
        sinc_filter = sinc_filter[None, None, :, :].to(x.device)  # Shape (1, 1, filter_size, filter_size)
        sinc_filter = sinc_filter.repeat(x.size(1), 1, 1, 1)  # Match number of channels
        x = F.conv2d(x, sinc_filter, padding='same', groups=x.size(1))
        return x

# training def
def train(args,model_path=None,dataloader=None,model=None,diffusion=None):
    setup_logging(args.run_name)
    device = args.device
    #if(model_path):
    #    ckpt = torch.load(model_path)
    #    model.load_state_dict(ckpt)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    #logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)
    loss_all=[]
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
            loss_all.append(loss)
            pbar.set_postfix(MSE=loss.item())
            #logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        sampled_images = diffusion.sample(model, n=images.shape[0],image_channels=args.image_channels)
        save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
        torch.save(model.state_dict(), model_path)
    return loss_all