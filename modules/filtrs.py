
from scipy.signal import firwin
from scipy.signal.windows import kaiser
import random
from scipy.special import j1
import numpy as np 
import torch
from matplotlib import pyplot as plt
import torch.nn.functional as F

def jinc_filter_2d(size=6, beta=14):
    # Similar to the sinc filter, create a 2D jinc filter (simplified)
    sinc_filter_1d = np.sinc(np.linspace(-size / 2, size / 2, size))
    window = kaiser(size, beta)
    jinc_filter_2d = np.outer(sinc_filter_1d * window, sinc_filter_1d * window)
    # Normalize the kernel
    jinc_filter_2d = jinc_filter_2d / np.sum(jinc_filter_2d)
    return torch.tensor(jinc_filter_2d, dtype=torch.float32)

def circularLowpassKernel(omega_c=np.pi, N=6,beta=None):  # omega = cutoff frequency in radians (pi is max), N = horizontal size of the kernel, also its vertical size.
    with np.errstate(divide='ignore',invalid='ignore'):
        kernel = np.fromfunction(lambda x, y: omega_c*j1(omega_c*np.sqrt((x - (N - 1)/2)**2 + (y - (N - 1)/2)**2))/(2*np.pi*np.sqrt((x - (N - 1)/2)**2 + (y - (N - 1)/2)**2)), [N, N])
    if N % 2:
        kernel[(N - 1)//2, (N - 1)//2] = omega_c**2/(4*np.pi)
    
    if beta is not None:
        # Create a 1D Kaiser window
        kaiser_window_1d = np.kaiser(N, beta)

        # Generate a 2D Kaiser window by outer product
        kaiser_window_2d = np.outer(kaiser_window_1d, kaiser_window_1d)

        # Apply the Kaiser window to the kernel
        kernel *= kaiser_window_2d
    # Normalize the kernel
    kernel=kernel/ np.sum(kernel)
    return torch.tensor(kernel, dtype=torch.float32)

def plot_filter_and_response(kernel,show_freq=True):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot the kernel
    cax1 = axs[0].imshow(kernel, vmin=-1, vmax=1, cmap='bwr')
    axs[0].set_title('2D Filter')
    fig.colorbar(cax1, ax=axs[0])
    
    # Compute the frequency response
    freq_response = np.fft.fftshift(np.fft.fft2(kernel))
    magnitude_response = np.abs(freq_response)
    
    # Plot the frequency response
    cax2 = axs[1].imshow(magnitude_response, cmap='viridis')
    axs[1].set_title('Frequency Response')

    # Set frequency axis labels
    if show_freq:
        num_rows, num_cols = kernel.shape
        freq_x = np.fft.fftshift(np.fft.fftfreq(num_cols))
        freq_y = np.fft.fftshift(np.fft.fftfreq(num_rows))
        axs[1].set_xticks([0, num_cols//4, num_cols//2, 3*num_cols//4, num_cols-1])
        axs[1].set_xticklabels([f'{freq:.2f}' for freq in freq_x[axs[1].get_xticks().astype(int)]])
        axs[1].set_yticks([0, num_rows//4, num_rows//2, 3*num_rows//4, num_rows-1])
        axs[1].set_yticklabels([f'{freq:.2f}' for freq in freq_y[axs[1].get_yticks().astype(int)]])

    fig.colorbar(cax2, ax=axs[1])
    
    plt.tight_layout()
    plt.show()


def custom_downsample( x, jinc_filter,factor=2):
    # Apply the Jinc filter before downsampling
    jinc_filter = jinc_filter[None, None, :, :].to(x.device)  # Shape (1, 1, filter_size, filter_size)
    jinc_filter = jinc_filter.repeat(x.size(1), 1, 1, 1)  # Match number of channels
    x = F.conv2d(x, jinc_filter, padding='same', groups=x.size(1))
    x = x[:, :, ::factor, ::factor]
    return x

def custom_upsample(x, sinc_filter,factor=2):
    # Upsample using zero padding followed by applying the sinc filter
    # Get the original dimensions
    batch_size, channels, height, width = x.shape

    # Create a new tensor with double the height and width filled with zeros
    upsampled = torch.zeros(batch_size, channels, height * factor, width * factor, device=x.device)

    # Assign the original values to the correct positions
    upsampled[:, :, ::factor, ::factor] = x
    x=upsampled
    # Apply the sinc filter (low-pass filter)
    sinc_filter = sinc_filter[None, None, :, :].to(x.device)  # Shape (1, 1, filter_size, filter_size)
    sinc_filter = sinc_filter.repeat(x.size(1), 1, 1, 1)  # Match number of channels
    x = F.conv2d(x, sinc_filter, padding='same', groups=x.size(1))
    return x    

