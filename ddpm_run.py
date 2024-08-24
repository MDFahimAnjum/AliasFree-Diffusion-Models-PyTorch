
# dataset: http://chaladze.com/l5/

# imports
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
import csv
import gc

def ddpm_run(params):
    # set params
    unet_variant=params['unet_v']
    dataset_name=params['dataset']

    args = argument()
    args.run_name = f"DDPM_Uncondtional_{dataset_name}_{unet_variant}"
    args.epochs = params['epochs']
    args.batch_size = params['batchsize']
    args.image_size = params['image_size']
    args.image_channels=params['image_channels']
    args.device = params['device']
    args.lr = params['lr']
    args.noise_steps=params['noise_steps']
    args.image_gen_n=params['image_gen_per_epoch']

    # params datapaths
    current_directory = os.getcwd() #parameters
    datapath = params['dataset_dir']
    modelpath= os.path.join(current_directory,f"models\DDPM_Uncondtional_{dataset_name}_{unet_variant}\ckpt_{dataset_name}_{unet_variant}.pt")
    args.dataset_path = datapath

    # Set filters
    if params['f_kernel'] is None:
        f_settings=None
    else:
        f_settings={}
        f_settings['kernel_size']=params['f_kernel']
        f_settings['kaiser_beta']=params['f_beta']
        f_settings['omega_c_down'] =params['f_down']
        f_settings['omega_c_up'] = params['f_up']

    # save training data images
    save_training_dataset=params['save_trining']
    tr_data_save_dir=os.path.join(current_directory,f"images\original\{dataset_name}")

    # params for gen images
    gen_savepath = os.path.join(current_directory,f"images\generated\{dataset_name}_{unet_variant}")
    gen_per_batch=params['gen_per_batch']
    total_gen=params['gen_total']

    images_per_collage=params['collage_n_per_image']
    total_image_collage=params['collage_n']
    logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

    random_seed=params['seed']
    set_seed(random_seed)

    if torch.cuda.is_available():
        print("CUDA is available. Device:", torch.cuda.get_device_name(0))
    else:
        print("CUDA is not available.")

    all_settings={
        'unet_v': unet_variant,
        'run_name': args.run_name,
        'epochs':args.epochs,
        'batch_size ':args.batch_size,
        'image_size':args.image_size,
        'image_channels':args.image_channels,
        'device':args.device,
        'lr':args.lr,
        'noise_steps':args.noise_steps,
        'image_gen_n':args.image_gen_n,
        'datapath':args.dataset_path,
        'modelpath':modelpath,
        'save_tr_data':save_training_dataset,
        'tr_save_path':tr_data_save_dir,
        'gen_savepath':gen_savepath,
        'gen_per_batch':gen_per_batch,
        'total_gen': total_gen,
        'seed': random_seed,
        'collage_n_per_image': images_per_collage,
        'collage_n': total_image_collage,
        'dataset': dataset_name
    }

    if f_settings is not None:
        all_settings['kernel_size']=f_settings['kernel_size']
        all_settings['kaiser_beta']=f_settings['kaiser_beta']
        all_settings['omega_c_down']=f_settings['omega_c_down']
        all_settings['omega_c_up']=f_settings['omega_c_up']
    else:
        all_settings['kernel_size']="None"
        all_settings['kaiser_beta']="None"
        all_settings['omega_c_down']="None"
        all_settings['omega_c_up']="None"

    # Format the dictionary as a string
    formatted_settings = "\n".join([f"{key}: {value}" for key, value in all_settings.items()])

    # Print the formatted dictionary
    print(formatted_settings)

    # Save the formatted string to a text file
    setting_savepath= os.path.join(current_directory,f"runs/DDPM_Uncondtional_{dataset_name}_{unet_variant}")
    os.makedirs(setting_savepath, exist_ok=True)

    # Save the formatted string to a text file in the specified directory
    with open(os.path.join(setting_savepath, f"settings_{dataset_name}_{unet_variant}.txt"), "w") as file:
        file.write(formatted_settings)

    #Filters

    if f_settings is not None:
        filter_size=f_settings['kernel_size']
        beta=f_settings['kaiser_beta']
        omega_c = f_settings['omega_c_down'] # Cutoff frequency in radians <= pi

        filters=[]
        filters.append( jinc_filter_2d(filter_size, beta))
        filters.append(circularLowpassKernel(omega_c, filter_size))
        filters.append(circularLowpassKernel(omega_c, filter_size,beta=beta))

        for filter in filters:  
            plot_filter_and_response(filter)

    # test model 

    net = UNet(c_in=args.image_channels, c_out=args.image_channels,
            image_size=args.image_size,f_settings=f_settings,device="cpu",variant=unet_variant)
    #net = UNet_conditional(num_classes=10, device="cpu")
    print(sum([p.numel() for p in net.parameters()]))
    x = torch.randn(2, args.image_channels, args.image_size, args.image_size)
    t = x.new_tensor([500] * x.shape[0]).long()
    y = x.new_tensor([1] * x.shape[0]).long()
    #print(net(x, t, y).shape) # conditional case
    print(net(x, t).shape)

    #test noise
    args_temp = argument()
    args_temp.batch_size = 1
    args_temp.image_size = args.image_size
    args_temp.image_channels= args.image_channels
    args_temp.device = args.device
    args_temp.lr = args.lr
    args_temp.dataset_path = args.dataset_path
    args_temp.noise_steps=args.noise_steps

    set_seed(random_seed)

    if dataset_name=="MNIST":
            dataloader, dataset = get_data_MNIST(args_temp)
    else:
        dataloader, dataset = get_data(args_temp)
    image = next(iter(dataloader))[0]
    image = image.to(args_temp.device)
    t = torch.Tensor(np.round(np.linspace(0,args_temp.noise_steps-1,9))).long().to(args_temp.device)
    diffusion = Diffusion(noise_steps=args_temp.noise_steps,img_size=args_temp.image_size, device=args_temp.device)
    noised_image, _ = diffusion.noise_images(image, t)
    noised_image = (noised_image.clamp(-1, 1) + 1) / 2
    noised_image = (noised_image * 255).type(torch.uint8)
    plot_images(noised_image)


    # test filter
    if f_settings is not None:
        # load an image
        set_seed(random_seed)
        if dataset_name=="MNIST":
            dataloader, dataset = get_data_MNIST(args_temp)
        else:
            dataloader, dataset = get_data(args_temp)
        image = next(iter(dataloader))[0]
        x=image


        images=[]
        images.append(image_data(x)) # original

        #filter params
        omega_c_down=f_settings['omega_c_down']
        omega_c_up=f_settings['omega_c_up']
        filter_size=f_settings['kernel_size']
        beta=f_settings['kaiser_beta']

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

    # test no filter
    if f_settings is not None:
        # load an image
        set_seed(random_seed)
        if dataset_name=="MNIST":
            dataloader, dataset = get_data_MNIST(args_temp)
        else:
            dataloader, dataset = get_data(args_temp)
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





    # train model
    set_seed(random_seed)
    if dataset_name=="MNIST":
            dataloader, dataset = get_data_MNIST(args)
    else:
        dataloader, dataset = get_data(args)
    model = UNet(c_in=args.image_channels, c_out=args.image_channels,
                image_size=args.image_size,f_settings=f_settings,device=args.device,variant=unet_variant).to(args.device)
    diffusion = Diffusion(noise_steps=args.noise_steps,img_size=args.image_size, device=args.device)
    loss_all=train(args,model_path=modelpath,dataloader=dataloader,model=model,diffusion=diffusion)

    # inspect training loss
    plot_loss(loss_all)
    with open(os.path.join(setting_savepath, f"trining_loss_MNIST_{unet_variant}.csv"), "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(loss_all)  # Write the list as a single row in the CSV file
    
    #free space
    torch.cuda.empty_cache()
    gc.collect()
    
    # load model
    set_seed(random_seed)
    model = UNet(c_in=args.image_channels, c_out=args.image_channels,
                image_size=args.image_size,f_settings=f_settings,device=args.device,variant=unet_variant).to(args.device)
    ckpt = torch.load(modelpath)
    model.load_state_dict(ckpt)
    diffusion = Diffusion(noise_steps=args.noise_steps,img_size=args.image_size, device=args.device)

    # sample images
    x = diffusion.sample(model, n=6,image_channels=args.image_channels)
    plot_images(x)

    # denoise image
    set_seed(random_seed)
    denoise_img = diffusion.revert(model, n=1,image_channels=args.image_channels)
    plot_images(denoise_img)
    denoise_img.shape

    # load and save training images

    if save_training_dataset & (dataset_name=="MNIST"):
        _, dataset = get_data_MNIST(args)
        save_dataset_MNIST(tr_data_save_dir,dataset)
    else:
        print('skipped saving training dataset')
    
    # generate images and save
    fileno_start=np.arange(0,total_gen,gen_per_batch)
    for start_no in fileno_start:
        fileno=np.arange(start_no,start_no+gen_per_batch,1)
        x = diffusion.sample(model, n=gen_per_batch,image_channels=args.image_channels)
        save_gen_images(gen_savepath,x,fileno)

    #make collage
    make_collage(gen_savepath,gen_savepath,images_per_collage,total_image_collage,args.image_size)

    #free space
    torch.cuda.empty_cache()
    gc.collect()
