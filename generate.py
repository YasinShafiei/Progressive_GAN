"""
Training of Pro-GAN 
Using WGAN-GP loss
"""
# import all libraries
import torch
from torch import optim
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import *
from model import Generator, Discriminator
from math import log2
from tqdm import tqdm
import config

# turn on torch cudnn benchmark
torch.backends.cudnn.benchmarks = True

def get_loader(image_size):
    """
    load the Celeb-Hq dataset
    """
    # define transforms
    transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Normalize(
                [0.5 for _ in range(config.CHANNELS_IMG)],
                [0.5 for _ in range(config.CHANNELS_IMG)],
        )])

    # define the batch size 
    batch_size = config.BATCH_SIZES[int(log2(image_size / 4))]

    # create the data loader
    dataset = datasets.ImageFolder(root=config.DATASET, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True)

    return loader, dataset

def train_function(critic, gen, loader, dataset, step, alpha, opt_critic, opt_gen, tensorboard_step, writer, scaler_gen, scaler_critic):
    """
    This function will perform training steps 
    """
    loop =  tqdm(loader, leave=True)

    # loop over all dataset
    for batch_idx, (real, _) in enumerate(loop):    
        real = real.to(config.DEVICE)
        current_batch_size = real.shape[0]

        # define noise 
        noise = torch.randn(current_batch_size, config.Z_DIM, 1, 1).to(config.DEVICE)

        ### TRAIN DISCRIMINATOR ###
        with torch.cuda.amp.autocast():
            # generate fake images
            fake = gen(noise, alpha, step)
            # make critic predictions
            critic_real = critic(real, alpha, step)
            critic_fake = critic(fake.detach(), alpha, step)
            # calculate gradient penalty
            gp = gradient_penalty(critic, real, fake, alpha, step, device=config.DEVICE)

            # calculate loss of critic 
            loss_critic = (
                -(torch.mean(critic_real) - torch.mean(critic_fake))
                + config.LAMBDA_GP * gp
                + (0.001 * torch.mean(critic_real ** 2))
            )

        # apply optimizer and scaler
        opt_critic.zero_grad()
        scaler_critic.scale(loss_critic).backward()
        scaler_critic.step(opt_critic)
        scaler_critic.update()

        ### TRAIN GENERATOR ### 
        with torch.cuda.amp.autocast():
            # make critic prediction
            gen_fake = critic(fake, alpha, step)
            # calculate generator loss
            loss_gen = -torch.mean(gen_fake)

        # apply optimizer and scaler
        opt_gen.zero_grad()
        scaler_gen.scale(loss_gen).backward()
        scaler_gen.step(opt_gen)
        scaler_gen.update()

        # update alpha
        alpha += current_batch_size / ((config.PROGRESSIVE_EPOCHS[step] * 0.5) * len(dataset))
        alpha = min(alpha, 1)

        ### TENSORBOARD PLOTTING ### 
        if batch_idx % 100 == 0:
            with torch.no_grad():
                # generate fixed images
                fixed_fakes = gen(config.FIXED_NOISE, alpha, step) * 0.5 + 0.5

            # plot to tensorbiard
            plot_to_tensorboard(writer, loss_critic.item(), loss_gen.item(), real.detach(), fixed_fakes.detach(), tensorboard_step)
            tensorboard_step += 1

        loop.set_postfix(gp=gp.item(), loss_critic=loss_critic.item())

    return tensorboard_step, alpha


def main():
    """
    This function perform all main operation of the program
    """
    # define generator and discriminator
    generator =  Generator(config.Z_DIM, config.IN_CHANNELS, img_channels=config.CHANNELS_IMG).to(config.DEVICE)
    discriminator = Discriminator(config.Z_DIM, config.IN_CHANNELS, img_channels=config.CHANNELS_IMG).to(config.DEVICE)

    # initialize optimizerz and scalers
    opt_gen = optim.Adam(generator.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.99))
    opt_disc = optim.Adam(discriminator.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.99))
    scaler_gen = torch.cuda.amp.GradScaler()
    scaler_disc = torch.cuda.amp.GradScaler()

    # *for Tensorboard plotting
    writer = SummaryWriter(f"logs/gan1")

    # load model if loading was True
    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN, generator, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC, discriminator, opt_disc, config.LEARNING_RATE,
        )
        print("Models loaded!")

    # train the models


    # do the training loop
    for num_epochs in config.PROGRESSIVE_EPOCHS[step:]:
        alpha = 1e-5  # start with very low alpha
        loader, dataset = get_loader(4 * 2 ** step)  
        print(f"Current image size: {4 * 2 ** step}")

        # TensorBoard and training
        for epoch in range(num_epochs):
            print(f"Epoch [{epoch+1}/{num_epochs}]")
            tensorboard_step, alpha = train_function(
                discriminator,
                generator,
                loader,
                dataset,
                alpha,
                opt_disc,
                opt_gen,
                tensorboard_step,
                writer,
                scaler_gen,
                scaler_disc,
            )

            # save the model
            if config.SAVE_MODEL:
                save_checkpoint(generator, opt_gen, filename=config.CHECKPOINT_GEN)
                save_checkpoint(discriminator, opt_disc, filename=config.CHECKPOINT_CRITIC)

        step += 1

if __name__ == "__main__":
    main()