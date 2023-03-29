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

# Configure Variables and Hyperparameters
START_TRAIN_AT_IMG_SIZE = 128
DATASET = 'D:\\python scripts\\pytorch\\GANS\\Pro GAN\\data\\celeba_hq\\train'
CHECKPOINT_GEN = "saved_models\\generator.pth"
CHECKPOINT_CRITIC = "saved_models\\critic.pth"
DEVICE = "cuda"
SAVE_MODEL = True
LOAD_MODEL = True
LEARNING_RATE = 1e-3
BATCH_SIZES = [32, 32, 32, 16, 16, 16, 16, 8, 4]
CHANNELS_IMG = 3
Z_DIM = 256  # should be 512 in original paper
IN_CHANNELS = 256  # should be 512 in original paper
CRITIC_ITERATIONS = 1
LAMBDA_GP = 10
PROGRESSIVE_EPOCHS = [80] * len(BATCH_SIZES)
FIXED_NOISE = torch.randn(8, Z_DIM, 1, 1).to(DEVICE)
NUM_WORKERS = 4

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
                [0.5 for _ in range(CHANNELS_IMG)],
                [0.5 for _ in range(CHANNELS_IMG)],
        )])

    # define the batch size 
    batch_size = BATCH_SIZES[int(log2(image_size / 4))]

    # create the data loader
    dataset = datasets.ImageFolder(root=DATASET, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

    return loader, dataset

def train_function(critic, gen, loader, dataset, step, alpha, opt_critic, opt_gen, tensorboard_step, writer, scaler_gen, scaler_critic):
    """
    This function will perform training steps 
    """
    loop =  tqdm(loader, leave=True)

    # loop over all dataset
    for batch_idx, (real, _) in enumerate(loop):    
        real = real.to(DEVICE)
        current_batch_size = real.shape[0]

        # define noise 
        noise = torch.randn(current_batch_size, Z_DIM, 1, 1).to(DEVICE)

        ### TRAIN DISCRIMINATOR ###
        with torch.cuda.amp.autocast():
            # generate fake images
            fake = gen(noise, alpha, step)
            # make critic predictions
            critic_real = critic(real, alpha, step)
            critic_fake = critic(fake.detach(), alpha, step)
            # calculate gradient penalty
            gp = gradient_penalty(critic, real, fake, alpha, step, device=DEVICE)

            # calculate loss of critic 
            loss_critic = (
                -(torch.mean(critic_real) - torch.mean(critic_fake))
                + LAMBDA_GP * gp
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
        alpha += current_batch_size / ((PROGRESSIVE_EPOCHS[step] * 0.5) * len(dataset))
        alpha = min(alpha, 1)

        ### TENSORBOARD PLOTTING ### 
        if batch_idx % 100 == 0:
            with torch.no_grad():
                # generate fixed images
                fixed_fakes = gen(FIXED_NOISE, alpha, step) * 0.5 + 0.5

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
    generator =  Generator(Z_DIM, IN_CHANNELS, img_channels=CHANNELS_IMG).to(DEVICE)
    discriminator = Discriminator(Z_DIM, IN_CHANNELS, img_channels=CHANNELS_IMG).to(DEVICE)

    # initialize optimizerz and scalers
    opt_gen = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.99))
    opt_disc = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.99))
    scaler_gen = torch.cuda.amp.GradScaler()
    scaler_disc = torch.cuda.amp.GradScaler()

    # *for Tensorboard plotting
    writer = SummaryWriter(f"logs/gan1")

    # load model if loading was True
    if LOAD_MODEL:
        load_checkpoint(
            CHECKPOINT_GEN, generator, opt_gen, LEARNING_RATE,
        )
        load_checkpoint(
            CHECKPOINT_CRITIC, discriminator, opt_disc, LEARNING_RATE,
        )
        print("Models loaded!")

    # train the models
    generator.train()
    discriminator.train()

    tensorboard_step = 0

    ### START TRAINING ###
    # define the step
    step = int(log2(START_TRAIN_AT_IMG_SIZE / 4))
    # un-comment the generate example line if you want to save generated images.
    #To save then you have to create a new folder called "saved_examples"
    # generate_examples(generator, step)

    # do the training loop
    for num_epochs in PROGRESSIVE_EPOCHS[step:]:
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
                step,
                alpha,
                opt_disc,
                opt_gen,
                tensorboard_step,
                writer,
                scaler_gen,
                scaler_disc,
            )

            # save the model
            if SAVE_MODEL:
                save_checkpoint(generator, opt_gen, filename=CHECKPOINT_GEN)
                save_checkpoint(discriminator, opt_disc, filename=CHECKPOINT_CRITIC)

        step += 1

if __name__ == "__main__":
    main()