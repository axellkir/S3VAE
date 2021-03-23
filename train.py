import torch
import yaml
import tqdm
import numpy as np
import wandb

from utils.loader import get_loader
from models.S3VAE import S3VAE
device = 'cuda:1'


if __name__ == '__main__':
    with open('configs/config.yml', 'r') as stream:
        config = yaml.safe_load(stream)
    train_data_loader = get_loader('data/train_new/', n_jobs=0, t=config['train']['T'],
                                   batch_size=config['train']['batch_size'])
    val_data_loader = get_loader('data/val_new/', n_jobs=0, t=config['train']['T'],
                                 batch_size=config['train']['batch_size'])
    vae = S3VAE(config=config, device=device)
    if config['load']:
        vae.load(config['path'])
    if config['wdb']:
        wandb.init(project="S3VAE", group="Position")
    train_steps = 0
    val_steps = 0
    for i in tqdm.trange(config['train']['num_epochs']):
        # train
        for batch in train_data_loader:
            train_steps += 1
            if train_steps % config['train']['num_steps'] == 0:
                break
            images = batch['image'].to(device)
            rewards = batch['position'].to(device)
            # print(torch.from_numpy(np.random.permutation(len(images))))
            permuted = images[torch.from_numpy(np.random.permutation(len(images)))].to(device)
            losses = vae.train_step(images, rewards, permuted)
            if config['wdb']:
                wandb.log({"train_loss/loss": losses[0].cpu().detach().numpy(),
                           "train_loss/vae": losses[1].cpu().detach().numpy(),
                           "train_loss/scc": losses[2].cpu().detach().numpy(),
                           "train_loss/dfp": losses[3].cpu().detach().numpy(),
                           "train_loss/mi": losses[4].cpu().detach().numpy()})
        # # validate
        for batch in val_data_loader:
            val_steps += 1
            if val_steps % 2 == 0:
                break
            images = batch['image'].to(device)
            rewards = batch['position'].to(device)
            permuted = images[torch.from_numpy(np.random.permutation(len(images)))].to(device)
            losses = vae.validate_step(images, rewards, permuted)
            if config['wdb']:
                wandb.log({"val_loss/loss": losses[0].cpu().detach().numpy(),
                           "val_loss/vae": losses[1].cpu().detach().numpy(),
                           "val_loss/scc": losses[2].cpu().detach().numpy(),
                           "train_loss/dfp": losses[3].cpu().detach().numpy(),
                           "val_loss/mi": losses[4].cpu().detach().numpy()})
                # video
                video = torch.cat((losses[5].detach()[0], images.cpu().detach()[0]), dim=2)
                wandb.log({"video": wandb.Video((video.numpy()*255).astype(np.uint8), fps=4, format="gif")})
        vae.save(config['path'])










