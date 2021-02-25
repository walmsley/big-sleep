import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam

import torchvision
from torchvision.utils import save_image

import os
import sys
import subprocess
import signal
from datetime import datetime
from pathlib import Path
from tqdm import tqdm, trange
from collections import namedtuple

from big_sleep.biggan import BigGAN
from big_sleep.clip import load, tokenize, normalize_image
from big_sleep.dall_e import load_model

import io, requests

from einops import rearrange

from .resample import resample

assert torch.cuda.is_available(), 'CUDA must be available in order to use Deep Daze'

# graceful keyboard interrupt

terminate = False

def signal_handling(signum,frame):
    global terminate
    terminate = True

signal.signal(signal.SIGINT,signal_handling)

# helpers

def exists(val):
    return val is not None

def open_folder(path):
    if os.path.isfile(path):
        path = os.path.dirname(path)

    if not os.path.isdir(path):
        return

    cmd_list = None
    if sys.platform == 'darwin':
        cmd_list = ['open', '--', path]
    elif sys.platform == 'linux2' or sys.platform == 'linux':
        cmd_list = ['xdg-open', path]
    elif sys.platform in ['win32', 'win64']:
        cmd_list = ['explorer', path.replace('/','\\')]
    if cmd_list == None:
        return

    try:
        subprocess.check_call(cmd_list)
    except subprocess.CalledProcessError:
        pass
    except OSError:
        pass

# load clip

perceptor, preprocess = load()

# load dall_e

class Latents(torch.nn.Module):
    def __init__(
        self,
        z_dim = 8192*32*32,
        init_fname = None,
    ):
        super().__init__()
        self.vec = torch.nn.Parameter(torch.zeros(z_dim))

        if init_fname is not None:
            data = torch.load(open(init_fname,'rb'), map_location='cuda:0')
            print(f'loaded data of size {data.size()}')
            with torch.no_grad():
                self.vec.copy_(data)

    def forward(self):
        return self.vec

class Model(nn.Module):
    def __init__(
        self,
        image_size,
        init_fname,
    ):
        super().__init__()
        assert image_size == 256, 'image size must be 256'
        self.dall_e_decoder = load_model("https://cdn.openai.com/dall-e/decoder.pkl", device=torch.device('cuda:0'))
        self.init_fname = init_fname
        self.init_latents()

    def init_latents(self):
        self.latents = Latents(init_fname=self.init_fname)

    def forward(self):
        #self.biggan.eval()
        #out = self.biggan(*self.latents(), 1)
        #return (out + 1) / 2
        reshaped_z = torch.reshape(self.latents(), (1,8192,32,32))
        x_stats = self.dall_e_decoder(reshaped_z).float()
        x_rec = torch.clamp((torch.sigmoid(x_stats[:, :3]) - 0.1) / (1 - 2 * 0.1), 0, 1) #from unmap_pixels function
        return x_rec

# load siren

class BigSleep(nn.Module):
    def __init__(
        self,
        num_cutouts = 16,
        loss_coefs = None,
        image_size = 256,
        bilinear = False,
        experimental_resample = False,
        init_fname = None,
    ):
        super().__init__()
        if loss_coefs is None:
            self.loss_coefs = (100, 0.02, 10., 50., 0.1)
        else:
            self.loss_coefs = loss_coefs
        self.image_size = image_size
        self.num_cutouts = num_cutouts
        self.experimental_resample = experimental_resample

        self.interpolation_settings = {'mode': 'bilinear', 'align_corners': False} if bilinear else {'mode': 'nearest'}

        self.model = Model(
            image_size = image_size,
            init_fname = init_fname,
        )

    def reset(self):
        self.model.init_latents()

    def forward(self, text_embed, return_loss = True):
        width, num_cutouts = self.image_size, self.num_cutouts

        out = self.model()

        if not return_loss:
            return out

        pieces = []
        for ch in range(num_cutouts):
            size = int(width * torch.zeros(1,).normal_(mean=.925, std=.1).clip(.875, .995))
            offsetx = torch.randint(0, width - size, ())
            offsety = torch.randint(0, width - size, ())
            apper = out[:, :, offsetx:offsetx + size, offsety:offsety + size]
            if (self.experimental_resample):
                apper = resample(apper, (224, 224))
            else:
                apper = F.interpolate(apper, (224, 224), **self.interpolation_settings)
            pieces.append(apper)

        into = torch.cat(pieces)
        into = normalize_image(into)

        image_embed = perceptor.encode_image(into)

        latents = self.model.latents()
        num_latents = latents.shape[0]
        latents_reshaped = latents.reshape((8192,32,32))
        ones_32 = torch.ones((32,32)).cuda()
        ones_256 = torch.ones((256,256)).cuda()
        img_grayscale = torch.mean(out, dim=(0,1))

        lat_loss = self.loss_coefs[1] * torch.abs(1024. - torch.sum(latents))
        lat_loss_2 = self.loss_coefs[2] * torch.mean(torch.abs(latents_reshaped.sum(dim=0) - ones_32))
        lat_loss_3 = self.loss_coefs[3] * torch.sum(torch.abs(latents_reshaped.max(dim=0)[0] - ones_32))
        lat_loss_4 = self.loss_coefs[4] * torch.mean(torch.abs(img_grayscale-0.5)**2)#torch.sum(torch.gt(img_grayscale,ones_256-0.1))


        # lat_loss =  torch.abs(1 - torch.std(latents, dim=1)).mean() + \
        #             torch.abs(torch.mean(latents, dim = 1)).mean() + \
        #             4 * torch.max(torch.square(latents).mean(), latent_thres)

        sim_loss = -self.loss_coefs[0] * torch.cosine_similarity(text_embed, image_embed, dim = -1).mean()

        print('losses', lat_loss.item(), lat_loss_2.item(), lat_loss_3.item(), lat_loss_4.item(), sim_loss.item())

        return (lat_loss+lat_loss_2+lat_loss_3+lat_loss_4, sim_loss)

class Imagine(nn.Module):
    def __init__(
        self,
        text,
        *,
        lr = .07,
        image_size = 256,
        gradient_accumulate_every = 1,
        save_every = 50,
        epochs = 20,
        iterations = 1050,
        save_progress = False,
        bilinear = False,
        open_folder = True,
        seed = None,
        torch_deterministic = False,
        save_date_time = False,
        save_best = False,
        experimental_resample = False,
        init_fname = None,
        loss_coefs = None,
    ):
        super().__init__()

        if torch_deterministic:
            assert not bilinear, 'the deterministic (seeded) operation does not work with interpolation (PyTorch 1.7.1)'
            torch.set_deterministic(True)

        if exists(seed):
            print(f'setting seed of {seed}')
            if seed == 0:
                print('you can override this with --seed argument in the command line, or --random for a randomly chosen one')
            torch.manual_seed(seed)


        self.epochs = epochs
        self.iterations = iterations

        model = BigSleep(
            image_size = image_size,
            bilinear = bilinear,
            experimental_resample = experimental_resample,
            init_fname = init_fname,
            loss_coefs = loss_coefs,
        ).cuda()

        self.model = model

        self.lr = lr
        self.optimizer = Adam(model.model.latents.parameters(), lr)
        self.gradient_accumulate_every = gradient_accumulate_every
        self.save_every = save_every

        self.save_progress = save_progress
        self.save_date_time = save_date_time

        self.save_best = save_best
        self.current_best_score = 0

        self.open_folder = open_folder
        self.total_image_updates = (self.epochs * self.iterations) / self.save_every

        self.set_text(text)

    def set_text(self, text):
        self.text = text
        textpath = self.text.replace(' ','_')[:255]
        if self.save_date_time:
            textpath = datetime.now().strftime("%y%m%d-%H%M%S-") + textpath

        self.textpath = textpath
        self.filename = Path(f'./{textpath}.png')
        encoded_text = tokenize(text).cuda()
        self.encoded_text = perceptor.encode_text(encoded_text).detach()

    def reset(self):
        self.model.reset()
        self.model = self.model.cuda()
        self.optimizer = Adam(self.model.model.latents.parameters(), self.lr)

    def train_step(self, epoch, i, pbar=None):
        total_loss = 0

        for _ in range(self.gradient_accumulate_every):
            losses = self.model(self.encoded_text)
            loss = sum(losses) / self.gradient_accumulate_every
            total_loss += loss
            loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()

        if (i + 1) % self.save_every == 0:
            with torch.no_grad():
                top_score = losses[1]
                image = self.model.model()[0].cpu()

                save_image(image, str(self.filename))
                if pbar is not None:
                    pbar.update(1)
                else:
                    print(f'image updated at "./{str(self.filename)}"')


                if self.save_progress:
                    total_iterations = epoch * self.iterations + i
                    num = total_iterations // self.save_every
                    save_image(image, Path(f'./{self.textpath}.{num}.png'))

                if self.save_best and top_score.item() < self.current_best_score:
                    self.current_best_score = top_score.item()
                    save_image(image, Path(f'./{self.textpath}.best.png'))

        return total_loss

    def forward(self):
        print(f'Imagining "{self.text}" from the depths of my weights...')

        self.model(self.encoded_text) # one warmup step due to issue with CLIP and CUDA

        if self.open_folder:
            open_folder('./')
            self.open_folder = False

        image_pbar = tqdm(total=self.total_image_updates, desc='image update', position=2, leave=True)
        for epoch in trange(self.epochs, desc = '      epochs', position=0, leave=True):
            pbar = trange(self.iterations, desc='   iteration', position=1, leave=True)
            image_pbar.update(0)
            for i in pbar:
                loss = self.train_step(epoch, i, image_pbar)
                pbar.set_description(f'loss: {loss.item():04.2f}')

                if terminate:
                    print('detecting keyboard interrupt, gracefully exiting')
                    return
