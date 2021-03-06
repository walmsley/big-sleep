import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch_optimizer import AdamP

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

# tensor helpers

def differentiable_topk(x, k, temperature=1.):
    n, dim = x.shape
    topk_tensors = []

    for i in range(k):
        is_last = i == (k - 1)
        values, indices = (x / temperature).softmax(dim=-1).topk(1, dim=-1)
        topks = torch.zeros_like(x).scatter_(-1, indices, values)
        topk_tensors.append(topks)
        if not is_last:
            x = x.scatter(-1, indices, float('-inf'))

    topks = torch.cat(topk_tensors, dim=-1)
    return topks.reshape(n, k, dim).sum(dim = 1)

# load clip

perceptor, preprocess = load()

# load biggan

class Latents(torch.nn.Module):
    def __init__(
        self,
        num_latents = 15,
        num_classes = 1000,
        z_dim = 128,
        max_classes = None,
        class_temperature = 2.,
        class_mean = -3.9,
        class_std = 0.3,
    ):
        super().__init__()
        self.normu = torch.nn.Parameter(torch.zeros(num_latents, z_dim).normal_(std = 1))
        self.cls = torch.nn.Parameter(torch.zeros(num_latents, num_classes).normal_(mean = class_mean, std = class_std))
        self.register_buffer('thresh_lat', torch.tensor(1))

        assert not exists(max_classes) or max_classes > 0 and max_classes <= num_classes, f'max_classes must be between 0 and {num_classes}'
        self.max_classes = max_classes
        self.class_temperature = class_temperature

    def forward(self):
        if exists(self.max_classes):
            classes = differentiable_topk(self.cls, self.max_classes, temperature = self.class_temperature)
        else:
            classes = torch.sigmoid(self.cls)

        return self.normu, classes

class Model(nn.Module):
    def __init__(
        self,
        image_size,
        max_classes = None,
        class_temperature = 2.,
        class_mean = -3.9,
        class_std = 0.3,
    ):
        super().__init__()
        assert image_size in (128, 256, 512), 'image size must be one of 128, 256, or 512'
        self.biggan = BigGAN.from_pretrained(f'biggan-deep-{image_size}')

        self.max_classes = max_classes
        self.class_temperature = class_temperature
        self.class_mean = class_mean
        self.class_std = class_std
        self.init_latents()

    def init_latents(self):
        self.latents = Latents(
            num_latents = len(self.biggan.config.layers) + 1,
            num_classes = self.biggan.config.num_classes,
            z_dim = self.biggan.config.z_dim,
            max_classes = self.max_classes,
            class_temperature = self.class_temperature,
            class_mean = self.class_mean,
            class_std = self.class_std,
        )

    def set_latents(self, latents):
        self.latents = latents

    def forward(self):
        self.biggan.eval()
        out = self.biggan(*self.latents(), 1)
        return (out + 1) / 2

# load siren

class BigSleep(nn.Module):
    def __init__(
        self,
        num_cutouts = 128,
        loss_coef = 100,
        image_size = 512,
        bilinear = False,
        max_classes = None,
        class_temperature = 2.,
        experimental_resample = False,
        class_mean = -3.9,
        class_std = 0.3,
    ):
        super().__init__()
        self.loss_coef = loss_coef
        self.image_size = image_size
        self.num_cutouts = num_cutouts
        self.experimental_resample = experimental_resample

        self.interpolation_settings = {'mode': 'bilinear', 'align_corners': False} if bilinear else {'mode': 'nearest'}

        self.model = Model(
            image_size = image_size,
            max_classes = max_classes,
            class_temperature = class_temperature,
            class_mean = class_mean,
            class_std = class_std,
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
            if self.num_cutouts > 1:
                size = int(width * torch.zeros(1,).normal_(mean=.8, std=.3).clip(.4375, .998)) #224 to 511
                offsetx = torch.randint(0, width - size, ())
                offsety = torch.randint(0, width - size, ())
            else:
                size = width
                offsetx = 0
                offsety = 0
            apper = out[:, :, offsetx:offsetx + size, offsety:offsety + size]
            if (self.experimental_resample):
                apper = resample(apper, (224, 224))
            else:
                apper = F.interpolate(apper, (224, 224), **self.interpolation_settings)
            pieces.append(apper)

        into = torch.cat(pieces)
        into = normalize_image(into)

        image_embed = perceptor.encode_image(into)

        latents, soft_one_hot_classes = self.model.latents()
        num_latents = latents.shape[0]
        latent_thres = self.model.latents.thresh_lat

        lat_loss =  torch.abs(1 - torch.std(latents, dim=1)).mean() + \
                    torch.abs(torch.mean(latents, dim = 1)).mean() + \
                    4 * torch.max(torch.square(latents).mean(), latent_thres)

        for array in latents:
            mean = torch.mean(array)
            diffs = array - mean
            var = torch.mean(torch.pow(diffs, 2.0))
            std = torch.pow(var, 0.5)
            zscores = diffs / std
            skews = torch.mean(torch.pow(zscores, 3.0))
            kurtoses = torch.mean(torch.pow(zscores, 4.0)) - 3.0

            lat_loss = lat_loss + torch.abs(kurtoses) / num_latents + torch.abs(skews) / num_latents

        cls_loss = ((50 * torch.topk(soft_one_hot_classes, largest = False, dim = 1, k = 999)[0]) ** 2).mean()
        cls_sum_loss = torch.abs(torch.sum(soft_one_hot_classes, dim=1) - 1.).mean()

        sim_loss = -self.loss_coef * torch.cosine_similarity(text_embed, image_embed, dim = -1).mean()
        return (lat_loss, cls_loss, sim_loss, cls_sum_loss)

class Imagine(nn.Module):
    def __init__(
        self,
        text,
        *,
        lr = .07,
        image_size = 512,
        gradient_accumulate_every = 1,
        save_every = 50,
        epochs = 20,
        iterations = 1050,
        save_progress = False,
        bilinear = False,
        open_folder = True,
        seed = None,
        torch_deterministic = False,
        max_classes = None,
        class_temperature = 2.,
        save_date_time = False,
        save_best = False,
        experimental_resample = False,
        textpath = None,
        num_cutouts = 128,
        use_adamp = False,
        scale_loss = (1.,1.,1.,0.),
        class_mean = -3.9,
        class_std = 0.3,
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
            max_classes = max_classes,
            class_temperature = class_temperature,
            experimental_resample = experimental_resample,
            num_cutouts = num_cutouts,
            class_mean = class_mean,
            class_std = class_std,
        ).cuda()

        self.model = model

        self.lr = lr
        self.use_adamp = use_adamp
        self.reset_optimizer()

        self.gradient_accumulate_every = gradient_accumulate_every
        self.save_every = save_every

        self.save_progress = save_progress
        self.save_date_time = save_date_time

        self.save_best = save_best
        self.current_best_score = 0

        self.scale_loss = scale_loss

        self.open_folder = open_folder
        self.total_image_updates = (self.epochs * self.iterations) / self.save_every

        self.set_text(text)
        if textpath is not None:
            self.set_textpath(textpath)

    def set_text(self, text):
        self.text = text
        textpath = self.text.replace(' ','_')[:255]
        if self.save_date_time:
            textpath = datetime.now().strftime("%y%m%d-%H%M%S-") + textpath

        self.textpath = textpath
        self.filename = Path(f'./{textpath}.png')
        encoded_text = tokenize(text).cuda()
        self.encoded_text = perceptor.encode_text(encoded_text).detach()

    def set_textpath(self, textpath):
        self.textpath = textpath
        self.filename = Path(f'./{textpath}.png')

    def set_latents(self, latents):
        self.model.model.set_latents(latents)
        self.reset_optimizer()

    def reset_optimizer(self):
        if self.use_adamp:
            self.optimizer = AdamP(self.model.model.latents.parameters(), self.lr)
        else:
            self.optimizer = Adam(self.model.model.latents.parameters(), self.lr)

    def reset(self):
        self.model.reset()
        self.model = self.model.cuda()
        self.reset_optimizer()

    def train_step(self, epoch, i, pbar=None):
        total_loss = 0

        for _ in range(self.gradient_accumulate_every):
            losses = self.model(self.encoded_text)
            loss = sum([loss*self.scale_loss[i] for i,loss in enumerate(losses)]) / self.gradient_accumulate_every
            total_loss += loss
            loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()

        if (i + 1) % self.save_every == 0:
            with torch.no_grad():
                print('losses', [loss.item() for loss in losses])
                top_score = losses[2]
                image = self.model.model()[0].cpu()

                save_image(image, str(self.filename))
                if pbar is not None:
                    pbar.update(1)
                else:
                    print(f'image updated at "./{str(self.filename)}"')


                if self.save_progress:
                    total_iterations = epoch * self.iterations + i
                    #num = total_iterations // self.save_every
                    save_image(image, Path(f'./{self.textpath}.{total_iterations}.png'))
                    torch.save(self.model.model.latents, Path(f'./{self.textpath}.{total_iterations}.pth'))

                if self.save_best and top_score.item() < self.current_best_score:
                    self.current_best_score = top_score.item()
                    save_image(image, Path(f'./{self.textpath}.best.png'))
                    torch.save(self.model.model.latents, Path(f'./{self.textpath}.best.pth'))

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
