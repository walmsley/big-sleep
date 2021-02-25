Some current rough notebook code, to be put in a colab and used similar to original BigSleep:

```
import subprocess

CUDA_version = [s for s in subprocess.check_output(["nvcc", "--version"]).decode("UTF-8").split(", ") if s.startswith("release")][0].split(" ")[-1]
print("CUDA version:", CUDA_version)

if CUDA_version == "10.0":
    torch_version_suffix = "+cu100"
elif CUDA_version == "10.1":
    torch_version_suffix = "+cu101"
elif CUDA_version == "10.2":
    torch_version_suffix = ""
else:
    torch_version_suffix = "+cu110"

! pip install torch==1.7.1{torch_version_suffix} torchvision==0.8.2{torch_version_suffix} -f https://download.pytorch.org/whl/torch_stable.html ftfy regex

!pip install git+https://github.com/walmsley/big-sleep.git@dalle --upgrade

pip install git+https://github.com/openai/DALL-E.git

from tqdm import trange
from IPython.display import Image, display
import torch

from big_sleep import Imagine
from google.colab import files
import numpy as np

indices = torch.randint(0,8192, (32,32))
out = torch.zeros((8192,32,32))
for i in range(32):
    for j in range(32):
        out[indices[i,j],i,j] = 1.0
out = torch.reshape(out, (8192*32*32,))
torch.save(out, 'rand.pth')

TEXT = "a photo of spiderman delivering a pizza" #@param {type:"string"}
TEXT = TEXT.replace('_',' ')
SAVE_EVERY =  25#@param {type:"number"}
LEARNING_RATE = 4e-5 #@param {type:"number"}
ITERATIONS = 101 #@param {type:"number"}
NUM_CUTOUTS = 4#@param {type: "number"}
SEND_TO_DRIVE = False#@param {type: "boolean"}
NUM_ATTEMPTS = 1#@param {type: "number"}
SAVE_IMGS = True

for attempt in range(NUM_ATTEMPTS):

    model = Imagine(
        text = TEXT,
        save_every = SAVE_EVERY,
        lr = LEARNING_RATE,
        iterations = ITERATIONS,
        save_progress = SAVE_IMGS,
        save_best = SAVE_IMGS,
        torch_deterministic = True,
        init_fname = 'rand.pth',
        loss_coefs = (100, 0.02, 10., 50., 0.),
    )

    for i in trange(ITERATIONS, desc = 'iteration', position=0, leave=True, mininterval=1.0):
        model.train_step(0, i)

        if i % max(model.save_every, 10) != 0:
           continue
        if i == 0:
            continue

        image = Image(f"./{TEXT.replace(' ', '_')}.png")
        print(f'\n{TEXT} {i}')
        display(image)
```


<img src="./samples/artificial_intelligence.png" width="250px"></img>

*artificial intelligence*

<img src="./samples/cosmic_love_and_attention.png" width="250px"></img>

*cosmic love and attention*

<img src="./samples/fire_in_the_sky.png" width="250px"></img>

*fire in the sky*

<img src="./samples/a_pyramid_made_of_ice.png" width="250px"></img>

*a pyramid made of ice*

<img src="./samples/a_lonely_house_in_the_woods.png" width="250px"></img>

*a lonely house in the woods*

<img src="./samples/marriage_in_the_mountains.png" width="250px"></img>

*marriage in the mountains*

<img src="./samples/a_lantern_dangling_from_the_tree_in_a_foggy_graveyard.png" width="250px"></img>

*lantern dangling from a tree in a foggy graveyard*

<img src="./samples/a_vivid_dream.png" width="250px"></img>

*a vivid dream*

<img src="./samples/balloons_over_the_ruins_of_a_city.png" width="250px"></img>

*balloons over the ruins of a city*

<img src="./samples/the_death_of_the_lonesome_astronomer.png" width="250px"></img>

*the death of the lonesome astronomer* - by <a href="https://github.com/moirage">moirage</a>

<img src="./samples/the_tragic_intimacy_of_the_eternal_conversation_with_oneself.png" width="250px"></img>

*the tragic intimacy of the eternal conversation with oneself* - by <a href="https://github.com/moirage">moirage</a>

## Big Sleep

<a href="https://twitter.com/advadnoun">Ryan Murdock</a> has done it again, combining OpenAI's <a href="https://github.com/openai/CLIP">CLIP</a> and the generator from a <a href="https://arxiv.org/abs/1809.11096">BigGAN</a>! This repository wraps up his work so it is easily accessible to anyone who owns a GPU.

You will be able to have the GAN dream up images using natural language with a one-line command in the terminal.

Original notebook [![Open In Colab][colab-badge]][colab-notebook]

Simplified notebook [![Open In Colab][colab-badge]][colab-notebook-2]

[colab-notebook]: <https://colab.research.google.com/drive/1NCceX2mbiKOSlAd_o7IU7nA9UskKN5WR?usp=sharing>
[colab-notebook-2]: <https://colab.research.google.com/drive/1MEWKbm-driRNF8PrU7ogS5o3se-ePyPb?usp=sharing>
[colab-badge]: <https://colab.research.google.com/assets/colab-badge.svg>

## Install

```bash
$ pip install big-sleep
```

## Usage

```bash
$ dream "a pyramid made of ice"
```

Images will be saved to wherever the command is invoked

## Advanced

You can invoke this in code with

```python
from big_sleep import Imagine

dream = Imagine(
    text = "fire in the sky",
    lr = 5e-2,
    save_every = 25,
    save_progress = True
)

dream()
```

You can also set a new text by using the `.set_text(<str>)` command

```python
dream.set_text("a quiet pond underneath the midnight moon")
```

And reset the latents with `.reset()`

```python
dream.reset()
```

To save the progression of images during training, you simply have to supply the `--save-progress` flag

```bash
$ dream "a bowl of apples next to the fireplace" --save-progress --save-every 100
```

Due to the class conditioned nature of the GAN, Big Sleep often steers off the manifold into noise. You can use a flag to save the best high scoring image (per CLIP critic) to `{filepath}.best.png` in your folder.

```bash
$ dream "a room with a view of the ocean" --save-best
```

## Experimentation

You can set the number of classes that you wish to restrict Big Sleep to use for the Big GAN with the `--max-classes` flag as follows (ex. 15 classes). This may lead to extra stability during training, at the cost of lost expressivity.

```bash
$ dream 'a single flower in a withered field' --max-classes 15
```

## Alternatives

<a href="https://github.com/lucidrains/deep-daze">Deep Daze</a> - CLIP and a deep SIREN network

## Used By

- <a href="https://dank.xyz/">Dank.xyz</a>

## Citations

```bibtex
@misc{unpublished2021clip,
    title  = {CLIP: Connecting Text and Images},
    author = {Alec Radford, Ilya Sutskever, Jong Wook Kim, Gretchen Krueger, Sandhini Agarwal},
    year   = {2021}
}
```

```bibtex
@misc{brock2019large,
    title   = {Large Scale GAN Training for High Fidelity Natural Image Synthesis}, 
    author  = {Andrew Brock and Jeff Donahue and Karen Simonyan},
    year    = {2019},
    eprint  = {1809.11096},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG}
}
```
