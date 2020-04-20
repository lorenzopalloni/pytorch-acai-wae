import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path

from config import data_dir, knobs
from models import FastEncoder, FastDecoder
from preprocessing import get_loader, inv_standardize, standardize

fid_dir = data_dir / 'fid_dir'
fid_dir.mkdir(exist_ok=True)
real_dir = fid_dir / 'real_images'
real_dir.mkdir(exist_ok=True)
reconstructed_dir = fid_dir / 'reconstructed_images'
reconstructed_dir.mkdir(exist_ok=True)
num_obs = 10_000

#counter = 0
#iterable = iter(get_loader())
#for i in range(num_obs // knobs["batch_size"]):
#    batch = next(iterable)
#    for j in range(knobs["batch_size"]):
#        torchvision.utils.save_image(inv_standardize(batch[i]), real_dir / f'{counter}.jpg')
#        counter += 1

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


encoder = FastEncoder()
decoder = FastDecoder()
checkpoint_dir = '/home/lore/Projects/pytorch-acai-wae/mnist/best_models/acwwai/16-04-00-02-05.pt'
checkpoint = torch.load(checkpoint_dir)
encoder.load_state_dict(checkpoint["encoder_state_dict"])
decoder.load_state_dict(checkpoint["decoder_state_dict"])
encoder.eval()
decoder.eval()

all_images = list(real_dir.iterdir())
len_all = len(all_images)
batch_size = 20


for img in real_dir.iterdir():
    x = standardize(
        torch.tensor(
                rgb2gray(plt.imread(img)), dtype=torch.double
            ).unsqueeze(0).unsqueeze(0)
        )
    codes = encoder(x)
    y = decoder(codes)
    torchvision.utils.save_image(inv_standardize(y.squeeze().unsqueeze(0)), reconstructed_dir / img)


'''
for i in range(len_all // batch_size):
    filenames = []
    batch = []
    for i in range(batch_size):
        batch.append(standardize(
            torch.tensor(
                rgb2gray(plt.imread(all_images[i])), dtype=torch.double
            ).unsqueeze(0))
        )
        filenames.append(all_images[i])
    x = torch.stack(batch)
    codes = encoder(x)
    y = decoder(codes)
    for num, el in enumerate(y):
        torchvision.utils.save_image(inv_standardize(y.squeeze()), reconstructed_dir / filenames[num])
'''
