import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from PIL import Image
import numpy as np

from tqdm import tqdm
from network import Net
from sklearn.manifold import TSNE

from imageloader import SELoader
import Augmentor

import matplotlib.pyplot as plt

from shutil import copyfile

model_path = "result/model.pth"
p = Augmentor.Pipeline()
p.resize(probability=1.0, width=64, height=64)

imagedir = '/media/kashgar/data/pnn_training/pix2pix/A/test'
maskdir = '/media/kashgar/data/pnn_training/pix2pix/B/test'

images = SELoader(imagedir, maskdir, p)

model = Net()
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

data = []
names = []
labels = []

for line in tqdm(images):
	name = line[1]
	names.append(os.path.join(imagedir, "{}.png".format(name)))
	x = line[0][0]
	x = x.view(1, *x.shape)
	feat = model(x)
	data.append(feat.data.numpy()[0])
	labels.append(0)

imagedir = '/media/kashgar/data/pnn_training/pix2pix/__q_frontcenterfoldfull/img'
maskdir = '/media/kashgar/data/pnn_training/pix2pix/__q_frontcenterfoldfull/mask'

images = SELoader(imagedir, maskdir, p)

for line in tqdm(images):
	name = line[1]
	names.append(os.path.join(imagedir, "{}.png".format(name)))
	x = line[0][0]
	x = x.view(1, *x.shape)
	feat = model(x)
	data.append(feat.data.numpy()[0])
	labels.append(1)

imagedir = '/media/kashgar/data/pnn_training/pix2pix/__q_frontcenterfoldpartial/img'
maskdir = '/media/kashgar/data/pnn_training/pix2pix/__q_frontcenterfoldpartial/mask'

images = SELoader(imagedir, maskdir, p)

for line in tqdm(images):
	name = line[1]
	names.append(os.path.join(imagedir, "{}.png".format(name)))
	x = line[0][0]
	x = x.view(1, *x.shape)
	feat = model(x)
	data.append(feat.data.numpy()[0])
	labels.append(2)

imagedir = '/media/kashgar/data/pnn_training/pix2pix/__q_hfel/img'
maskdir = '/media/kashgar/data/pnn_training/pix2pix/__q_hfel/mask'

images = SELoader(imagedir, maskdir, p)

for line in tqdm(images):
	name = line[1]
	names.append(os.path.join(imagedir, "{}.png".format(name)))
	x = line[0][0]
	x = x.view(1, *x.shape)
	feat = model(x)
	data.append(feat.data.numpy()[0])
	labels.append(3)

targets = set([0, 1, 2, 3])
colors = ['r', 'g', 'b', 'c']

ret = TSNE(n_components=2, random_state=0).fit_transform(data)

plt.figure(figsize=(12, 10))

for label in targets:
	idx = np.where(np.array(labels)==label)[0]
	plt.scatter(ret[idx, 0], ret[idx, 1], c=colors[label])

plt.savefig("tmp.png")

target = "/media/kashgar/data/pnn_training/pix2pix/__q_hfel/img/30.png"
data = np.array(data)
t_data = data[names.index(target)]
scores = data.dot(t_data.reshape(-1, 1)).flatten()
top = scores.argsort()[::-1]
top = top[1:10]

copyfile(target, "01_{:03d}.png".format(0))

for i in range(len(top)):
	copyfile(names[top[i]], "01_{:03d}.png".format(i+1))