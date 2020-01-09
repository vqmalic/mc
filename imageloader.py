import os
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

class SELoader(object):
    def __init__(self, imagedir, maskdir, pipeline):
        self.imagedir = imagedir
        self.maskdir = maskdir
        self.filenames = [x.split(".")[0] for x in os.listdir(imagedir)]
        self.p = pipeline
        self.tt = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    def __len__(self):
        return(len(self.filenames))
    
    def __getitem__(self, idx):
        name = self.filenames[idx]
        img = Image.open(os.path.join(self.imagedir, "{}.png".format(name)))
        mask = Image.open(os.path.join(self.maskdir, "{}.png".format(name)))
        mask = np.array(mask)
        mask -= 255
        mask = -mask
        mask = Image.fromarray(mask.astype("uint8"))
        add = np.random.randint(0, 256, size=(img.height, img.width, 3))
        add = Image.fromarray(add.astype("uint8"))
        img.paste(add, (0, 0), mask)
        img1 = img.copy()
        img2 = img1.copy()
        imgs = [img1, img2]
        out = []
        for img in imgs:
            if self.p:
                for operation in self.p.operations:             
                    r = np.random.random()
                    if r <= operation.probability:
                        img = operation.perform_operation([img])[0]
            img = self.tt(img)
            img = self.normalize(img)
            out.append(img)
        return (out, name)