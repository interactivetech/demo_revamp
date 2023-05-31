import os
import shutil
from typing import Any, Dict
from urllib.parse import urlparse
from urllib.request import urlretrieve

import numpy as np
import torch
from torchvision.transforms import Compose, ToTensor
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from PIL import Image

def download_data(download_directory: str, data_config: Dict[str, Any]) -> str:
    if not os.path.exists(download_directory):
        os.makedirs(download_directory, exist_ok=True)
    url = data_config["url"]
    filename = os.path.basename(urlparse(url).path)
    filepath = os.path.join(download_directory, filename)
    if not os.path.exists(filepath):
        urlretrieve(url, filename=filepath)
        shutil.unpack_archive(filepath, download_directory)


def collate_fn(batch):
    return tuple(zip(*batch))


def get_transform():
    transforms = []
    transforms.append(ToTensor())
    return Compose(transforms)


def load_and_transform_image(img_path):
    image = Image.open(img_path).convert("RGB")
    image = get_transform()(image)
    return image


def draw_example(image, labels, title=None):
    fig,ax = plt.subplots(1)
    plt.title(title)
    ax.imshow(image)
    boxes = labels['boxes'].numpy()
    boxes = np.vsplit(boxes, boxes.shape[0])
    for box in boxes:
        box = np.squeeze(box)
        bottom, left = box[0], box[1]
        width = box[2] - box[0]
        height = box[3] - box[1]
        rect = patches.Rectangle((bottom,left),width,height,linewidth=2,edgecolor='r',facecolor='none')
        # # Add the patch to the Axes
        ax.add_patch(rect)
    plt.axis('off')
    plt.show()
