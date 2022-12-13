import os

import numpy as np

from torch.utils.data import Dataset
from torchvision import transforms

from PIL import ImageFile, Image

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageNet(Dataset):
    def __init__(self, root_dir, split="train"):
        self.image_dir = os.path.join(root_dir, split)
        self.images = []
        self.labels = []

        with open(os.path.join(root_dir, "meta", split + ".txt"), "r") as f:
            for line in f.readlines():
                #readlines()返回列表list,包含所有行
                image, label = line.strip().split()
                #strip()没指定变量的话 移除字符串首尾的空格
                #split()没指定变量的话 以空格为分隔 分割字符串
                self.images.append(image)
                self.labels.append(int(label))

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        #把一些列transforms结合起来，利用transforms.compose？对的
        if split == "train":
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self.normalize
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                self.normalize
            ])
    #getitem/len函数
    def __getitem__(self, index):
        image = Image.open(os.path.join(self.image_dir, self.images[index]))
        image = image.convert('RGB')
        image = self.transform(image)
        return {
            "images": image,
            "labels": self.labels[index],
            "filename": self.images[index],
        }

    def __len__(self):
        return len(self.images)
