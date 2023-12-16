from PIL import Image
from torch.utils.data import Dataset
import torch
import os

class CelebADataset(Dataset):
    def __init__(self, img_dir, attr_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.attrs = []

        with open(attr_file, 'r') as file:
            file.readline()  # Skip the first line
            self.attr_names = file.readline().split()  # Get attribute names
            for line in file:
                line = line.split()
                filename = line[0]
                attributes = line[1:]
                self.attrs.append((filename, attributes))

    def __len__(self):
        return len(self.attrs)

    def __getitem__(self, idx):
        img_name, attributes = self.attrs[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        attributes = torch.tensor([int(a) for a in attributes])
        return image, attributes