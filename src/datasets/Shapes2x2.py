from torchvision import transforms
from torchvision.datasets import VisionDataset
import torch
import numpy as np
from PIL import Image

class Shapes2x2(VisionDataset):
    default_class_groups = [0, 1]
    name = 'Shapes2x2'
    mean=torch.load("Shapes2x2/train_mean.pt")
    std=torch.load("Shapes2x2/train_std.pt")
    default_transform = transforms.Compose([
        transforms.ToTensor(),
    #    transforms.Normalize(mean, std)
    ])
    inverse_transform = transforms.Compose([
    #    transforms.Normalize((0.,0.,0.), 1/std),
    #    transforms.Normalize(-mean, (1., 1., 1.))
    ])
    class_labels = [0, 1]

    @staticmethod
    def define_label(str):
        if str[0:2]=='bc' or str[9:11]=='bc':
            label=1
        else: 
            label=0
        return label
    
    @staticmethod
    def to_uint8(data):
        data= np.clip(data, 0 , 1)
        data = data * 255.
        data = data.astype(np.uint8)
        return data
    
    def __init__(
        self,
        root="Shapes2x2",
        split="train",
        transform=None,
        inv_transform=None,
        target_transform=None,
    ):
        if transform is None:
            transform=Shapes2x2.default_transform
        else:
            transform=transforms.Compose([Shapes2x2.default_transform, transform])
        if inv_transform is None:
            inv_transform=Shapes2x2.inverse_transform
        train=(split=="train")
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.train=train
        self.split=split
        self.inverse_transform=inv_transform #MUST HAVE THIS FOR MARK DATASET TO WORK
        self.classes=[0,1]

        self.train_data_path = "Shapes2x2/train_data.npy"
        self.test_data_path = "Shapes2x2/test_data.npy"
        self.train_labels_path = "Shapes2x2/train_labels.npy"
        self.test_labels_path = "Shapes2x2/test_labels.npy"

        self.train_indices = np.arange(20000)
        self.val_indices = np.arange(2000)

    def load_data(self, idx, split):
        if split == "train":
            data = np.load(self.train_data_path, mmap_mode='r')[idx]
            target = np.load(self.train_labels_path, mmap_mode='r')[idx]
        else:
            data = np.load(self.test_data_path, mmap_mode='r')[idx]
            target = np.load(self.test_labels_path, mmap_mode='r')[idx]
        data = np.copy(data)
        data = self.to_uint8(data)
        return data, target
    
    def __getitem__(self, id):
        data, label = self.load_data(id, self.split)
        img = Image.fromarray(data.transpose(1,2,0))
        label = self.define_label(label)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label
    
    def __len__(self):
        if self.split=="train":
            return len(self.train_labels)
        else:
            return len(self.test_labels)