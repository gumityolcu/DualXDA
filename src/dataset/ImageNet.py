from torchvision.datasets import ImageNet as tvImageNet
from torchvision import transforms
import torch
import os

class ImageNet(tvImageNet):
    default_class_groups = [[i] for i in range(1000)]
    class_labels=[i for i in range(1000)]
    name='ImageNet'
    default_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
    inverse_transform = transforms.ToTensor()

    @staticmethod
    def get_index(item, split):
        if split == "train":
            cls = item // 40
            index = item % 40
            return cls * 50 + index
        else:
            cls = item // 10
            index = item % 10
            return cls * 50 + index + 40
    

    def __init__(
            self,
            root= "imagenet/",
            split="train",
            transform=None,
            inv_transform=None,
            validation_size=10000
    ):
        if transform is None:
            transform=ImageNet.default_transform
        else:
            transform=transforms.Compose([transform, ImageNet.default_transform])
        if inv_transform is not None:
            self.inverse_transform = ImageNet.inverse_transform  # MUST HAVE THIS FOR MARK DATASET TO WORK
        super().__init__(root=root, split="val", transform=transform, target_transform=None)
        self.val_split = split

        #for each class, first 40 samples go to train, the last 10 to val

    def __getitem__(self, item):
        if self.val_split=="train":
            assert item < int(super(tvImageNet, self).__len__() * 0.8), f"index {item} out of range for train set"
            train_item = self.get_index(item, "train")
            im, label = super(tvImageNet, self).__getitem__(train_item)
        else:
            assert item < int(super(tvImageNet, self).__len__() * 0.2), f"index {item} out of range for val set"
            val_item = self.get_index(item, "val")
            im, label = super(tvImageNet, self).__getitem__(val_item)
        return im, label


    def __len__(self):
        if self.val_split=="train":
            return int(super(tvImageNet, self).__len__() * 0.8)
        else:
            return int(super(tvImageNet, self).__len__() * 0.2)