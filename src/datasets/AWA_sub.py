from torchvision import transforms
from torchvision.datasets import VisionDataset
from PIL import Image
import torch
import numpy as np
import os

class_names = dict()
class_names[0] = 'antelope' 
class_names[1] = 'bat'
class_names[2] = 'beaver'
class_names[3] = 'blue whale' 
class_names[4] = 'bobcat'
class_names[5] = 'buffalo' 
class_names[6] = 'chihuahua' 
class_names[7] = 'chimpanzee'
class_names[8] = 'collie' 
class_names[9] = 'cow'
class_names[10] = 'dalmation' 
class_names[11] = 'deer'
class_names[12] = 'dolphin'
class_names[13] = 'elephant' 
class_names[14] = 'fox'
class_names[15] = 'german shepherd' 
class_names[16] = 'giant panda' 
class_names[17] = 'giraffe'
class_names[18] = 'gorilla' 
class_names[19] = 'grizzly bear'
class_names[20] = 'hamster' 
class_names[21] = 'hippopotamus'
class_names[22] = 'horse'
class_names[23] = 'humpback whale' 
class_names[24] = 'killer whale'
class_names[25] = 'leopard' 
class_names[26] = 'lion' 
class_names[27] = 'mole'
class_names[28] = 'moose' 
class_names[29] = 'mouse'
class_names[30] = 'otter' 
class_names[31] = 'ox'
class_names[32] = 'persian cat'
class_names[33] = 'pig' 
class_names[34] = 'polar bear'
class_names[35] = 'rabbit' 
class_names[36] = 'raccoon' 
class_names[37] = 'rat'
class_names[38] = 'rhinoceros' 
class_names[39] = 'seal'
class_names[40] = 'sheep' 
class_names[41] = 'siamese cat'
class_names[42] = 'skunk'
class_names[43] = 'spider monkey' 
class_names[44] = 'squirrel'
class_names[45] = 'tiger' 
class_names[46] = 'walrus' 
class_names[47] = 'weasel'
class_names[48] = 'wolf' 
class_names[49] = 'zebra'

class_labels = list(class_names.values())

class AWA_sub(VisionDataset):
    default_class_groups = [[i] for i in range(50)]
    name = 'AWA'
    # data was normalised before and turned back into uint8 by transformation in each channel: data -> (data - data_min) / (data_max - data_min)
    # new mean: (0 - data_min) / (data_max - data_min)
    # new std: 1 / (data_max - data_min)
    d0_max = 2.2489083
    d0_min = -2.117904
    d1_max = 2.4285715
    d1_min = -2.0357141
    d2_max = 2.64
    d2_min = -1.8044444
    mean = -np.array([d0_min / (d0_max - d0_min), d1_min / (d1_max - d1_min), d2_min / (d2_max - d2_min)])
    std = 1 / np.array([d0_max - d0_min, d1_max - d1_min, d2_max -d2_min])
    default_transform = transforms.Compose([
        transforms.Normalize(tuple(mean), tuple(std))
    ])
    inverse_transform = transforms.Compose([
        transforms.Normalize((0.,0.,0.), tuple(1/std)),
        transforms.Normalize(tuple(-mean), (1., 1., 1.))
    ])

    @staticmethod
    def to_0_1(data, d0_max=d0_max, d0_min=d0_min, d1_max=d1_max, d1_min=d1_min, d2_max=d2_max, d2_min=d2_min):
        # function to unnormalize and turn into uint8
        data[:,0,:,:] -= d0_min
        data[:,1,:,:] -= d1_min
        data[:,2,:,:] -= d2_min
        data[:,0,:,:] /= (d0_max - d0_min)
        data[:,1,:,:] /= (d1_max - d1_min)
        data[:,2,:,:] /= (d2_max - d2_min)
        return data
    
    def __init__(
        self,
        root="",
        split="train",
        transform=None,
        inv_transform=None,
        target_transform=None, 
        #download=False,
        validation_size=500
    ):
        if transform is None:
            transform=AWA_sub.default_transform
        if inv_transform is None:
            inv_transform=AWA_sub.inverse_transform
        train=(split=="train")
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.train=train
        self.split=split
        self.inverse_transform=inv_transform #MUST HAVE THIS FOR MARK DATASET TO WORK
        self.classes=[i for i in range(50)]

        self.data = np.empty(shape=(3733,3,224,224), dtype=np.float32) 
        self.targets = np.empty(shape=(3733))
        self.data[:2987,:,:,:] = self.to_0_1(np.squeeze(np.load(os.path.join(root, 'AWA_rawdata/AWA_train_input_sub.npy'))))
        self.targets[:2987] = np.squeeze(np.load(os.path.join(root, 'AWA_rawdata/AWA_train_label_sub.npy')))
        self.data[2987:,:,:,:] = self.to_0_1(np.squeeze(np.load(os.path.join(root, 'AWA_rawdata/AWA_val_input_sub.npy'))))
        self.targets[2987:] = np.squeeze(np.load(os.path.join(root, 'AWA_rawdata/AWA_val_label_sub.npy')))
        '''
        self.data = self.to_0_1(np.squeeze(np.load(os.path.join(root, 'AWA_val_input.npy'))))
        self.targets = np.squeeze(np.load(os.path.join(root, 'AWA_val_label.npy')))
        '''

        # Recalculating max and min in each channel
        #print("Dimension 0")
        #print(self.data[:,0,:,:].max())
        #print(self.data[:,0,:,:].min())
        #print("Dimension 1")
        #print(self.data[:,1,:,:].max())
        #print(self.data[:,1,:,:].min())
        #print("Dimension 2")
        #print(self.data[:,2,:,:].max())
        #print(self.data[:,2,:,:].min())
        
        N = len(self.targets)

        if not train:
            if (os.path.isfile("AWA_val_sub_ids") and os.path.isfile("AWA_test_sub_ids")):
                self.val_ids=torch.load("AWA_val_sub_ids")
                self.test_ids=torch.load("AWA_test_sub_ids")
            else:
                torch.manual_seed(42)  # THIS SHOULD NOT BE CHANGED BETWEEN TRAIN TIME AND TEST TIME
                perm = torch.randperm(N)
                self.val_ids = torch.tensor([i for i in perm[:validation_size]])
                self.test_ids = torch.tensor([i for i in perm[validation_size:]])
                #torch.save(self.val_ids, 'AWA_val_ids')
                #torch.save(self.test_ids, 'AWA_test_ids')

            print("Validation ids:")
            print(self.val_ids)
            print("Test ids:")
            print(self.test_ids)
            self.test_targets=torch.tensor(self.targets)[self.test_ids]

    def __getitem__(self, item):
        if self.split=="train":
            id=item
        elif self.split=="val":
            id=self.val_ids[item]
        else:
            id=self.test_ids[item]
        img, target = self.data[id], self.targets[id]
        img = torch.from_numpy(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target
    
    def __len__(self):
        if self.split=="train":
            return len(self.targets)
        elif self.split=="val":
            return len(self.val_ids)
        else:
            return len(self.test_ids)