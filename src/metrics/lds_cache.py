import torch
from utils.data import RestrictedDataset
import numpy as np
from metrics import RetrainMetric
import os

class LinearDatamodelingScoreCacher(RetrainMetric):
    name = "LinearDatamodelingScoreCacher"

    def __init__(self, dataset_name, train, test, model_name, epochs, loss, lr, momentum, optimizer, scheduler,
                 weight_decay, augmentation, sample_nr, cache_dir, batch_size, num_classes=10, alpha=0.5, device="cuda"):
        super().__init__(dataset_name, train, test, model_name,
                         epochs, loss, lr, momentum, optimizer, scheduler,
                         weight_decay, augmentation, num_classes, batch_size, device)
        np.random.seed(sample_nr)
        torch.manual_seed(sample_nr)
        self.train = train
        self.alpha = alpha
        self.sample_nr = sample_nr
        self.sample_indices = torch.tensor(np.random.choice(len(self.train), size= int(alpha * len(self.train)), replace=False),device=device).cpu()
        self.save_path = os.path.join(cache_dir, f'lds{alpha}', f'lds{alpha}_{int(sample_nr):02d}')
        os.makedirs(os.path.join(cache_dir, f'lds{alpha}'), exist_ok=True)
        self.cache()
    
    def cache(self):
        ds = RestrictedDataset(self.train, self.sample_indices)
        retrained_model = self.retrain(ds)

        save_dict = {
            'sample_indices': self.sample_indices.cpu(),
            'model_state': retrained_model.state_dict(),
            'dataset_name': self.dataset_name,
            'model_name': self.model_name,
            'epochs': self.epochs,
            'loss': self.loss,
            'lr': self.lr,
            'momentum': self.momentum,
            'optimizer': self.optimizer,
            'scheduler': self.scheduler,
            'weight_decay': self.weight_decay,
            'augmentation': self.augmentation,
            'sample_nr': self.sample_nr,
            'alpha': self.alpha,
        }

        torch.save(save_dict, self.save_path)

    def __call__(self, *args, **kwargs):
        pass

    def get_result(self, *args, **kwargs):
        pass