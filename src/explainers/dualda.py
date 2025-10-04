import torch
import os
import subprocess
import time
from copy import deepcopy
from utils.csv_io import read_matrix, write_data
from utils.explainers import FeatureKernelExplainer
from struct import pack
from sklearn.svm import LinearSVC ##modified LIBLINEAR MCSVM_CS_Solver which returns the dual variables
from tqdm import tqdm


class DualDA(FeatureKernelExplainer):
    name = "DualDAExplainer"

    def get_name(self):
        return f"{self.name}-{str(self.C)}"
    
    def __init__(self, model, dataset, device, dir, features_dir, use_preds=False, C=1.0, max_iter=1000000, normalize=False):
        super().__init__(model, dataset, device, features_dir, normalize=normalize)
        self.C=C
        if dir[-1]=="\\":
            dir=dir[:-1]
        self.dir=dir
        self.features_dir=features_dir
        self.max_iter=max_iter
        os.makedirs(self.dir, exist_ok=True)
        os.makedirs(self.features_dir, exist_ok=True)

    def read_variables(self):
        self.learned_weight = torch.load(os.path.join(self.dir,"weights"), map_location=self.device).to(torch.float)
        self.coefficients=torch.load(os.path.join(self.dir,"coefficients"), map_location=self.device).to(torch.float)
        self.train_time=torch.load(os.path.join(self.dir,"train_time"), map_location=self.device).to(torch.float)
        self.cache_time=torch.load(os.path.join(self.features_dir,"cache_time")).to(torch.float)

    def train(self):
        tstart = time.time()
        
        if not os.path.isfile(os.path.join(self.features_dir, "samples")):
            torch.save(self.normalized_samples,os.path.join(self.features_dir, "samples"))
        if not os.path.isfile(os.path.join(self.features_dir, "labels")):
            torch.save(self.labels,os.path.join(self.features_dir, "labels"))
        
        if os.path.isfile(os.path.join(self.dir,'weights')) and os.path.isfile(os.path.join(self.dir,'coefficients')):
            self.read_variables()
        else:
            model = LinearSVC(multi_class="crammer_singer", max_iter=self.max_iter, C=self.C)
            model.fit(self.normalized_samples.cpu(),self.labels.cpu())
            accuracy = model.score(self.normalized_samples.cpu(), self.labels.cpu())
            print(f"SVC Accuracy: {accuracy:.2f}")

            self.coefficients=torch.tensor(model.alpha_.T,dtype=torch.float,device=self.device)
            self.learned_weight=torch.tensor(model.coef_,dtype=torch.float, device=self.device)
            self.train_time = torch.tensor(time.time() - tstart)

            torch.save(self.train_time,os.path.join(self.dir,'train_time'))
            torch.save(self.learned_weight,os.path.join(self.dir,'weights'))
            torch.save(self.coefficients,os.path.join(self.dir,'coefficients'))
            print(f"Training took {self.train_time} seconds")
        return self.train_time

    def self_influences(self, only_coefs=False):
        self_coefs=super().self_influences()
        if only_coefs:
            return self_coefs
        else:
            return self.normalized_samples.norm(dim=-1)*self_coefs
