import os.path
from abc import ABC, abstractmethod

from torch.cuda import is_available
from utils.data import FeatureDataset
import torch
from time import time
from utils.data import ReduceLabelDataset, CorruptLabelDataset, GroupLabelDataset, MarkDataset
from math import sqrt
from tqdm import tqdm

class Explainer(ABC):
    def __init__(self, model, dataset, device):
        self.model = model
        self.device = device
        self.images = dataset
        self.samples = []
        self.labels = []
        dev = torch.device(device)
        self.model.to(dev)

    @abstractmethod
    def explain(self, x, xpl_targets):
        pass

    def train(self):
        pass

    def save_coefs(self, dir):
        pass


class FeatureKernelExplainer(Explainer):
    def __init__(self, model, dataset, device, dir=None,normalize=True):
        super().__init__(model, dataset, device)
        # self.sanity_check = sanity_check
        if dir is not None:
            if not os.path.isdir(dir):
                dir = None
        feature_ds = FeatureDataset(self.model, dataset, device, dir)
        self.coefficients = None  # the coefficients for each training datapoint x class
        self.learned_weights = None
        self.normalize=normalize
        self.samples = feature_ds.samples.to(self.device)
        self.mean = self.samples.sum(0) / self.samples.shape[0]
        #self.mean = torch.zeros_like(self.mean)
        self.stdvar = torch.sqrt(torch.sum((self.samples - self.mean) ** 2, dim=0) / self.samples.shape[0])
        #self.stdvar=torch.ones_like(self.stdvar)
        self.normalized_samples=self.normalize_features(self.samples) if normalize else self.samples
        self.labels = torch.tensor(feature_ds.labels, dtype=torch.int, device=self.device)

    def normalize_features(self, features):
        return (features - self.mean) / self.stdvar

    def explain(self, x, xpl_targets):
        assert self.coefficients is not None
        x = x.to(self.device)
        f = self.model.features(x)
        if self.normalize:
            f = self.normalize_features(f)
        crosscorr = torch.matmul(f, self.normalized_samples.T)
        crosscorr = crosscorr[:, :, None]
        xpl = self.coefficients * crosscorr
        indices = xpl_targets[:, None, None].expand(-1, self.samples.shape[0], 1)
        xpl = torch.gather(xpl, dim=-1, index=indices)
        return torch.squeeze(xpl)

    def save_coefs(self, dir):
        torch.save(self.coefficients, os.path.join(dir, f"{self.name}_coefs"))

class GradDotExplainer(Explainer):
    name="GradDotExplainer"
    def __init__(self,model,dataset,dir,dimensions, ds_name, ds_type, loss=False, device="cuda" if torch.cuda.is_available() else "cpu"):
        # if dimension=None, no random projection will be done
        super().__init__(model,dataset,device)
        self.loss=loss
        self.number_of_params=0
        for p in list(self.model.parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            self.number_of_params += nn
        self.dataset = dataset
        self.norms=torch.ones(len(self.dataset),device=self.device)

        self.dir=dir
        self.dimensions=dimensions
        self.random_matrix=None
        self.train_grads=None
        self.ds_name = ds_name
        self.ds_type = ds_type

    def train(self):
        t0=time()
        file_path_random_matrix = f'C:/Users/weckbecker/DualView-wip/src/explainers/random_matrix_dim128/random_matrix_{self.ds_name}_{self.ds_type}' if not torch.cuda.is_available() else f'/mnt/outputs/random_matrix_{self.ds_name}_{self.ds_type}'
        if self.dimensions:
            if os.path.isfile(file_path_random_matrix):
                print("Random matrix found.")
                self.random_matrix=torch.load(file_path_random_matrix, map_location=self.device)
            else:
                self.random_matrix=self.make_random_matrix()
                torch.save(self.random_matrix, file_path_random_matrix)

        file_path_train_grads = f'C:/Users/weckbecker/DualView-wip/src/explainers/random_matrix_dim128/train_grads_{self.ds_name}_{self.ds_type}' if not torch.cuda.is_available() else f'/mnt/outputs/train_grads_{self.ds_name}_{self.ds_type}'
        if os.path.isfile(file_path_train_grads):
            print("Train grads found.")
            self.train_grads=torch.load(file_path_train_grads,map_location=self.device)
        else:
            self.train_grads=self.make_train_grads()
            torch.save(self.train_grads, file_path_train_grads)
        return time()-t0

    def make_random_matrix(self):
        unitvar = torch.randn((self.dimensions,self.number_of_params),device=self.device)
        return unitvar/sqrt(self.dimensions)

    def make_train_grads(self):
        grad_dim=self.number_of_params if self.dimensions is None else self.dimensions
        train_grads=torch.empty(len(self.dataset),grad_dim,device=self.device)
        for i,(x,y) in tqdm(enumerate(self.dataset)):
            train_grads[i]=self.get_param_grad(x,y)
        return train_grads

    def explain(self, x, xpl_targets):
        xpl=torch.empty(x.shape[0],len(self.dataset),device=self.device)
        for i in tqdm(range(x.shape[0])):
            test_grad=self.get_param_grad(x[i],xpl_targets[i])
            xpl[i]=torch.matmul(self.train_grads,test_grad)
        return xpl

    def get_param_grad(self, x, index):
        x = x.to(self.device)
        self.model.zero_grad()
        out = self.model(x[None, :, :])
        if self.loss:
            output=torch.nn.functional.cross_entropy(out,torch.tensor([index],device=self.device))
        else:
            output=out[0][index]
        output.backward()
        cumul_grads = torch.empty(0, device=self.device)
        for par in self.model.parameters():
            grad = par.grad.flatten()
            cumul_grads = torch.cat((cumul_grads, grad), 0)
        if self.random_matrix is not None:
            cumul_grads=torch.matmul(self.random_matrix,cumul_grads)
        return cumul_grads
    

class GradCosExplainer(GradDotExplainer):
    name="GradCosExplainer"
    def get_param_grad(self, x, index):
        grad = super().get_param_grad(x, index)
        return grad/grad.norm()