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
    def __init__(self, model, dataset, device, dir=None,normalize=False):
        super().__init__(model, dataset, device)
        # self.sanity_check = sanity_check
        os.makedirs(dir, exist_ok=True)
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
        with torch.no_grad():
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


    def self_influences(self, only_coefs=False):
        self_coefs = self.coefficients[torch.arange(self.coefficients.shape[0]), self.labels]
        if only_coefs:
            return self_coefs
        else:
            return self.normalized_samples.norm(dim=-1)*self_coefs
        

    def save_coefs(self, dir):
        torch.save(self.coefficients, os.path.join(dir, f"{self.name}_coefs"))

class GradDotExplainer(Explainer):
    name="GradDotExplainer"     
    gradcos_name="GradCosExplainer"
    def __init__(self,
                 model,
                 dataset,
                 mat_dir,
                 grad_dir,
                 dimensions,
                 loss=False,
                 device="cuda" if torch.cuda.is_available() else "cpu"):
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

        self.mat_dir=mat_dir
        self.grad_dir=grad_dir
        self.dimensions=dimensions
        self.random_matrix=None
        self.train_grads=None
        os.makedirs(self.mat_dir, exist_ok=True)
        os.makedirs(self.grad_dir, exist_ok=True)

    def train(self):
        t0=time()
        rand_mat_path=os.path.join(self.mat_dir, "random_matrix")
        if os.path.isfile(rand_mat_path):
            print("Random matrix found.")
            self.random_matrix=torch.load(rand_mat_path, map_location=self.device)
            assert self.dimensions==self.random_matrix.shape[0], f"Cached random matrix has dimension {self.random_matrix.shape[0]} but expected {self.dimensions}"
            print('Random matrix dimensions:', self.random_matrix.shape)
        else:
            self.random_matrix=self.make_random_matrix()
            torch.save(self.random_matrix, rand_mat_path)
        
        grads_path=os.path.join(self.grad_dir, "grads")
        norms_path=os.path.join(self.grad_dir, "self_influences")

        if os.path.isfile(grads_path):
            print("Train grads found.")
            self.train_grads=torch.load(grads_path, map_location=self.device)
            print('Train grads dimensions:', self.train_grads.shape)
            self.norms=torch.load(norms_path, map_location=self.device)
            print('Norm dimensions:', self.norms.shape)
            self.train_time=torch.load(os.path.join(self.grad_dir, "train_time"), map_location=self.device)
        else:
            self.train_grads, self.norms = self.make_train_grads()
            torch.save(self.train_grads, grads_path)
            print(f"saved grads at {grads_path}")
            torch.save(self.norms, norms_path)
            print(f"saved norms at {norms_path}")

            self.train_time=time()-t0
            torch.save(self.train_time, os.path.join(self.grad_dir, "train_time"))
        return self.train_time

    def make_random_matrix(self):
        unitvar = torch.randn((self.dimensions,self.number_of_params),device=self.device)
        return unitvar/sqrt(self.dimensions)

    def make_train_grads(self):
        grad_norms=torch.empty(len(self.dataset),device=self.device)
        grad_dim=self.number_of_params if self.dimensions is None else self.dimensions
        train_grads=torch.empty(len(self.dataset),grad_dim,device=self.device)
        for i,(x,y) in tqdm(enumerate(self.dataset)):
            train_grads[i], grad_norms[i] = self.get_param_grad(x,y,norm=True)
        return train_grads, grad_norms

    def explain(self, x, xpl_targets, normalize_train=False, normalize_test=True):
        xpl=torch.empty(x.shape[0],len(self.dataset),device=self.device)
        for i in tqdm(range(x.shape[0])):
            test_grad=self.get_param_grad(x[i],xpl_targets[i], normalize=normalize_test) # normalize the test gradient to prevent overflow by default. doesn't change the tda ranking. but normalize_test=False for calls from within the TracInExplainer
            xpl[i]=torch.matmul(self.train_grads,test_grad)
        if normalize_train:
            xpl=xpl/self.norms
        return xpl

    def self_influences(self):
        return self.norms
    
    def get_param_grad(self, x, index, norm = False, normalize = False):
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
        grad_norm=cumul_grads.norm()
        if normalize:
            cumul_grads=cumul_grads/grad_norm
        if norm:
            return cumul_grads, grad_norm
        return cumul_grads
    
class GradCosExplainer(GradDotExplainer):
    name="GradCosExplainer"
    def get_param_grad(self, x, index, norm):
        return super().get_param_grad(x, index, norm, normalize=True)