import os.path
from abc import ABC, abstractmethod

from torch.cuda import is_available
from utils.data import FeatureDataset
import torch
import time
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

# this class is to be deleted when graddot is seen to work
class GradientProductExplainer(Explainer):
    name = "GradientProductExplainer"

    def get_param_grad(self, x, index=None):
        x = x.to(self.device)
        out = self.model(x[None, :, :])
        if index is None:
            index = range(self.model.classifier.out_features)
        else:
            index = [index]
        grads = torch.empty(len(index), self.number_of_params)

        for i, ind in enumerate(index):
            assert ind > -1 and int(ind) == ind
            self.model.zero_grad()
            if self.loss is not None:
                out_new = self.loss(out, torch.eye(out.shape[1], device=self.device)[None, ind])
                out_new.backward(retain_graph=True)
            else:
                out[0][ind].backward(retain_graph=True)
            cumul = torch.empty(0, device=self.device)
            for par in self.model.sim_parameters():
                grad = par.grad.flatten()
                cumul = torch.cat((cumul, grad), 0)
            grads[i] = cumul

        return torch.squeeze(grads)

    def __init__(self, model, dataset, device, loss=None):
        super().__init__(model, dataset, device)
        self.number_of_params = 0
        self.loss = loss
        for p in list(self.model.sim_parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            self.number_of_params += nn
        # USE get_param_grad instead of grad_ds = GradientDataset(self.model, dataset)
        self.dataset = dataset

    def explain(self, x, xpl_targets):
        xpl = torch.zeros((x.shape[0], len(self.dataset)),dtype=torch.float)
        xpl = xpl.to(self.device)
        t = time.time()
        for j in range(len(self.dataset)):
            tr_sample, y = self.dataset[j]
            train_grad = self.get_param_grad(tr_sample, y)
            train_grad=train_grad/torch.norm(train_grad)
            train_grad.to(self.device)
            for i in range(x.shape[0]):
                if self.loss is None:
                    test_grad = self.get_param_grad(x[i], preds[i])
                else:
                    test_grad = self.get_param_grad(x[i], targets[i])
                test_grad.to(self.device)
                xpl[i, j] = torch.matmul(train_grad, test_grad)
            if j % 1000 == 0:
                tdiff = time.time() - t
                mins = int(tdiff / 60)
                print(
                    f'{int(j / 1000)}/{int(len(self.dataset) / 1000)}k- 1000 images done in {mins} minutes {tdiff - 60 * mins}')
                t = time.time()
        return xpl


class GradDotExplainer(Explainer):
    name="GradDotExplainer"
    def __init__(self,model,dataset,dir,dimensions, loss=False, device="cuda" if torch.cuda.is_available() else "cpu"):
        # if dimension=None, no random projection will be done
        super().__init__(model,dataset,device)
        self.loss=loss
        self.number_of_params=0
        for p in list(self.model.sim_parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            self.number_of_params += nn
        self.dataset = dataset
        self.norms=torch.ones(len(self.dataset),device=self.device)

        self.dir=dir
        self.dimensions=dimensions

    def train(self):
        t0=time()
        if self.dimensions is not None:
            self.random_matrix=None
        else:
            file_path=os.path.join(self.dir,"random_matrix_tensor")
            if os.path.isfile(file_path):
                self.random_matrix=torch.load(file_path,map_location=self.device)
            else:
                self.random_matrix=self.make_random_matrix()
                torch.save(self.random_matrix,file_path)

        file_path=os.path.join(self.dir,"train_grads_tensor")
        if os.path.isfile(file_path):
            self.train_grads=torch.load(file_path,map_location=self.device)
        else:
            self.train_grads=self.make_train_grads()
            torch.save(self.train_grads,file_path)
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
        out = self.model(x[None, :, :])
        self.model.zero_grad()
        if self.loss:
            output=torch.nn.functional.cross_entropy(out,[index])
        else:
            output=out[0][index]
        output.backward()
        cumul_grads = torch.empty(0, device=self.device)
        for par in self.model.sim_parameters():
            grad = par.grad.flatten()
            cumul_grads = torch.cat((cumul_grads, grad), 0)
        if self.random_matrix is not None:
            cumul_grads=torch.matmul(self.random_matrix,cumul_grads)
        return cumul_grads
    

class GradCosExplainer(GradDotExplainer):
    def get_param_grad(self, x, index):
        grad = super().get_param_grad(x, index)
        return grad/grad.norm