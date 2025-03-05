from trak import TRAKer
from trak.projectors import CudaProjector, ChunkedCudaProjector
from utils.explainers import Explainer
from trak.projectors import ProjectionType
import os
from glob import glob
from shutil import copytree, rmtree
import torch
from time import time
from utils.data import FeatureDataset
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

class FeatureSimilarityExplainer(Explainer):
    name = "FeatureSimilarityExplainer"
    def __init__(self, model, dataset, dir, features_dir, device, mode='dot'):
        super(FeatureSimilarityExplainer, self).__init__(model, dataset, device)
        os.makedirs(dir, exist_ok=True)
        self.features_dir = os.path.join(features_dir, "samples")
        self.labels_dir = os.path.join(features_dir, "labels")
        self.mode=mode
        #feature_ds = FeatureDataset(self.model, dataset, device, dir)
        #self.labels = torch.tensor(feature_ds.labels, dtype=torch.int, device=self.device)

    def train(self):
        if os.path.isfile(self.features_dir):
            print("Features found.")
            self.features=torch.load(self.features_dir, map_location=self.device)
            print("Labels found.")
            self.labels=torch.load(self.labels_dir, map_location=self.device)

    def explain(self, x, xpl_targets):
        labels_expanded = self.labels.view(1, -1)
        xpl_targets_expanded = xpl_targets.view(-1, 1)
        comparison = labels_expanded == xpl_targets_expanded
        M = torch.where(comparison, torch.tensor(1), torch.tensor(-1))

        x=x.to(self.device)
        f=self.model.features(x).to(self.device)

        dataloader = DataLoader(self.features, batch_size=200, shuffle=False)
        xpl = torch.empty((x.shape[0], 0), device=self.device)
        for features in dataloader:
            if self.mode == 'dot':
                xpl_curr = f @ features.T
            elif self.mode == 'cos':
                f_norm = f / torch.norm(f, dim=1)[:, None]
                features_norm = features / features.norm(dim=1)[:, None]
                xpl_curr = f_norm @ features_norm.T
            elif self.mode == 'l2':
                # using pythagoras: |a-b|^2 = |a|^2 + |b|^2 - 2ab
                features_norm_squared = torch.sum(features ** 2, dim=1, keepdim=True).T
                f_norm_squared = torch.sum(f ** 2, dim=1, keepdim=True)
                dot_product = 2 * torch.mm(features, f.t()).T
                distance_matrix_squared = features_norm_squared + f_norm_squared - dot_product
                distance_matrix_squared = torch.clamp(distance_matrix_squared, min=0.0)
                xpl_curr = torch.sqrt(distance_matrix_squared)
            xpl = torch.cat((xpl, xpl_curr), dim=1)
        return torch.squeeze(xpl * M)

    def self_influences(self):
        if self.mode == 'dot':
            self_inf = torch.pow(self.features, 2).sum(dim=1)
        else: 
            raise Exception("self influences are constant for all other modes")
        print("self influences are computed")
        return self_inf
    
class InputSimilarityExplainer(Explainer):
    name = "InputSimilarityExplainer"
    def __init__(self, model, dataset, dir, features_dir, device, mode='dot'):
        super(InputSimilarityExplainer, self).__init__(model, dataset, device)
        os.makedirs(dir, exist_ok=True)
        self.mode=mode
        self.train_ds=dataset
        self.labels_dir = os.path.join(features_dir, "labels")
        #self.train_ds = Dataset(self.model, dataset, device, dir)
        #self.labels = torch.tensor(self.train_ds.labels, dtype=torch.int, device=self.device)

    def train(self):
        if os.path.isfile(self.labels_dir):
            print("Labels found.")
            self.labels=torch.load(self.labels_dir, map_location=self.device)

    def explain(self, x, xpl_targets):
        labels_expanded = self.labels.view(1, -1)
        xpl_targets_expanded = xpl_targets.view(-1, 1)
        comparison = labels_expanded == xpl_targets_expanded
        M = torch.where(comparison, torch.tensor(1), torch.tensor(-1))

        x=x.to(self.device)
        x=x.flatten(start_dim=1)

        dataloader = DataLoader(self.train_ds, batch_size=200, shuffle=False)
        xpl = torch.empty((x.shape[0], 0), device=self.device)
        for train_x, _ in dataloader:
            train_x=train_x.flatten(start_dim=1).to(self.device)
            if self.mode == 'dot':
                xpl_curr = x @ train_x.T
            elif self.mode == 'cos':
                x_norm = x / torch.norm(x, dim=1)[:, None]
                train_x_norm = train_x / train_x.norm(dim=1)[:, None]
                xpl_curr = x_norm @ train_x_norm.T
            elif self.mode == 'l2':
                # using pythagoras: |a-b|^2 = |a|^2 + |b|^2 - 2ab
                x_norm_squared = torch.sum(train_x ** 2, dim=1, keepdim=True).T
                train_x_norm_squared = torch.sum(x ** 2, dim=1, keepdim=True)
                dot_product = 2 * torch.mm(train_x, x.t()).T
                distance_matrix_squared = train_x_norm_squared + x_norm_squared - dot_product
                distance_matrix_squared = torch.clamp(distance_matrix_squared, min=0.0)
                xpl_curr = torch.sqrt(distance_matrix_squared)
            else:
                raise Exception("mode not implemented")
            xpl = torch.cat((xpl, xpl_curr), dim=1)
        return torch.squeeze(xpl * M)

    def self_influences(self):
        if self.mode == 'dot':
            self_inf = torch.pow(self.features, 2).sum(dim=1)
        else: 
            raise Exception("self influences are constant for all other modes")
        print("self influences are computed")
        return self_inf