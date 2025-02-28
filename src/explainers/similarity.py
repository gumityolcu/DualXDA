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

class SimilarityExplainer(Explainer):
    name = "SimilarityExplainer"
    def __init__(self, model, dataset, features_dir, dir, device, mode='dot'):
        super(SimilarityExplainer, self).__init__(model, dataset, device)
        self.dataset=dataset
        self.number_of_params=0
        self.dir=dir
        os.makedirs(dir, exist_ok=True)
        self.features_dir = features_dir
        self.mode=mode
        feature_ds = FeatureDataset(self.model, dataset, device, dir)
        self.labels = torch.tensor(feature_ds.labels, dtype=torch.int, device=self.device)

    def train(self):
        if os.path.isfile(self.features_dir):
            print("Features found.")
            self.features=torch.load(self.features_dir, map_location=self.device)

    def explain(self, x, xpl_targets):
        labels_expanded = self.labels.view(-1, 1)
        xpl_targets_expanded = xpl_targets.view(1, -1)
        comparison = labels_expanded == xpl_targets_expanded
        M = torch.where(comparison, torch.tensor(1), torch.tensor(-1))

        x=x.to(self.device)
        f=self.model.features(x)
        if self.mode == 'dot':
            xpl = self.features @ f
            return torch.squeeze(xpl * M)
        elif self.mode == 'cos':
            f_norm = f / torch.norm(f, dim=0)
            features_norm = f / f.norm(dim=1)[:, None]
            xpl = features_norm @ f_norm
            return torch.squeeze(xpl * M)
        elif self.mode == 'l2':
            xpl = torch.norm(self.features - f, dim=1)
            return torch.squeeze(xpl * M)
        else:
            raise Exception("mode not implemented")

    def self_influences(self):
        if self.mode == 'dot':
            self_inf = torch.power(self.features, 2).sum(dim=1)
        else: 
            raise Exception("self influences are constant for all other modes")
        print("self influences are computed")
        return self_inf