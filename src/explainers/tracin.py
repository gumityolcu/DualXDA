import os.path
import torch
import time
from utils.data import ReduceLabelDataset, CorruptLabelDataset, GroupLabelDataset, MarkDataset
from math import sqrt
from tqdm import tqdm
from utils.explainers import Explainer, GradDotExplainer
from copy import deepcopy

class TracInExplainer(Explainer):
    name="TracInExplainer"
    @staticmethod
    def load_explainers(model, dataset, save_dir, ckpt_dir, dimensions, device):
        explainers=[]
        assert os.path.isdir(ckpt_dir), f"Given checkpoint path {ckpt_dir} is not a directory."
        ckpt_files=[os.path.join(ckpt_dir,f) for f in os.listdir(ckpt_dir) if "best" not in f and not os.path.isdir(os.path.join(ckpt_dir,f))]

        for i, ckpt in enumerate(ckpt_files):
            modelcopy=deepcopy(model)
            checkpoint=torch.load(ckpt)
            modelcopy.load_state_dict(checkpoint["model_state"])
            dir_path=os.path.join(save_dir,f"_{i}")
            os.makedirs(dir_path,exist_ok=True)
            explainers.append((checkpoint["optimizer_state"]["param_groups"][0]["lr"],GradDotExplainer(modelcopy,dataset,dir=dir_path,dimensions=dimensions,loss=True,device=device)))
        return explainers

    def __init__(self,model,dataset, dir, ckpt_dir, dimensions, device="cuda" if torch.cuda.is_available() else "cpu"):
        # if dimension=None, no random projection will be done
        super().__init__(model,dataset,device)
        self.dataset=dataset
        self.explainers=TracInExplainer.load_explainers(model,dataset,dir, ckpt_dir,dimensions,device)

    def train(self):
        for _, x in self.explainers:
            x.train() 
    
    def explain(self, x, xpl_targets):
        attr=torch.zeros((x.shape[0], len(self.dataset)),device=self.device)
        for rate,xplainer in self.explainers:
            attr=attr+rate*xplainer.explain(x,xpl_targets)
        return attr/len(self.explainers)
            
