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
    def load_explainers(model, dataset, save_dir, ckpt_dir, learning_rates, dimensions, device):
        explainers=[]
        assert os.path.isdir(ckpt_dir), f"Given checkpoint path f{ckpt_dir} is not a directory."
        ckpt_files=[f for f in os.listdir(ckpt_dir) if f[-5]==".ckpt"]
        for i, ckpt, lr in enumerate(zip(ckpt_files,learning_rates)):
            modelcopy=deepcopy(model)
            checkpoint=torch.load(ckpt)
            modelcopy.load_state_dict(checkpoint["model_state"])
            dir_path=os.path.join(save_dir,f"_{i}")
            os.makedirs(dir_path,exist_ok=True)
            explainers.append((lr,GradDotExplainer(modelcopy,dataset,dir_path,device)))
        return explainers

    def __init__(self,model,dataset, save_dir, ckpt_dir, learning_rates, dimensions, device="cuda" if torch.cuda.is_available() else "cpu"):
        # if dimension=None, no random projection will be done
        super().__init__(model,dataset,device)
        self.explainers=TracInExplainer.load_explainers(model,dataset,save_dir, ckpt_dir, learning_rates,dimensions,device)
        assert len(self.explainers)==len(learning_rates)
    
    def explain(self, x, xpl_targets):
        attr=torch.zeros((x.shape[0], len(self.dataset)))
        for rate,xplainer in self.explainers:
            attr=attr+rate*xplainer.explain(x,xpl_targets)
        return attr
            
