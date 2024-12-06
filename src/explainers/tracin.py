import os.path
import torch
import time
from utils.data import ReduceLabelDataset, CorruptLabelDataset, GroupLabelDataset, MarkDataset
from math import sqrt
from tqdm import tqdm
from utils.explainers import Explainer, GradDotExplainer
from copy import deepcopy
from utils.models import clear_resnet_from_checkpoints

class TracInExplainer(Explainer):
    name="TracInExplainer"


    def __init__(self,model,dataset, dir, ckpt_dir, dimensions, device="cuda" if torch.cuda.is_available() else "cpu"):
        # if dimension=None, no random projection will be done
        super().__init__(model,dataset,device)
        self.dataset=dataset    
        self.dimensions=dimensions
        self.dir=dir
        self.device=device
        self.explainers_info=[]
        assert os.path.isdir(ckpt_dir), f"Given checkpoint path {ckpt_dir} is not a directory."
        ckpt_files=[os.path.join(ckpt_dir,f) for f in os.listdir(ckpt_dir) if not os.path.isdir(os.path.join(ckpt_dir,f))]
        best_epoch_seen=False
        for i, ckpt in enumerate(ckpt_files):
            epoch=ckpt.split("_")[-1]
            checkpoint=torch.load(ckpt, map_location=device)
            checkpoint = clear_resnet_from_checkpoints(checkpoint) #this MIGHT be lefover from using older checkpoints
            grad_path=os.path.join(dir,epoch)
            print(f"epoch being processed: {epoch}")
            best_epoch_seen=best_epoch_seen or epoch=="best"
            os.makedirs(grad_path,exist_ok=True)
            self.explainers_info.append(
                (
                    checkpoint["optimizer_state"]["param_groups"][0]["lr"], grad_path
                )
            )
        assert best_epoch_seen, "No checkpoint with the _best suffix found in checkpoint directory."
    # Old version: Created memory problems for AWA and ResNet-50
    '''
    def train(self):
        for i, (_, x) in enumerate(self.explainers):
            print(f"Handling checkpoint number {i}")
            x.train() 
            torch.cuda.empty_cache()
    
    def explain(self, x, xpl_targets):
        attr=torch.zeros((x.shape[0], len(self.dataset)),device=self.device)
        for rate,xplainer in self.explainers:
            attr=attr+rate*xplainer.explain(x,xpl_targets)
        return attr/len(self.explainers)
    '''

    def train(self):
        time=0.
        for (_, path) in self.explainers_info:
            graddot=GradDotExplainer(
                model=self.model,
                dataset=self.dataset,
                mat_dir=self.dir,
                grad_dir=path,
                dimensions=self.dimensions,
                loss=True,
                device=self.device)
            time=time+graddot.train() 
            torch.cuda.empty_cache()
        self.train_time=time
        torch.save(self.train_time, os.path.join(self.dir, "train_time"))
        return self.train_time

    def explain(self, x, xpl_targets):
        attr=torch.zeros((x.shape[0], len(self.dataset)),device=self.device)
        nr_explainers = len(self.explainers)
        for i, (rate, path) in enumerate(self.explainers):
            graddot=GradDotExplainer(
                model=self.model,
                dataset=self.dataset,
                mat_dir=self.dir,
                grad_dir=path,
                dimensions=self.dimensions,
                loss=True,
                device=self.device)
            attr=attr+rate*graddot.explain(x,xpl_targets) 
        return attr/nr_explainers

    def self_influences(self):
        if os.path.exists(os.path.join(self.dir, "self_influences")):
            self_inf=torch.load(os.path.join(self.dir, "self_influences"), map_location=self.device)
        else:
            self_inf=torch.zeros((len(self.dataset),), device=self.device)
            nr_explainers = len(self.explainers)
            for i, (rate, path) in enumerate(self.explainers):
                graddot=GradDotExplainer(
                model=self.model,
                dataset=self.dataset,
                mat_dir=self.dir,
                grad_dir=path,
                dimensions=self.dimensions,
                loss=True,
                device=self.device)
                self_inf=self_inf+rate*graddot.self_influences()
            self_inf = self_inf/nr_explainers
            torch.save(self_inf, os.path.join(self.dir, "self_influences"))
        return self_inf