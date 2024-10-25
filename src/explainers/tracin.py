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


    def load_explainers(self, model, dataset, ds_name, ds_type, dir, ckpt_dir, dimensions, device):
        explainers=[]
        assert os.path.isdir(ckpt_dir), f"Given checkpoint path {ckpt_dir} is not a directory."
        ckpt_files=[os.path.join(ckpt_dir,f) for f in os.listdir(ckpt_dir) if "best" not in f and not os.path.isdir(os.path.join(ckpt_dir,f))]
        epochs=sorted([int(f.split("_")[-1]) for f in ckpt_files])
        last_epoch=epochs[-1]
        for i, ckpt in enumerate(ckpt_files):
            modelcopy=deepcopy(model)
            epoch=int(ckpt.split("_")[-1])
            checkpoint=torch.load(ckpt, map_location=device)
            checkpoint = clear_resnet_from_checkpoints(checkpoint)
            modelcopy.load_state_dict(checkpoint["model_state"])
            grad_path=os.path.join(dir,f"_{epoch}") if epoch!=last_epoch else dir
            os.makedirs(grad_path,exist_ok=True)
            mat_path=dir
            explainers.append(
                (
                    checkpoint["optimizer_state"]["param_groups"][0]["lr"],
                    GradDotExplainer(
                        model=modelcopy,
                        dataset=dataset, 
                        mat_dir=self.dir,
                        dimensions=dimensions,
                        ds_name=ds_name,
                        ds_type=ds_type,
                        loss=True,
                        device=device,
                        )
                )
            )
        return explainers

    def __init__(self,model,dataset, dir, ckpt_dir, dimensions, ds_name, ds_type, device="cuda" if torch.cuda.is_available() else "cpu"):
        # if dimension=None, no random projection will be done
        super().__init__(model,dataset,device)
        self.dataset=dataset
        self.explainers=self.load_explainers(model,dataset, ds_name, ds_type, dir, ckpt_dir, dimensions, device)

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

    # New version: For every checkpoint creates an explainer in load_explainers, then handles and afterwards deletes them one by one 
    def train(self):
        pass       
    
    def explain(self, x, xpl_targets):
        attr=torch.zeros((x.shape[0], len(self.dataset)),device=self.device)
        nr_explainers = len(self.explainers)
        for i, (rate, xplainer) in enumerate(self.explainers):
            print(f"Handling checkpoint number {i}")
            xplainer.train()
            attr=attr+rate*xplainer.explain(x,xpl_targets) 
        return attr/nr_explainers