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
    @staticmethod
    def load_explainers(model, dataset, ds_name, ds_type, save_dir, ckpt_dir, dimensions, device, random_matrix):
        explainers=[]
        assert os.path.isdir(ckpt_dir), f"Given checkpoint path {ckpt_dir} is not a directory."
        ckpt_files=[os.path.join(ckpt_dir,f) for f in os.listdir(ckpt_dir) if "best" not in f and not os.path.isdir(os.path.join(ckpt_dir,f))]

        for i, ckpt in enumerate(ckpt_files):
            modelcopy=deepcopy(model)
            checkpoint=torch.load(ckpt, map_location=device)
            checkpoint = clear_resnet_from_checkpoints(checkpoint)
            modelcopy.load_state_dict(checkpoint["model_state"])
            dir_path=os.path.join(save_dir,f"_{i}")
            os.makedirs(dir_path,exist_ok=True)
            explainers.append((checkpoint["optimizer_state"]["param_groups"][0]["lr"],GradDotExplainer(modelcopy, dataset, dir_path, dimensions, ds_name, ds_type, cp_nr=i, loss=True, device=device, random_matrix=random_matrix)))
        return explainers

    def __init__(self,model,dataset, dir, ckpt_dir, dimensions, ds_name, ds_type, device="cuda" if torch.cuda.is_available() else "cpu"):
        # if dimension=None, no random projection will be done
        super().__init__(model,dataset,device)
        self.dataset=dataset

        file_path_random_matrix = f'C:/Users/weckbecker/DualView-wip/src/explainers/random_matrix_dim128/random_matrix_{ds_name}_{ds_type}' if not torch.cuda.is_available() else f'/mnt/dataset/dualview_random_matrix_dim128/random_matrix_{ds_name}_{ds_type}'
        save_path_random_matrix = f'C:/Users/weckbecker/DualView-wip/src/explainers/random_matrix_dim128/random_matrix_{ds_name}_{ds_type}' if not torch.cuda.is_available() else f'/mnt/outputs/random_matrix_{ds_name}_{ds_type}'
        if os.path.isfile(file_path_random_matrix):
            print("Random matrix found.")
            self.random_matrix=torch.load(file_path_random_matrix, map_location=self.device)
            print('Random matrix dimensions:', self.random_matrix.shape)
        else:
            self.random_matrix=self.make_random_matrix()
            torch.save(self.random_matrix, save_path_random_matrix)

        self.explainers=TracInExplainer.load_explainers(model,dataset, ds_name, ds_type, dir, ckpt_dir, dimensions, device, self.random_matrix)

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