from trak import TRAKer
from trak.projectors import CudaProjector, ChunkedCudaProjector
from utils.explainers import Explainer
from trak.projectors import ProjectionType
import os
from glob import glob
from shutil import copytree, rmtree
import torch
from time import time

class TRAK(Explainer):
    name = "TRAK"
    def __init__(self, model, dataset, base_cache_dir, dir, device, proj_dim=100, batch_size=32):
        super(TRAK, self).__init__(model, dataset, device)
        self.dataset=dataset
        self.batch_size=batch_size
        self.number_of_params=0

        self.dir=dir
        os.makedirs(dir, exist_ok=True)
        self.base_cache_dir=base_cache_dir
        self.copied_cache=False
        if os.path.isdir(os.path.join(base_cache_dir, "trak_results")):
            self.copied_cache=True
            copytree(os.path.join(base_cache_dir, "trak_results"), os.path.join(dir, "trak_results"))
        for p in list(self.model.parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            self.number_of_params += nn
        if device=="cuda":
            projector=CudaProjector(grad_dim=self.number_of_params,proj_dim=proj_dim,seed=21,device=device, proj_type=ProjectionType.normal, max_batch_size=32)
        else:
            projector=None
        self.traker = TRAKer(model=model, task='image_classification', train_set_size=len(dataset),
                             projector=projector, proj_dim=proj_dim, projector_seed=42, save_dir=os.path.join(dir,"trak_results"),
                             device=device
                             )
        print("dir is {dir}")
        for l in os.listdir(dir):
            print(l)
        exit()

    def train(self):
        t=time()
        ld=torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size)
        self.traker.load_checkpoint(self.model.state_dict(),model_id=0)
        for (i,(x,y)) in enumerate(iter(ld)):
            batch=x.to(self.device), y.to(self.device)
            self.traker.featurize(
                batch=batch,
                inds = torch.tensor(
                    [
                        i*self.batch_size+j
                                for j in range(min(self.batch_size,len(self.dataset)-i*self.batch_size))
                     ]))
        self.traker.finalize_features()

    def explain(self, x, xpl_targets):
        x=x.to(self.device)
        self.traker.start_scoring_checkpoint(model_id=0,
                                             checkpoint=self.model.state_dict(),
                                             exp_name='test',
                                            num_targets=x.shape[0])
        self.traker.score(batch=(x,xpl_targets), num_samples=x.shape[0])
        ret_xpl = torch.from_numpy(self.traker.finalize_scores(exp_name='test')).T.to(self.device)
        self.clean_cache()
        return ret_xpl
    
    def clean_cache(self):
        # this restores the cache folder to the initial state before any explanations were generated
        for f in glob(os.path.join(self.dir, "trak_results", "scores", "*")):
            pass
            os.remove(f)
        for f in glob(os.path.join(self.dir, "trak_results", "0", "test*")):
            pass
            os.remove(f)
        with open(os.path.join(self.dir, "trak_results", "experiments.json"), "w") as f:
            f.write("{}")


    def self_influences(self):
        if os.path.exists(os.path.join(self.dir, "self_influences")):
            self_inf=torch.load(os.path.join(self.dir, "self_influences"))
        else:
            self_inf=self.compute_self_influences_brute_force()
            torch.save(self_inf, os.path.join(self.dir, "self_influences"))
        self.clean_cache()
        return self_inf

    def __del__(self):
        if self.copied_cache and os.path.isdir(os.path.join(self.dir, "trak_results")):
            rmtree(os.path.join(self.dir, "trak_results"))

    def compute_self_influences_brute_force(self):
        ld=torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size)
        self.traker.start_scoring_checkpoint(model_id=0,
                             checkpoint=self.model.state_dict(),
                             exp_name=f'test',
                             num_targets=len(self.dataset)) 
        for (i,(x,y)) in enumerate(iter(ld)):  
            batch=x.to(self.device), y.to(self.device)
            self.traker.score(batch=batch, num_samples=x.shape[0])
        selfinf = torch.from_numpy(self.traker.finalize_scores(exp_name=f'test')).diag().to(self.device)
        return selfinf