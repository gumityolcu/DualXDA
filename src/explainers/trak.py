from trak import TRAKer
from trak.projectors import CudaProjector, ChunkedCudaProjector
from utils.explainers import Explainer
from trak.projectors import ProjectionType
import os
import torch
from time import time

class TRAK(Explainer):
    name = "TRAK"
    def __init__(self, model, dataset, dir, device, proj_dim=100, batch_size=32):
        super(TRAK, self).__init__(model, dataset, device)
        self.dataset=dataset
        self.batch_size=batch_size
        self.number_of_params=0
        self.dir=dir
        for p in list(self.model.parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            self.number_of_params += nn
        projector_dict = {"cuda": CudaProjector(grad_dim=self.number_of_params,proj_dim=proj_dim,seed=21,device=device, proj_type=ProjectionType.normal, max_batch_size=32), "cpu": None}
        self.traker = TRAKer(model=model, task='image_classification', train_set_size=len(dataset),
                             projector=projector_dict[device], proj_dim=proj_dim, projector_seed=42, save_dir=os.path.join(dir,"trak_results"),
                             )

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
        return torch.from_numpy(self.traker.finalize_scores(exp_name='test')).T.to(self.device)

    def self_influences(self):
        if os.path.exists(os.path.join(self.dir, "self_influences")):
            self_inf=torch.load(os.path.join(self.dir, "self_influences"))
        else:
            self_inf=torch.zeros((len(self.dataset),), device=self.device)
            self.compute_self_influences_brute_force()
            torch.save(self_inf, os.path.join(self.dir, "self_influences"))
        return self_inf
    
    def compute_self_influences_brute_force(self):
        pass # TODO : implement self-influences and caching
        # ld=torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size)
        
        # for (i,(x,y)) in enumerate(iter(ld)):
        #     self.traker.start_scoring_checkpoint(model_id=0,
        #                                      checkpoint=self.model.state_dict(),
        #                                      exp_name=f'self_influences_{i}',
        #                                     num_targets=x.shape[0])
        #     self.traker.score(batch=(x,y), num_samples=x.shape[0])
        #     xpl = torch.from_numpy(self.traker.finalize_scores(exp_name=f'self_influences_{i}')).T.to(self.device)
        #     batch=x.to(self.device), y.to(self.device)
        #     self.traker.score(batch=batch, num_samples=min(self.batch_size,len(self.dataset)-i*self.batch_size))
        #     self_inf[i*self.batch_size:i*self.batch_size+min(self.batch_size,len(self.dataset)-i*self.batch_size)]=torch.from_numpy(self.traker.finalize_scores(exp_name='test')).T.to(self.device).diag()
        # return self_inf