import torch
from utils import Metric
from torch.nn import CosineSimilarity
from torchmetrics.regression import SpearmanCorrCoef

class SwitchMetric(Metric):
    name = "SwitchMetric"

    def __init__(self, device="cuda", **kwargs):
        self.scores = torch.empty(0, dtype=torch.float, device=device)
        self.device = device

    def __call__(self, xpl, xpl_switched, start_index):
        xpl.to(self.device)
        xpl_switched.to(self.device)
        #if xpl.nelement() == 0: #to exit if xpl is empty
        #    return
        #print(xpl.shape)
        #print(xpl_switched.shape)
        
        #self.scores = torch.norm(xpl-xpl_switched, p=1, dim=1)

        # Cosine implementation
        #cos = torch.nn.CosineSimilarity(dim=1, eps=1e-10) 
        #self.scores = cos(xpl, xpl_switched)

        # Spearman rank correlation implementation
        spearman = SpearmanCorrCoef(num_outputs=xpl.shape[0])
        self.scores = spearman(xpl, xpl_switched)

    def get_result(self, dir=None, file_name=None):
        self.scores = self.scores.to('cpu').detach().numpy()
        score = self.scores.sum() / self.scores.shape[0]
        resdict = {'metric': self.name, 'all_scores': self.scores, 'avg_score': score,
                   'num_examples': self.scores.shape[0]}
        if dir is not None:
            self.write_result(resdict, dir, file_name)
        return resdict