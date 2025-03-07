from utils import Metric
import torch
from torcheval.metrics.functional import binary_auprc


class MarkImageMetric(Metric):
    name = "MarkImageMetric"

    def __init__(self, train, test, model, device="cuda", filter=True, topk=1):
        self.marked_cls = torch.tensor(train.cls_to_mark,device=device)
        self.filter=filter
        self.train = train
        self.test = test 
        self.topk = topk
        self.scores = torch.empty(0, dtype=torch.float, device=device)
        self.device = device
        self.binary_shortcut_indices: torch.Tensor = torch.tensor(
            [
                1 if i in self.train.mark_samples else 0
                for i in range(len(self.train))
            ],
            device=self.device,
        )


    def __call__(self, xpl, start_index):
        xpl=xpl.to(self.device)
        #most_influential_ids = torch.argsort(-xpl, dim=-1)[:,:self.topk]
        cnt=0
        def include_datapoint(i):
            if not self.filter:
                return True
            return (torch.tensor(self.test[i][1],device=self.device)==self.marked_cls) and (torch.tensor(self.test.dataset[i][1],device=self.device)!=self.marked_cls) #first condition means "prediction is the marked class"
        filter_indices=[include_datapoint(i) for i in range(xpl.shape[0])]   
        xpl=xpl[filter_indices]
        expanded_binary_indices=self.binary_shortcut_indices.expand((xpl.shape[0], len(self.train)))     
        self.scores = torch.cat((self.scores,  binary_auprc(xpl, expanded_binary_indices, num_tasks=xpl.shape[0])), dim=0)



    def get_result(self, dir, file_name):
        self.scores = self.scores.to('cpu').numpy()
        score = self.scores.sum() / self.scores.shape[0]
        resdict = {'metric': self.name, 'all_scores': self.scores, 'avg_score': score,
                   'num_examples': self.scores.shape[0]}
        if dir is not None:
            self.write_result(resdict, dir, file_name)
        return resdict