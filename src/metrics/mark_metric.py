from utils import Metric
import torch

class MarkImageMetric(Metric):
    name = "MarkImageMetric"

    def __init__(self, train, test, model, device="cuda", filter=True):
        self.marked_cls = train.cls_to_mark
        self.filter=filter
        self.marked_samples = train.mark_samples.to(device)
        self.train = train
        self.test = test 
        self.scores = torch.empty(0, dtype=torch.float, device=device)
        self.device = device

    def __call__(self, xpl, start_index):
        xpl.to(self.device)
        most_influential_ids = xpl.argmax(axis=-1)
        def include_datapoint(i):
            if not self.filter:
                return True
            return (self.test[i][1]==self.marked_cls) and (self.test.dataset[i][1]!=self.marked_cls)
        passed = [1. if most_influential_ids[i] in self.marked_samples else 0.
            for i in range(xpl.shape[0]) if include_datapoint(i)]
        passed = torch.tensor(passed, device=self.device)
        self.scores = torch.cat((self.scores, passed), dim=0)
        

    def get_result(self, dir, file_name):
        self.scores = self.scores.to('cpu').numpy()
        score = self.scores.sum() / self.scores.shape[0]
        resdict = {'metric': self.name, 'all_scores': self.scores, 'avg_score': score,
                   'num_examples': self.scores.shape[0]}
        if dir is not None:
            self.write_result(resdict, dir, file_name)
        return resdict