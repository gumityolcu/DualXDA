from utils import Metric
import torch

class MarkImageMetric(Metric):
    name = "MarkImageMetric"

    def __init__(self, train, test, model, device="cuda", filter=True):
        self.marked_cls = torch.tensor(train.cls_to_mark,device=device)
        self.filter=filter
        self.marked_samples = train.mark_samples
        self.train = train
        self.test = test 
        self.scores = torch.empty(0, dtype=torch.float, device=device)
        self.device = device

    def __call__(self, xpl, start_index):
        xpl=xpl.to(self.device)
        most_influential_ids = xpl.argmax(axis=-1)
        self.marked_samples=torch.tensor(self.marked_samples).to(device)
        def include_datapoint(i):
            if not self.filter:
                return True
            return (torch.tensor(self.test[i][1],device=self.device)==self.marked_cls) and (torch.tensor(self.test.dataset[i][1],device=self.device)!=self.marked_cls)
        passed = []
        for i in range(xpl.shape[0]):
            if include_datapoint(i):
                if most_influential_ids[i] in self.marked_samples:
                    passed.append(1.)
                else:
                    passed.append(0.)
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