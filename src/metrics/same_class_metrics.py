import torch
from utils.data import GroupLabelDataset
from utils import Metric


class SameClassMetric(Metric):
    name = "SameClassMetric"

    def __init__(self, train, test, device="cuda"):
        if train.name != 'AWA':
            if isinstance(train.targets,list):
                train.targets=torch.tensor(train.targets,device=device)
        self.train = train
        self.test = test
        self.scores = torch.empty(0, dtype=torch.float, device=device)
        self.device = device


    def __call__(self, xpl, start_index):
        xpl.to(self.device)
        most_influential_indices = xpl.argmax(axis=-1)
        for i in range(len(most_influential_indices)):
            most_influential_label = self.train[most_influential_indices[i]][1]
            test_label = self.test[start_index+i][1]
            is_equal = (test_label == most_influential_label) * 1.
            self.scores = torch.cat((self.scores, torch.tensor([is_equal]).to(self.device)), dim=0)


    def get_result(self, dir=None, file_name=None):
        self.scores = self.scores.to('cpu').numpy()
        score = self.scores.sum() / self.scores.shape[0]
        resdict = {'metric': self.name, 'all_scores': self.scores, 'avg_score': score,
                   'num_examples': self.scores.shape[0]}
        if dir is not None:
            self.write_result(resdict, dir, file_name)
        return resdict


class SameSubclassMetric(SameClassMetric):
    name = "SameSubclassMetric"

    def __init__(self, train, test, device="cuda"):
        assert isinstance(train, GroupLabelDataset)
        assert isinstance(test.dataset, GroupLabelDataset)
        super().__init__(train.dataset, test.dataset.dataset, device)
