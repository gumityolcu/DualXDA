import torch
from utils.data import GroupLabelDataset
from utils import Metric

class TopKSameClassMetric(Metric):
    name = "TopKSameClassMetric"

    def __init__(self, train, test, k=5, device="cuda"):
        if train.name != "AWA":
            if isinstance(train.targets,list):
                train.targets=torch.tensor(train.targets,device=device)
            self.train_labels = train.targets.to(device)
        self.train = train
        self.test = test
        self.scores = torch.empty(0, dtype=torch.float, device=device)
        self.k = k
        self.device = device

    def __call__(self, xpl, start_index):
        xpl.to(self.device)
        if xpl.nelement() == 0: #to exit if xpl is empty
            return
        topk_idx = xpl.topk(self.k)[1]
        for i in range(topk_idx.shape[0]):
            topk_most_influential_labels = torch.Tensor([self.train[topk_idx[i, j]][1] for j in range(self.k)]).to('cpu')
            test_label = torch.tensor([self.test[start_index+i][1] for _ in range(self.k)]).to('cpu')
            is_equal = torch.mean((test_label == topk_most_influential_labels) * 1., axis=0)[None]
            self.scores = torch.cat((self.scores, torch.tensor(is_equal).to(self.device)), dim=0)

    def get_result(self, dir=None, file_name=None):
        self.scores = self.scores.to('cpu').numpy()
        score = self.scores.sum() / self.scores.shape[0]
        resdict = {'metric': self.name, 'all_scores': self.scores, 'avg_score': score,
                   'num_examples': self.scores.shape[0]}
        if dir is not None:
            self.write_result(resdict, dir, file_name)
        return resdict


class TopKSameSubclassMetric(TopKSameClassMetric):
    name = "TopKSameSubclassMetric"

    def __init__(self, train, test, k=5, device="cuda"):
        assert isinstance(train, GroupLabelDataset)
        assert isinstance(test.dataset, GroupLabelDataset)
        super().__init__(train.dataset, test.dataset.dataset, k, device)
