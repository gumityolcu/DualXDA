import torch
from utils.data import GroupLabelDataset
from utils import Metric

class TopKSameClassMetric(Metric):
    name = "TopKSameClassMetric"

    def __init__(self, train, test, k=5, device="cuda"):
        if isinstance(train.targets,list):
            train.targets=torch.tensor(train.targets,device=device)
        self.train_labels = train.targets.to(device)
        self.test_labels = test.test_targets.to(device)
        self.scores = torch.empty(0, dtype=torch.float, device=device)
        self.k = k
        self.device = device

    def __call__(self, xpl, start_index):
        xpl.to(self.device)
        if xpl.nelement() == 0: #to exit if xpl is empty
            return
        print(xpl.shape)
        topk_idx = xpl.topk(self.k)[1]
        print(topk_idx.shape)
        topk_most_influential_labels = torch.stack([self.train_labels.index_select(dim=0, index=topk_idx[i]) for i in range(len(topk_idx))]).t() #throws errors if xpl is empty tensor
        test_labels = self.test_labels[start_index:start_index + xpl.shape[0]].repeat(self.k, 1)
        is_equal_ratio = torch.mean((test_labels == topk_most_influential_labels) * 1., axis=0)
        self.scores = torch.cat((self.scores, is_equal_ratio), dim=0)

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
        assert isinstance(test, GroupLabelDataset)
        super().__init__(train.dataset, test.dataset, k, device)
