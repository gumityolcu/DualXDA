from utils import Metric
import torch
import copy

class LeaveOutMetric(Metric):
    name = "LeaveOutMetric"

    def __init__(self, train, test, model, batchsize, device="cuda"):
        self.train_labels = train.dataset.targets.to(device)
        self.test_labels = test.dataset.targets.to(device)
        self.model_retrain = copy.deepcopy(model)

    
    def __call__(self, xpl, start_index):
        xpl.to(self.device)
        ids_sorted = xpl.argsort(dim=-1, descending=True)
        for iter in range((len(self.train) // self.batchsize) + 1):
            train_leave_out = torch.cat([self.train[:self.batchsize*iter], self.train[self.batchsize*(iter+1):]])
            train_labels_leave_out = torch.cat([self.train_labels[:self.batchsize*iter], self.train_labels[self.batchsize*(iter+1):]])
            # retrain self.model_retrain
            pred_logits = self.model_retrain(self.test)
            # extract logit of interest (that of the correct test_label)
        # for every testpoint have logit for every testpoint and left out batch
        # how to calculate score from this? 


    def retrain(train): # from scratch


####

class MarkImageMetric(Metric):
    name = "MarkImageMetric"

    @staticmethod
    def get_pred_labels(model, test, marked_label, device):
        loader = torch.utils.data.DataLoader(test, batch_size=64, shuffle=False)
        labels = torch.empty(0, dtype=torch.int, device=device)
        misclassified=0
        model.eval()
        counts=0
        cur_index=0
        counted_test_indices=torch.zeros(0,device=device)
        for x, y in iter(loader):
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            batch_labels=torch.argmax(logits, dim=-1)
            labels = torch.concat((labels, batch_labels), dim=0)
            counts=counts+(y!=marked_label).sum()
            misclassified=misclassified+((batch_labels[y!=marked_label]==marked_label)).sum()
            counted_test_indices=torch.concat((counted_test_indices,cur_index+torch.where((batch_labels==2)*(y!=2))[0]),dim=0)
            cur_index=cur_index+x.shape[0]
        print(float(misclassified)/float(counts))
        print(f"{misclassified} samples classified wrongly as the marked class over {counts} candidate test samples")
        return labels

    def __init__(self, train, test, model, device="cuda"):
        self.marked_cls = train.cls_to_mark
        self.marked_samples = train.mark_samples.to(device)
        targs=train.dataset.targets
        if isinstance(targs,list):
            targs=torch.tensor(targs,device=device)
        self.train_labels = targs.to(device)
        self.pred_labels = self.get_pred_labels(model, test, self.marked_cls, device).to(device)
        self.test_labels = test.dataset.test_targets.to(device)
        nonmarked_class_samples = torch.nonzero(self.test_labels != self.marked_cls).squeeze().to(device)
        self.eval_ids = torch.tensor([i for i in nonmarked_class_samples if self.pred_labels[i]==self.marked_cls]).to(device)
        self.scores = torch.empty(0, dtype=torch.float, device=device)
        self.device = device

    def __call__(self, xpl, start_index):
        xpl.to(self.device)
        most_influential_ids = xpl.argmax(axis=-1)
        passed = [1. if most_influential_ids[i] in self.marked_samples else 0.
                  for i in range(xpl.shape[0])]
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