import matplotlib.pyplot as plt
import torch
import os
from utils import Metric


class CorruptLabelMetric(Metric):
    name = "CorruptLabelMetric"

    def __init__(self, train, test, device="cuda"):
        self.corrupt_labels = train.corrupt_labels.to(device)
        self.corrupt_samples = train.corrupt_samples.to(device)
        self.scores = torch.zeros(len(train), dtype=torch.float, device=device)
        self.device = device
        self.num_test_samples = 0
        self.ds_name = train.dataset.name

    def __call__(self, selfinf):
        self.scores=selfinf     
    
    def compute_score(self, score_array):
        sorted_indices = torch.argsort(score_array, descending=True)
        detected = []
        det_count = 0
        for id in sorted_indices:
            if id in self.corrupt_samples:
                det_count += 1
            detected.append(det_count)
        detected = torch.tensor(detected, dtype=torch.float, device=self.device)
        detected = detected / self.corrupt_samples.shape[0]
        return detected.sum() / self.scores.shape[0], detected

    def add_coef_evaluation(self, resdict, coefs):
        max_score = resdict['max_score']
        min_score = resdict['min_score']

        score, curve = self.compute_score(coefs)
        resdict['coefs_auc_score'] = (score-min_score) / (max_score-min_score)
        resdict['coefs_label_flipping_curve'] = curve
        return resdict

    def get_result(self, dir, file_name, coef_influences=None):
        print("Self.corrupt_samples  ", type(self.corrupt_samples))
        if type(self.corrupt_samples) == dict:
            print(self.corrupt_samples.keys())
        max_score = (self.corrupt_samples.shape[0] + 1) / 2 + self.scores.shape[0] - self.corrupt_samples.shape[0]
        max_score = max_score / self.scores.shape[0]
        score, curve = self.compute_score(self.scores)
        min_score = self.corrupt_samples.shape[0] / (2 * self.scores.shape[0])
        score = (score - min_score) / (max_score - min_score)
        resdict = {'metric': self.name, 'auc_score': score, 'label_flipping_curve': curve,
                   'num_examples': self.num_test_samples, 'num_corrupt_samples': self.corrupt_samples.shape[0],
                   'max_score': max_score, "min_score":min_score}

        plt.figure()
        plt.plot(curve.to("cpu"))
        plt.savefig(os.path.join(dir,f"{file_name.replace('.json', '')}_corrupt_plot.png"))
        if coef_influences is not None:
            resdict=self.add_coef_evaluation(resdict, coef_influences)
        if dir is not None:
            self.write_result(resdict, dir, file_name)
        return resdict
