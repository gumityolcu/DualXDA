from quanda.metrics.downstream_eval import *
from copy import deepcopy
from quanda.metrics.ground_truth import LinearDatamodelingMetric
import torch
import os
from metrics import Metric
from utils.data import GroupLabelDataset

class LDS(Metric):
    name = "LDS"

    def process_cache_dir(self, pretrained_models):
        subset_indices=[]
        subset_fpath=os.path.join(self.cache_dir, "subset_indices")
        if not os.path.exists(subset_fpath):
            print(f"{subset_fpath} not found")
            # if there is no subset_indices file, that means the cache is the output of LDSCacher and
            # we handle files to have only model state_dicts in files and have a seperate subset_indices file
            for model in pretrained_models:
                fpath=os.path.join(self.cache_dir, model)
                assert os.path.isfile(fpath)
                fdict=torch.load(fpath, map_location=self.device)
                subset_indices.append(fdict["sample_indices"])
                ckpt=fdict["model_state"]
                torch.save(ckpt, fpath)
            self.subset_ids=torch.stack(subset_indices, dim=0).to(self.device)
            torch.save(subset_indices, subset_fpath)

        else:
            self.subset_ids=torch.load(subset_fpath)
            for model in pretrained_models:
                fpath=os.path.join(self.cache_dir, model)
                assert os.path.isfile(fpath)
                fdict=torch.load(fpath)
        return

    def get_test_datapoints(self, start, length):
        if length==0:
            return None, None
        targets=[]
        samples=[]
        for i in range(length):
            x, y = self.test[start+i]
            targets.append(y)
            samples.append(x)
        return torch.stack(samples).to(self.device), torch.tensor(targets, device=self.device)
    
    def __init__(self, train, test, model, pretrained_models, cache_dir, device="cuda"):
        super().__init__(train,test)
        if train.name != "AWA":
            if isinstance(train.targets,list):
                train.targets=torch.tensor(train.targets,device=device)
            self.train_labels = train.targets.to(device)
        self.device=device
        self.cache_dir=cache_dir
        self.process_cache_dir(pretrained_models)
        self.train = train
        self.test = test
        self.scores = torch.empty(0, dtype=torch.float, device=device)
        
        self.quanda_metric=LinearDatamodelingMetric(
            model=model,
            train_dataset=train,
            pretrained_models=pretrained_models,
            subset_ids=self.subset_ids,
            cache_dir=cache_dir
            )

        self.device = device

    def __call__(self, xpl, start_index):
        BATCH_SIZE=10
        xpl.to(self.device)
        if xpl.nelement() == 0: #to exit if xpl is empty
            return
        if xpl.shape[0]<=BATCH_SIZE:
            t_data,t_labels=self.get_test_datapoints(start_index, xpl.shape[0])
            self.quanda_metric.update(explanations=xpl, test_targets=t_labels, test_data=t_data)
        else:
            start_new=deepcopy(start_index)
            for i in range(int(xpl.shape[0]/BATCH_SIZE)+1):
                t_data,t_labels=self.get_test_datapoints(start_new, min(BATCH_SIZE, xpl.shape[0]-start_new))
                if t_data is not None:
                    self.quanda_metric.update(explanations=xpl[start_new:start_new+min(BATCH_SIZE, xpl.shape[0]-start_new)], test_targets=t_labels, test_data=t_data)
                else:
                    break
                start_new=start_new+BATCH_SIZE
                

    def get_result(self, dir=None, file_name=None):
        corr_scores = torch.cat(self.quanda_metric.results["scores"])
        score=corr_scores.mean().item()
        resdict = {'metric': self.name, 'correlation_scores': corr_scores , 'avg_score': score}
        if dir is not None:
            self.write_result(resdict, dir, file_name)
        return resdict

class QuandaClassDetection(Metric):
    name = "QuandaClassDetection"

    def get_test_datapoints(self, start, length):
        targets=[]
        samples=[]
        for i in range(length):
            x, y = self.test[start+i]
            targets.append(y)
            samples.append(x)
        return torch.stack(samples).to(self.device), torch.tensor(targets, device=self.device)
    
    def __init__(self, train, test, model, k=5, device="cuda"):
        super().__init__(train,test)
        if train.name != "AWA":
            if isinstance(train.targets,list):
                train.targets=torch.tensor(train.targets,device=device)
            self.train_labels = train.targets.to(device)
        self.device=device
        self.train = train
        self.test = test
        self.scores = torch.empty(0, dtype=torch.float, device=device)
        
        self.quanda_metric=ClassDetectionMetric(
            model=model,
            train_dataset=train,
            k=k
            )

        self.device = device

    def __call__(self, xpl, start_index):
        xpl.to(self.device)
        if xpl.nelement() == 0: #to exit if xpl is empty
            return
        t_data,t_labels=self.get_test_datapoints(start_index, xpl.shape[0])
        self.quanda_metric.update(explanations=xpl, test_targets=t_labels)
        
    def get_result(self, dir=None, file_name=None):
        scores = torch.cat(self.quanda_metric.scores)
        score=scores.mean().item()
        resdict = {'metric': self.name, 'detection_scores': scores , 'avg_score': score}
        if dir is not None:
            self.write_result(resdict, dir, file_name)
        return resdict

class QuandaSubclassDetection(QuandaClassDetection): 
    name = "QuandaSubclassDetection"

    def __init__(self, train, test, model,k=5, device="cuda"):
        assert isinstance(train, GroupLabelDataset)
        assert isinstance(test.dataset, GroupLabelDataset)
        super().__init__(train.dataset, test.dataset.dataset, model, k=k, device=device)

class QuandaShortcutDetection(Metric):
    name = "QuandaShortcutDetection"

    def get_test_datapoints(self, start, length):
        targets=[]
        samples=[]
        for i in range(length):
            x, y = self.test[start+i]
            targets.append(y)
            samples.append(x)
        return torch.stack(samples).to(self.device), torch.tensor(targets, device=self.device)
    
    def __init__(self, train, test, model, device="cuda"):
        super().__init__(train,test)
        if train.name != "AWA":
            if isinstance(train.targets,list):
                train.targets=torch.tensor(train.targets,device=device)
            self.train_labels = train.targets.to(device)
        self.device=device
        self.train = train
        self.test = test
        self.scores = torch.empty(0, dtype=torch.float, device=device)
        
        self.quanda_metric=ShortcutDetectionMetric(
            model=model,
            train_dataset=train,
            shortcut_indices=train.mark_samples,
            device=device,
            filter_by_prediction = True,
            filter_by_class = True,
            shortcut_cls=train.cls_to_mark
            )

    def __call__(self, xpl, start_index):
        xpl.to(self.device)
        if xpl.nelement() == 0: #to exit if xpl is empty
            return
        t_data,t_labels=self.get_test_datapoints(start_index, xpl.shape[0])
        self.quanda_metric.update(explanations=xpl, test_data=t_data, test_targets=t_labels)
        
    def get_result(self, dir=None, file_name=None):
        scores = torch.cat(self.quanda_metric.results["scores"])
        score=scores.mean().item()
        resdict = {'metric': self.name, 'detection_scores': scores , 'avg_score': score}
        if dir is not None:
            self.write_result(resdict, dir, file_name)
        return resdict