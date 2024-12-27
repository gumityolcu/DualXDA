from utils.explainers import Explainer
from kronfluence.task import Task
import torch
import torch.nn.functional as F
from kronfluence.analyzer import prepare_model, Analyzer
from kronfluence.arguments import FactorArguments, ScoreArguments

from time import time
import os

class ClassificationTask(Task):

    def compute_train_loss(
        self,
        batch,
        model,
        sample = False,
    ):
        inputs, labels = batch
        logits = model(inputs)
        with torch.no_grad():
            probs = torch.nn.functional.softmax(logits.detach(), dim=-1)
            sampled_labels = torch.multinomial(
                probs,
                num_samples=1,
            ).flatten()
        targets = sampled_labels if sample else labels
        return F.cross_entropy(logits, targets, reduction="sum")

    def compute_measurement(
        self,
        batch,
        model,
    ):
        inputs, labels = batch
        logits = model(inputs)
        return F.cross_entropy(logits, labels, reduction="sum")        

class KronfluenceExplainer(Explainer):
    name="KronfluenceExplainer"
    def __init__(
            self,
            model,
            dataset,
            device,
            dir,
            covariance_max_examples=100000,
            covariance_data_partitions=1,
            lambda_max_examples=100000,
            lambda_data_partitions=1,
            use_iterative_lambda_aggregation=False,
            score_data_partitions=1,
            batch_size=32,
            disable_tqdm=True
    ):
        self.factor_kwargs={
            "covariance_max_examples":covariance_max_examples,
            "covariance_data_partitions":covariance_data_partitions,
            "lambda_max_examples":lambda_max_examples,
            "lambda_data_partitions":lambda_data_partitions,
            "use_iterative_lambda_aggregation":use_iterative_lambda_aggregation,
        }
        self.score_kwargs={
            "data_partitions":score_data_partitions
        }
        super(KronfluenceExplainer, self).__init__(model, dataset, device)
        self.dir = dir
        os.makedirs(self.dir,exist_ok=True)
        self.dataset=dataset
        self.batch_size=batch_size
        task=ClassificationTask()
        model=prepare_model(model=self.model, task=task)
        self.analyzer = Analyzer(
        analysis_name="exp",
        model=model,
        task=task,
        disable_tqdm=disable_tqdm,
        output_dir=dir,
    )
        
    def train(self):
        t=time()
        factor_args = FactorArguments(strategy="ekfac", ** self.factor_kwargs)
        self.analyzer.fit_all_factors(
        factors_name="exp_factors",
        factor_args=factor_args,
        dataset=self.dataset,
        per_device_batch_size=None,
        overwrite_output_dir=False,# this parameter allows loading from cache
    )   
        t=time()-t
        if os.path.exists(os.path.join(self.dir, "train_time")):
            self.train_time = torch.load(os.path.join(self.dir, "train_time"), map_location=self.device)
        else:
            torch.save(t, os.path.join(self.dir, "train_time"))
            self.train_time=t
        return self.train_time
        
    def explain(self, x, xpl_targets):
        eval_dataset=torch.utils.data.TensorDataset(x, xpl_targets)
        score_args = ScoreArguments(**self.score_kwargs)
        self.analyzer.compute_pairwise_scores(
            scores_name="exp_scores",
            score_args=score_args,
            factors_name="exp_factors",
            query_dataset=eval_dataset,
            train_dataset=self.dataset,
            per_device_query_batch_size=self.batch_size,
            overwrite_output_dir=False,# this parameter allows loading from cache
        )
        xpl = self.analyzer.load_pairwise_scores("exp_scores")["all_modules"]
        return xpl
        
    def self_influences(self):
        if os.path.exists(os.path.join(self.dir, "self_influences")):
            return torch.load(os.path.join(self.dir, "self_influences"),map_location=self.device)
        else:
            score_args = ScoreArguments(**self.score_kwargs)
            self.analyzer.compute_self_scores(scores_name="self", factors_name="exp_factors", score_args=score_args, train_dataset=self.dataset)
            scores = self.analyzer.load_self_scores(scores_name="self")["all_modules"]
            torch.save(scores, os.path.join(self.dir, "self_influences"))
            return scores
