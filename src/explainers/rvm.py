"""Relevance Vector Machine feature-kernel explainer."""

import os
import time

import numpy as np
import torch
from skrvm import RVC

from utils.explainers import FeatureKernelExplainer

__all__ = ["RVM"]


class RVM(FeatureKernelExplainer):
    """Explain predictions with a linear one-vs-rest RVM surrogate.

    The fitted binary RVM for each class retains only a sparse subset of the
    training samples. Its posterior means are scattered back into a dense
    training-sample-by-class coefficient matrix so that the standard
    ``FeatureKernelExplainer`` decomposition can be used. Bias terms are saved
    separately and are intentionally excluded from per-training-sample
    attributions.
    """

    name = "RVMExplainer"
    multi_class = "ovr"
    solver_version = 2

    def __init__(
        self,
        model,
        dataset,
        device,
        dir,
        features_dir,
        use_preds=False,
        n_iter=3000,
        n_iter_posterior=50,
        tol=1e-3,
        alpha=1e-6,
        threshold_alpha=1e9,
        beta=1e-6,
        beta_fixed=False,
        bias_used=True,
        verbose=False,
        normalize=False,
    ):
        super().__init__(
            model,
            dataset,
            device,
            dir=features_dir,
            normalize=normalize,
        )
        self.dir = os.path.normpath(dir)
        self.features_dir = os.path.normpath(features_dir)
        self.use_preds = use_preds
        self.rvm_params = {
            "n_iter": n_iter,
            "n_iter_posterior": n_iter_posterior,
            "tol": tol,
            "alpha": alpha,
            "threshold_alpha": threshold_alpha,
            "beta": beta,
            "beta_fixed": beta_fixed,
            "bias_used": bias_used,
            "verbose": verbose,
        }
        self.biases = None
        self.classes = None
        self.relevance_indices = None
        os.makedirs(self.dir, exist_ok=True)
        os.makedirs(self.features_dir, exist_ok=True)

    def _artifact_path(self, name):
        return os.path.join(self.dir, name)

    def read_variables(self):
        """Load fitted RVM artifacts from disk."""
        self.learned_weight = torch.load(
            self._artifact_path("weights"), map_location=self.device
        ).float()
        self.coefficients = torch.load(
            self._artifact_path("coefficients"), map_location=self.device
        ).float()
        self.train_time = torch.load(
            self._artifact_path("train_time"), map_location=self.device
        ).float()
        self.biases = torch.load(
            self._artifact_path("biases"), map_location=self.device
        ).float()
        self.classes = torch.load(
            self._artifact_path("classes"), map_location=self.device
        ).long()
        self.relevance_indices = torch.load(
            self._artifact_path("relevance_indices"), map_location=self.device
        )

        cache_time_path = os.path.join(self.features_dir, "cache_time")
        if os.path.isfile(cache_time_path):
            cache_time = torch.load(cache_time_path, map_location=self.device)
            self.cache_time = torch.as_tensor(cache_time, device=self.device).float()
        else:
            self.cache_time = torch.tensor(0.0, device=self.device)

    def _rvm_artifacts_exist(self):
        artifact_names = (
            "weights",
            "coefficients",
            "train_time",
            "biases",
            "classes",
            "relevance_indices",
            "multiclass_strategy",
            "solver_version",
        )
        if not all(os.path.isfile(self._artifact_path(name)) for name in artifact_names):
            return False
        cached_strategy = torch.load(
            self._artifact_path("multiclass_strategy"), map_location="cpu"
        )
        cached_solver_version = torch.load(
            self._artifact_path("solver_version"), map_location="cpu"
        )
        return (
            cached_strategy == self.multi_class
            and cached_solver_version == self.solver_version
        )

    def _extract_ovr_parameters(self, rvm):
        """Convert the fitted OvR estimators to full training-set tensors."""
        classes = np.asarray(rvm.classes_)
        expected_classes = np.arange(classes.size)
        if not np.array_equal(classes, expected_classes):
            raise ValueError(
                "RVM explanations require zero-based contiguous integer class labels."
            )

        n_samples = self.normalized_samples.shape[0]
        coefficients = torch.zeros(
            (n_samples, classes.size), dtype=torch.float32, device=self.device
        )
        biases = torch.zeros(classes.size, dtype=torch.float32, device=self.device)
        relevance_indices = []

        for class_index, estimator in enumerate(rvm.multi_.estimators_):
            indices = torch.as_tensor(
                estimator.relevance_indices_, dtype=torch.long, device=self.device
            )
            posterior_mean = estimator.m_[:-1] if estimator.bias_used else estimator.m_
            values = torch.as_tensor(
                posterior_mean, dtype=torch.float32, device=self.device
            )

            if indices.numel() != values.numel():
                raise RuntimeError(
                    "The RVM relevance indices and posterior means have different lengths."
                )

            coefficients[indices, class_index] = values
            if estimator.bias is not None:
                biases[class_index] = float(estimator.bias)
            relevance_indices.append(indices)

        return coefficients, biases, torch.as_tensor(classes), relevance_indices

    def _extract_parameters(self, rvm, labels):
        """Extract attribution parameters from the fitted multiclass RVM."""
        return self._extract_ovr_parameters(rvm)

    def train(self):
        """Fit or load the configured linear multiclass RVM surrogate."""
        if not os.path.isfile(os.path.join(self.features_dir, "samples")):
            torch.save(self.samples, os.path.join(self.features_dir, "samples"))
        if not os.path.isfile(os.path.join(self.features_dir, "labels")):
            torch.save(self.labels, os.path.join(self.features_dir, "labels"))

        if self._rvm_artifacts_exist():
            self.read_variables()
            return self.train_time

        start_time = time.perf_counter()
        rvm = RVC(kernel="linear", multi_class=self.multi_class, **self.rvm_params)
        samples = self.normalized_samples.detach().cpu().numpy()
        labels = self.labels.detach().cpu().numpy()
        if np.unique(labels).size < 3:
            raise ValueError("The RVM explainer requires a dataset with at least three classes.")
        rvm.fit(samples, labels)

        accuracy = rvm.score(samples, labels)
        print(f"RVM Accuracy: {accuracy:.2f}")

        (
            self.coefficients,
            self.biases,
            self.classes,
            self.relevance_indices,
        ) = self._extract_parameters(rvm, labels)
        self.learned_weight = self.coefficients.T @ self.normalized_samples.float()
        self.train_time = torch.tensor(time.perf_counter() - start_time)

        torch.save(self.train_time, self._artifact_path("train_time"))
        torch.save(self.learned_weight, self._artifact_path("weights"))
        torch.save(self.coefficients, self._artifact_path("coefficients"))
        torch.save(self.biases, self._artifact_path("biases"))
        torch.save(self.classes, self._artifact_path("classes"))
        torch.save(self.relevance_indices, self._artifact_path("relevance_indices"))
        torch.save(self.multi_class, self._artifact_path("multiclass_strategy"))
        torch.save(self.solver_version, self._artifact_path("solver_version"))
        print(f"Training took {self.train_time} seconds")
        return self.train_time
