"""FastRVM feature-kernel explainer."""

import os
import time

import numpy as np
import torch
from fastrvm import RVC

from utils.explainers import FeatureKernelExplainer

__all__ = ["FastRVM"]


class FastRVM(FeatureKernelExplainer):
    """Explain predictions with FastRVM's linear one-vs-rest classifier.

    FastRVM exposes a global union of relevance-vector indices and aligns every
    class's dual coefficients to that union. This explainer scatters that
    matrix back into the full training-set order expected by
    ``FeatureKernelExplainer``. Pruned samples consequently receive zero
    coefficients. Intercepts are saved separately because they do not belong
    to any training sample's attribution.
    """

    name = "FastRVMExplainer"
    multi_class = "ovr"

    def __init__(
        self,
        model,
        dataset,
        device,
        dir,
        features_dir,
        use_preds=False,
        max_iter=10000,
        fit_intercept=False,
        n_jobs=None,
        prioritize_addition=False,
        prioritize_deletion=True,
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
        self.dir = dir
        self.features_dir = features_dir
        self.use_preds = use_preds
        self.rvm_params = {
            "max_iter": max_iter,
            "fit_intercept": fit_intercept,
            "n_jobs": n_jobs,
            "prioritize_addition": prioritize_addition,
            "prioritize_deletion": prioritize_deletion,
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
        """Load fitted FastRVM artifacts from disk."""
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
        )
        return all(
            os.path.isfile(self._artifact_path(name)) for name in artifact_names
        )

    def _extract_ovr_parameters(self, rvm):
        """Scatter FastRVM's union-aligned dual matrix into training order."""
        classes = np.asarray(rvm.classes_)
        expected_classes = np.arange(classes.size)
        if not np.array_equal(classes, expected_classes):
            raise ValueError(
                "FastRVM explanations require zero-based contiguous integer class labels."
            )

        global_indices = np.asarray(rvm.relevance_, dtype=int)
        dual_coefficients = np.asarray(rvm.dual_coef_)
        expected_shape = (classes.size, global_indices.size)
        if dual_coefficients.shape != expected_shape:
            raise RuntimeError(
                "FastRVM's dual coefficient matrix is not aligned with its relevance indices."
            )

        n_samples = self.normalized_samples.shape[0]
        coefficients = torch.zeros(
            (n_samples, classes.size), dtype=torch.float32, device=self.device
        )
        indices = torch.as_tensor(
            global_indices, dtype=torch.long, device=self.device
        )
        coefficients[indices] = torch.as_tensor(
            dual_coefficients.T, dtype=torch.float32, device=self.device
        )
        biases = torch.as_tensor(
            rvm.intercept_, dtype=torch.float32, device=self.device
        )

        relevance_indices = []
        for class_index in range(classes.size):
            class_positions = np.flatnonzero(rvm.alpha_[class_index] != 0)
            if class_positions.size != rvm.n_relevance_[class_index]:
                raise RuntimeError(
                    "FastRVM's per-class relevance counts do not match its alpha matrix."
                )
            relevance_indices.append(
                torch.as_tensor(
                    global_indices[class_positions],
                    dtype=torch.long,
                    device=self.device,
                )
            )

        return coefficients, biases, torch.as_tensor(classes), relevance_indices

    def _build_classifier(self):
        return RVC(kernel="linear", **self.rvm_params)

    def _extract_parameters(self, rvm, labels):
        return self._extract_ovr_parameters(rvm)

    def train(self):
        """Fit or load the configured linear FastRVM surrogate."""
        if not os.path.isfile(os.path.join(self.features_dir, "samples")):
            torch.save(self.samples, os.path.join(self.features_dir, "samples"))
        if not os.path.isfile(os.path.join(self.features_dir, "labels")):
            torch.save(self.labels, os.path.join(self.features_dir, "labels"))

        if self._rvm_artifacts_exist():
            self.read_variables()
            return self.train_time

        start_time = time.perf_counter()
        samples = self.normalized_samples.detach().cpu().numpy()
        labels = self.labels.detach().cpu().numpy()
        classes = np.unique(labels)
        if classes.size < 3:
            raise ValueError("The FastRVM explainer requires at least three classes.")
        if not np.array_equal(classes, np.arange(classes.size)):
            raise ValueError(
                "FastRVM explanations require zero-based contiguous integer class labels."
            )

        rvm = self._build_classifier()
        rvm.fit(samples, labels)

        accuracy = rvm.score(samples, labels)
        print(f"FastRVM Accuracy: {accuracy:.2f}")

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
        print(f"Training took {self.train_time} seconds")
        return self.train_time
