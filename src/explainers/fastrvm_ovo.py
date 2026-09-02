"""One-vs-one FastRVM feature-kernel explainer."""

from itertools import combinations

import numpy as np
import torch
from fastrvm import RVC
from sklearn.multiclass import OneVsOneClassifier

from .fastrvm import FastRVM

__all__ = ["FastRVMOvO"]


class FastRVMOvO(FastRVM):
    """Average signed attributions from pairwise FastRVM classifiers."""

    name = "FastRVMOvO"
    multi_class = "ovo"

    def _build_classifier(self):
        estimator = RVC(kernel="linear", **self.rvm_params)
        return OneVsOneClassifier(estimator)

    def _extract_parameters(self, rvm, labels):
        """Map pair-local FastRVM coefficients to the original training rows."""
        classes = np.asarray(rvm.classes_)
        expected_classes = np.arange(classes.size)
        if not np.array_equal(classes, expected_classes):
            raise ValueError(
                "FastRVM explanations require zero-based contiguous integer class labels."
            )

        class_pairs = list(combinations(range(classes.size), 2))
        if len(rvm.estimators_) != len(class_pairs):
            raise RuntimeError(
                "The number of one-vs-one FastRVM estimators does not match "
                "the class pairs."
            )

        n_samples = self.normalized_samples.shape[0]
        coefficients = torch.zeros(
            (n_samples, classes.size), dtype=torch.float32, device=self.device
        )
        biases = torch.zeros(classes.size, dtype=torch.float32, device=self.device)
        relevance_indices = []
        divisor = classes.size - 1

        for estimator, (negative_class, positive_class) in zip(
            rvm.estimators_, class_pairs
        ):
            pair_mask = np.logical_or(
                labels == classes[negative_class], labels == classes[positive_class]
            )
            pair_indices = np.flatnonzero(pair_mask)
            global_indices = pair_indices[np.asarray(estimator.relevance_, dtype=int)]
            indices = torch.as_tensor(
                global_indices, dtype=torch.long, device=self.device
            )
            values = torch.as_tensor(
                estimator.dual_coef_.ravel(),
                dtype=torch.float32,
                device=self.device,
            )

            if indices.numel() != values.numel():
                raise RuntimeError(
                    "The FastRVM relevance indices and dual coefficients have "
                    "different lengths."
                )

            averaged_values = values / divisor
            coefficients[indices, negative_class] -= averaged_values
            coefficients[indices, positive_class] += averaged_values

            averaged_bias = float(estimator.intercept_[0]) / divisor
            biases[negative_class] -= averaged_bias
            biases[positive_class] += averaged_bias
            relevance_indices.append(indices)

        return coefficients, biases, torch.as_tensor(classes), relevance_indices
