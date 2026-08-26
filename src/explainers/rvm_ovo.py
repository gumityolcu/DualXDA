"""One-vs-one Relevance Vector Machine feature-kernel explainer."""

from itertools import combinations

import numpy as np
import torch

from .rvm import RVM

__all__ = ["RVMOvO"]


class RVMOvO(RVM):
    """Average signed attributions across all associated pairwise RVMs.

    For a pairwise classifier between classes ``a`` and ``b``, its raw score
    is oriented toward ``b`` by scikit-learn's one-vs-one encoding. Therefore,
    its coefficients contribute negatively to class ``a`` and positively to
    class ``b``. Each class receives contributions from ``n_classes - 1``
    pairwise classifiers, so their sum is divided by that number.
    """

    name = "RVMOvO"
    multi_class = "ovo"

    def _extract_parameters(self, rvm, labels):
        """Map pair-local coefficients to global rows and average by class."""
        classes = np.asarray(rvm.classes_)
        expected_classes = np.arange(classes.size)
        if not np.array_equal(classes, expected_classes):
            raise ValueError(
                "RVM explanations require zero-based contiguous integer class labels."
            )

        class_pairs = list(combinations(range(classes.size), 2))
        estimators = rvm.multi_.estimators_
        if len(estimators) != len(class_pairs):
            raise RuntimeError(
                "The number of one-vs-one RVM estimators does not match the class pairs."
            )

        n_samples = self.normalized_samples.shape[0]
        coefficients = torch.zeros(
            (n_samples, classes.size), dtype=torch.float32, device=self.device
        )
        biases = torch.zeros(classes.size, dtype=torch.float32, device=self.device)
        relevance_indices = []
        divisor = classes.size - 1

        for estimator, (negative_class, positive_class) in zip(
            estimators, class_pairs
        ):
            pair_mask = np.logical_or(
                labels == classes[negative_class], labels == classes[positive_class]
            )
            pair_indices = np.flatnonzero(pair_mask)
            global_indices = pair_indices[estimator.relevance_indices_]
            indices = torch.as_tensor(
                global_indices, dtype=torch.long, device=self.device
            )
            posterior_mean = estimator.m_[:-1] if estimator.bias_used else estimator.m_
            values = torch.as_tensor(
                posterior_mean, dtype=torch.float32, device=self.device
            )

            if indices.numel() != values.numel():
                raise RuntimeError(
                    "The RVM relevance indices and posterior means have different lengths."
                )

            averaged_values = values / divisor
            coefficients[indices, negative_class] -= averaged_values
            coefficients[indices, positive_class] += averaged_values

            if estimator.bias is not None:
                averaged_bias = float(estimator.bias) / divisor
                biases[negative_class] -= averaged_bias
                biases[positive_class] += averaged_bias
            relevance_indices.append(indices)

        return coefficients, biases, torch.as_tensor(classes), relevance_indices
