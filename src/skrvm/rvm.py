"""Relevance Vector Machine classes for regression and classification."""

import warnings

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.metrics.pairwise import linear_kernel, polynomial_kernel, rbf_kernel
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.utils.validation import check_X_y


class BaseRVM(BaseEstimator):
    """Base Relevance Vector Machine class.

    Implementation of Mike Tipping's Relevance Vector Machine using the
    scikit-learn API. Add a posterior over weights method and a predict
    in subclass to use for classification or regression.
    """

    def __init__(
        self,
        kernel="rbf",
        degree=3,
        coef1=None,
        coef0=0.0,
        n_iter=3000,
        tol=1e-3,
        alpha=1e-6,
        threshold_alpha=1e9,
        beta=1.0e-6,
        beta_fixed=False,
        bias_used=True,
        verbose=False,
    ):
        """Copy params to object properties, no validation."""
        self.kernel = kernel
        self.degree = degree
        self.coef1 = coef1
        self.coef0 = coef0
        self.n_iter = n_iter
        self.tol = tol
        self.alpha = alpha
        self.threshold_alpha = threshold_alpha
        self.beta = beta
        self.beta_fixed = beta_fixed
        self.bias_used = bias_used
        self.verbose = verbose

    def get_params(self, deep=True):
        """Return parameters as a dictionary."""
        params = {
            "kernel": self.kernel,
            "degree": self.degree,
            "coef1": self.coef1,
            "coef0": self.coef0,
            "n_iter": self.n_iter,
            "tol": self.tol,
            "alpha": self.alpha,
            "threshold_alpha": self.threshold_alpha,
            "beta": self.beta,
            "beta_fixed": self.beta_fixed,
            "bias_used": self.bias_used,
            "verbose": self.verbose,
        }
        return params

    def set_params(self, **parameters):
        """Set parameters using kwargs."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def _apply_kernel(self, x, y):
        """Apply the selected kernel function to the data.

        Ensures inputs are 2D to satisfy newer scikit-learn pairwise APIs
        when users pass a single sample as a 1D array.
        """
        if isinstance(x, np.ndarray) and x.ndim == 1:
            x = x.reshape(1, -1)
        if self.kernel == "linear":
            phi = linear_kernel(x, y)
        elif self.kernel == "rbf":
            phi = rbf_kernel(x, y, self.coef1)
        elif self.kernel == "poly":
            phi = polynomial_kernel(x, y, self.degree, self.coef1, self.coef0)
        elif callable(self.kernel):
            phi = self.kernel(x, y)
            if len(phi.shape) != 2:
                raise ValueError("Custom kernel function did not return 2D matrix")
            if phi.shape[0] != x.shape[0]:
                raise ValueError(
                    "Custom kernel function did not return matrix with rows"
                    " equal to number of data points."
                )
        else:
            raise ValueError("Kernel selection is invalid.")

        if self.bias_used:
            phi = np.append(phi, np.ones((phi.shape[0], 1)), axis=1)

        return phi

    def _prune(self):
        """Remove basis functions based on alpha values."""
        keep_alpha = self.alpha_ < self.threshold_alpha

        if not np.any(keep_alpha):
            keep_alpha[0] = True
            if self.bias_used:
                keep_alpha[-1] = True

        if self.bias_used:
            if not keep_alpha[-1]:
                self.bias_used = False
            keep_relevance = keep_alpha[:-1]
        else:
            keep_relevance = keep_alpha

        pruned_indices = self.relevance_indices_[~keep_relevance]
        self.relevance_ = self.relevance_[keep_relevance]
        self.relevance_indices_ = self.relevance_indices_[keep_relevance]
        if pruned_indices.size:
            self.pruned_indices_ = np.sort(
                np.concatenate((self.pruned_indices_, pruned_indices))
            )

        self.alpha_ = self.alpha_[keep_alpha]
        self.alpha_old = self.alpha_old[keep_alpha]
        self.gamma = self.gamma[keep_alpha]
        self.phi = self.phi[:, keep_alpha]
        self.sigma_ = self.sigma_[np.ix_(keep_alpha, keep_alpha)]
        self.m_ = self.m_[keep_alpha]

    def _update_alpha(self):
        """Update relevance precisions from the posterior covariance."""
        self.gamma = 1 - self.alpha_ * np.diag(self.sigma_)
        self.alpha_ = self.gamma / (self.m_**2)

    def fit(self, X, y):
        """Fit the RVR to the training data."""
        X, y = check_X_y(X, y)

        n_samples, n_features = X.shape

        self.phi = self._apply_kernel(X, X)

        n_basis_functions = self.phi.shape[1]

        self.relevance_ = X
        self.relevance_indices_ = np.arange(n_samples)
        self.pruned_indices_ = np.array([], dtype=int)
        self.y = y

        self.alpha_ = self.alpha * np.ones(n_basis_functions)
        self.beta_ = self.beta

        self.m_ = np.zeros(n_basis_functions)

        self.alpha_old = self.alpha_

        for i in range(self.n_iter):
            self._posterior()

            self._update_alpha()

            if not self.beta_fixed:
                self.beta_ = (n_samples - np.sum(self.gamma)) / (
                    np.sum((y - np.dot(self.phi, self.m_)) ** 2)
                )

            self._prune()

            if self.verbose:
                print("Iteration: {}".format(i))
                print("Alpha: {}".format(self.alpha_))
                print("Beta: {}".format(self.beta_))
                print("Gamma: {}".format(self.gamma))
                print("m: {}".format(self.m_))
                print("Relevance Vectors: {}".format(self.relevance_.shape[0]))
                print()

            delta = np.amax(np.absolute(self.alpha_ - self.alpha_old))

            if delta < self.tol and i > 1:
                break

            self.alpha_old = self.alpha_

        if self.bias_used:
            self.bias = self.m_[-1]
        else:
            self.bias = None

        return self


class RVR(BaseRVM, RegressorMixin):
    """Relevance Vector Machine Regression.

    Implementation of Mike Tipping's Relevance Vector Machine for regression
    using the scikit-learn API.
    """

    def _posterior(self):
        """Compute the posterior distriubtion over weights."""
        i_s = np.diag(self.alpha_) + self.beta_ * np.dot(self.phi.T, self.phi)
        self.sigma_ = np.linalg.inv(i_s)
        self.m_ = self.beta_ * np.dot(self.sigma_, np.dot(self.phi.T, self.y))

    def predict(self, X, eval_MSE=False):
        """Evaluate the RVR model at x."""
        single_sample = isinstance(X, np.ndarray) and X.ndim == 1
        phi = self._apply_kernel(X, self.relevance_)

        y = np.dot(phi, self.m_)

        if eval_MSE:
            MSE = (1 / self.beta_) + np.dot(phi, np.dot(self.sigma_, phi.T))
            if single_sample:
                return y[0], MSE[0, 0]
            return y, MSE[:, 0]
        else:
            if single_sample:
                return y[0]
            return y


class RVC(BaseRVM, ClassifierMixin):
    """Relevance Vector Machine Classification.

    Implementation of Mike Tipping's Relevance Vector Machine for
    classification using the scikit-learn API.
    """

    def __init__(self, n_iter_posterior=50, multi_class="ovo", **kwargs):
        """Copy params to object properties, no validation."""
        self.n_iter_posterior = n_iter_posterior
        self.multi_class = multi_class
        super(RVC, self).__init__(**kwargs)

    def get_params(self, deep=True):
        """Return parameters as a dictionary."""
        params = super(RVC, self).get_params(deep=deep)
        params["n_iter_posterior"] = self.n_iter_posterior
        params["multi_class"] = self.multi_class
        return params

    def _classify(self, m, phi):
        return expit(np.dot(phi, m))

    def _prune(self):
        """Prune classification bases and retain the exact covariance block."""
        keep_alpha = np.isfinite(self.alpha_) & (self.alpha_ > 0)
        keep_alpha &= self.alpha_ < self.threshold_alpha

        if not np.any(keep_alpha):
            raise FloatingPointError(
                "RVC pruned every basis function. The posterior update did not "
                "produce a usable model."
            )

        if self.bias_used:
            if not keep_alpha[-1]:
                self.bias_used = False
            keep_relevance = keep_alpha[:-1]
        else:
            keep_relevance = keep_alpha

        pruned_indices = self.relevance_indices_[~keep_relevance]
        self.relevance_ = self.relevance_[keep_relevance]
        self.relevance_indices_ = self.relevance_indices_[keep_relevance]
        if pruned_indices.size:
            self.pruned_indices_ = np.sort(
                np.concatenate((self.pruned_indices_, pruned_indices))
            )

        self.alpha_ = self.alpha_[keep_alpha]
        self.alpha_old = self.alpha_old[keep_alpha]
        self.gamma = self.gamma[keep_alpha]
        self.phi = self.phi[:, keep_alpha]
        self.m_ = self.m_[keep_alpha]

        # The SVD representation is for the posterior before the ARD precision
        # update, just like ``sigma_`` in the original implementation.
        posterior_alpha = self._posterior_alpha[keep_alpha]
        right_vectors = self._posterior_right_vectors[keep_alpha]
        scaled_vectors = right_vectors / np.sqrt(posterior_alpha)[:, np.newaxis]
        self.sigma_ = np.diag(1.0 / posterior_alpha)
        self.sigma_ -= np.dot(
            scaled_vectors * self._posterior_shrinkage,
            scaled_vectors.T,
        )
        self.sigma_ = 0.5 * (self.sigma_ + self.sigma_.T)

    def _update_alpha(self):
        """Update classification precisions using stable effective dimensions."""
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            updated_alpha = self.gamma / np.square(self.m_)
        updated_alpha[~np.isfinite(updated_alpha)] = np.inf
        self.alpha_ = updated_alpha

    def _log_posterior(self, m, alpha, phi, t):
        logits = np.dot(phi, m)
        y = expit(logits)
        signed_logits = np.where(t == 1, -logits, logits)
        log_p = np.sum(np.logaddexp(0, signed_logits))
        log_p = log_p + 0.5 * np.dot(alpha, np.square(m))

        jacobian = alpha * m - np.dot(phi.T, (t - y))

        return log_p, jacobian

    def _hessian(self, m, alpha, phi, t):
        y = self._classify(m, phi)
        curvature = y * (1 - y)
        hessian = np.dot(phi.T, curvature[:, np.newaxis] * phi)
        hessian.flat[:: hessian.shape[0] + 1] += alpha
        return hessian

    def _hessian_dot(self, m, p, alpha, phi, t):
        """Multiply by the Hessian without materializing its dense matrix."""
        y = self._classify(m, phi)
        curvature = y * (1 - y)
        return alpha * p + np.dot(phi.T, curvature * np.dot(phi, p))

    def _linear_scaled_design_svd(self, curvature):
        """SVD of B**.5 Phi A**-.5 through the low-rank linear features."""
        features = self.X_
        n_features = features.shape[1]
        n_relevance = self.relevance_.shape[0]
        n_basis = n_relevance + int(self.bias_used)

        sample_factor = np.empty(
            (features.shape[0], n_features + int(self.bias_used)), dtype=np.float64
        )
        sample_factor[:, :n_features] = features

        basis_factor = np.zeros(
            (n_features + int(self.bias_used), n_basis), dtype=np.float64
        )
        basis_factor[:n_features, :n_relevance] = self.relevance_.T

        if self.bias_used:
            sample_factor[:, -1] = 1.0
            basis_factor[-1, -1] = 1.0

        sample_factor *= np.sqrt(curvature)[:, np.newaxis]
        basis_factor /= np.sqrt(self.alpha_)[np.newaxis, :]

        _, sample_triangular = np.linalg.qr(sample_factor, mode="reduced")
        basis_orthogonal, basis_triangular = np.linalg.qr(
            basis_factor.T, mode="reduced"
        )
        core = np.dot(sample_triangular, basis_triangular.T)
        _, singular_values, core_right = np.linalg.svd(core, full_matrices=False)
        right_vectors = np.dot(basis_orthogonal, core_right.T)
        return singular_values, right_vectors

    def _posterior_statistics(self):
        """Compute posterior effective dimensions without inverting the Hessian.

        If ``R = B**.5 Phi A**-.5`` and ``R = U S V.T``, then

        ``A**.5 Sigma A**.5 = I - V diag(S**2 / (1 + S**2)) V.T``.

        This form retains the prior contribution in null-space directions and
        therefore remains valid when the kernel design is rank deficient.
        """
        if not np.all(np.isfinite(self.alpha_)) or np.any(self.alpha_ <= 0):
            raise FloatingPointError(
                "RVC posterior precisions must be finite and strictly positive."
            )

        probabilities = self._classify(self.m_, self.phi)
        curvature = probabilities * (1 - probabilities)

        if self.kernel == "linear":
            singular_values, right_vectors = self._linear_scaled_design_svd(curvature)
        else:
            scaled_design = np.sqrt(curvature)[:, np.newaxis] * self.phi
            scaled_design /= np.sqrt(self.alpha_)[np.newaxis, :]
            _, singular_values, right_transpose = np.linalg.svd(
                scaled_design, full_matrices=False
            )
            right_vectors = right_transpose.T

        ratios = singular_values / np.hypot(1.0, singular_values)
        shrinkage = np.square(ratios)
        gamma = np.sum(
            np.square(right_vectors) * shrinkage[np.newaxis, :], axis=1
        )

        self.gamma = np.clip(gamma, 0.0, 1.0)
        self._posterior_alpha = self.alpha_.copy()
        self._posterior_right_vectors = right_vectors
        self._posterior_shrinkage = shrinkage

    def _posterior(self):
        result = minimize(
            fun=self._log_posterior,
            hessp=self._hessian_dot,
            x0=self.m_,
            args=(self.alpha_, self.phi, self.t),
            method="Newton-CG",
            jac=True,
            options={"maxiter": self.n_iter_posterior},
        )

        if not np.isfinite(result.fun) or not np.all(np.isfinite(result.x)):
            raise FloatingPointError(
                "RVC posterior optimization produced NaN or infinity."
            )
        if not result.success:
            warnings.warn(
                "RVC posterior optimization did not converge: {}".format(
                    result.message
                ),
                RuntimeWarning,
            )

        self.m_ = result.x
        self._posterior_statistics()

    def fit(self, X, y):
        """Check target values and fit model."""
        if self.multi_class not in {"ovo", "ovr"}:
            raise ValueError("multi_class must be either 'ovo' or 'ovr'.")

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        if n_classes < 2:
            raise ValueError("Need 2 or more classes.")
        elif n_classes == 2:
            self.t = np.zeros(y.shape)
            self.t[y == self.classes_[1]] = 1
            # Pairwise kernels of float32 features are especially ill-conditioned:
            # perform all RVC kernel and posterior arithmetic in float64.
            X = np.asarray(X, dtype=np.float64)
            self.X_ = X
            return super(RVC, self).fit(X, self.t)
        else:
            if self.multi_class == "ovo":
                self.multi_ = OneVsOneClassifier(self)
            else:
                self.multi_ = OneVsRestClassifier(self)
            self.multi_.fit(X, y)
            return self

    def predict_proba(self, X):
        """Return an array of class probabilities."""
        if len(self.classes_) > 2:
            if self.multi_class == "ovr":
                return self.multi_.predict_proba(X)
            raise AttributeError(
                "predict_proba is only available for binary classification "
                "or multi_class='ovr'."
            )

        phi = self._apply_kernel(X, self.relevance_)
        y = self._classify(self.m_, phi)
        return np.column_stack((1 - y, y))

    def predict(self, X):
        """Return an array of classes for each input."""
        if len(self.classes_) == 2:
            y = self.predict_proba(X)
            res = np.empty(y.shape[0], dtype=self.classes_.dtype)
            res[y[:, 1] <= 0.5] = self.classes_[0]
            res[y[:, 1] >= 0.5] = self.classes_[1]
            return res
        else:
            return self.multi_.predict(X)
