from time import time
import os

import torch
from captum._utils.av import AV 
from captum.influence import (  
    SimilarityInfluence,
    TracInCP,
    TracInCPFast,
    TracInCPFastRandProj,
)

# TODO Should be imported directly from captum.influence once available
from captum.influence._core.arnoldi_influence_function import ArnoldiInfluenceFunction
from captum.influence._core.influence_function import InfluenceFunctionBase


from utils.explainers import Explainer

## todo checkpoint_load_func should be created here and checkpoint is a dict which is taken from the supplied model

class CustomArnoldiInfluenceFunction(ArnoldiInfluenceFunction):
    def __init__(
        self,
        model,
        train_dataset,
        layers = None,
        loss_fn = None,
        batch_size = 1,
        hessian_dataset = None,
        test_loss_fn = None,
        sample_wise_grads_per_batch = False,
        projection_dim = 50,
        seed = 0,
        arnoldi_dim = 200,
        arnoldi_tol = 1e-1,
        hessian_reg = 1e-3,
        hessian_inverse_tol = 1e-4,
        projection_on_cpu = True,
        show_progress = False
    ):
        checkpoint = model.state_dict()
        def checkpoints_load_func(model, ckpt):
            model.load_state_dict(ckpt) 
        InfluenceFunctionBase.__init__(
            self,
            model,
            train_dataset,
            checkpoint,
            checkpoints_load_func,
            layers,
            loss_fn,
            batch_size,
            hessian_dataset,
            test_loss_fn,
            sample_wise_grads_per_batch,
        )

        self.projection_dim = projection_dim
        torch.manual_seed(seed)  # for reproducibility

        self.arnoldi_dim = arnoldi_dim
        self.arnoldi_tol = arnoldi_tol
        self.hessian_reg = hessian_reg
        self.hessian_inverse_tol = hessian_inverse_tol
        self.projection_on_cpu = projection_on_cpu
        self.show_progress = show_progress
        # infer the device the model is on.  all parameters are assumed to be on the
        # same device
        self.model_device = next(model.parameters()).device

    def compute_R(self):
        t=time()
        self.R = self._retrieve_projections_arnoldi_influence_function(
            self.hessian_dataloader,
            self.projection_on_cpu,
            self.show_progress,
        ) 
        return t - time()

class ArnoldiInfluenceFunctions(Explainer):

    def __init__(
        self,
        model,
        train_dataset,
        dir,
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none"),
        layers = None,
        batch_size = 1,
        test_loss_fn = None,
        sample_wise_grads_per_batch = False,
        projection_dim = 50,
        seed = 0,
        arnoldi_dim = 200,
        arnoldi_tol = 1e-1,
        hessian_dataset_size = 5000,
        hessian_reg = 1e-3,
        hessian_inverse_tol = 1e-4,
        projection_on_cpu = True,
        show_progress = False,
        device = "cuda",
    ):
        """
        Initializer for CaptumArnoldi explainer.

        Parameters
        ----------
        model : Union[torch.nn.Module, pl.LightningModule]
            The model to be used for the influence computation.
        train_dataset : torch.utils.data.Dataset
            Training dataset to be used for the influence computation.
        checkpoint : str
            Checkpoint file for the model.
        loss_fn : Union[torch.nn.Module, Callable], optional
            Loss function which is applied to the model. Required to be a reduction='none' loss.
            Defaults to CrossEntropyLoss with reduction='none'.
        layers : Optional[List[str]], optional
            Layers used to compute the gradients. If None, all layers are used. Defaults to None.
        batch_size : int, optional
            Batch size used for iterating over the dataset. Defaults to 1.
        hessian_dataset : Optional[torch.utils.data.Dataset], optional
            Dataset for calculating the Hessian. It should be smaller than train_dataset.
            If None, the entire train_dataset is used. Defaults to None.
        test_loss_fn : Optional[Union[torch.nn.Module, Callable]], optional
            Loss function which is used for the test samples. If None, loss_fn is used. Defaults to None.
        sample_wise_grads_per_batch : bool, optional
            Whether to compute sample-wise gradients per batch. Defaults to False.
            Note: This feature is currently not supported.
        projection_dim : int, optional
            Captum's ArnoldiInfluenceFunction produces a low-rank approximation of the (inverse) Hessian.
            projection_dim is the rank of that approximation. Defaults to 50.
        seed : int, optional
            Random seed for reproducibility. Defaults to 0.
        arnoldi_dim : int, optional
            Calculating the low-rank approximation of the (inverse) Hessian requires approximating
            the Hessian's top eigenvectors / eigenvalues.
            This is done by first computing a Krylov subspace via the Arnoldi iteration,
            and then finding the top eigenvectors / eigenvalues of the restriction of the Hessian to the Krylov subspace.
            Because only the top eigenvectors / eigenvalues computed in the restriction will be similar to
            those in the full space, `arnoldi_dim` should be chosen to be larger than `projection_dim`.
            Defaults to 200.
        arnoldi_tol : float, optional
            After many iterations, the already-obtained basis vectors may already approximately span the Krylov subspace,
            in which case the addition of additional basis vectors involves normalizing a vector with a small norm.
            These vectors are not necessary to include in the basis and furthermore,
            their small norm leads to numerical issues.
            Therefore we stop the Arnoldi iteration when the addition of additional
            vectors involves normalizing a vector with norm below a certain threshold.
            This argument specifies that threshold. Defaults to 1e-1.
        hessian_reg : float, optional
            After computing the basis for the Krylov subspace, the restriction of the Hessian to the
            subspace may not be positive definite, which is required, as we compute a low-rank approximation
            of its square root via eigen-decomposition. `hessian_reg` adds an entry to the diagonals of the
            restriction of the Hessian to encourage it to be positive definite. This argument specifies that entry.
            Note that the regularized Hessian (i.e. with `hessian_reg` added to its diagonals) does not actually need
            to be positive definite - it just needs to have at least 1 positive eigenvalue.
            Defaults to 1e-3.
        hessian_inverse_tol : float, optional
            The tolerance to use when computing the pseudo-inverse of the (square root of) hessian,
            restricted to the Krylov subspace. Defaults to 1e-4.
        projection_on_cpu : bool, optional
            Whether to move the projection, i.e. low-rank approximation of the inverse Hessian, to cpu, to save gpu memory.
            Defaults to True.
        show_progress : bool, optional
            Whether to display a progress bar. Defaults to False.
        device : Union[str, torch.device], optional
            Device to run the computation on. Defaults to "cpu".
        **explainer_kwargs : Any
            Additional keyword arguments passed to the explainer.
        """
        
        super().__init__(
            model=model,
            train_dataset=train_dataset,
            device=device
        )

        self.dir=dir

        hessian_dataset = self.get_hessian_dataset(dir, hessian_dataset_size, train_dataset)

        explainer_kwargs = {
                "model": model,
                "train_dataset": self.train_dataset,
                "layers": layers,
                "loss_fn": loss_fn,
                "batch_size": batch_size,
                "hessian_dataset": hessian_dataset,
                "test_loss_fn": test_loss_fn,
                "sample_wise_grads_per_batch": sample_wise_grads_per_batch,
                "projection_dim": projection_dim,
                "seed": seed,
                "arnoldi_dim": arnoldi_dim,
                "arnoldi_tol": arnoldi_tol,
                "hessian_reg": hessian_reg,
                "hessian_inverse_tol": hessian_inverse_tol,
                "projection_on_cpu": projection_on_cpu,
                "show_progress": show_progress,
            }
        
        self.captum_explainer = CustomArnoldiInfluenceFunction(**explainer_kwargs)

        self.train()

    def train(self):
        train_time = self.captum_explainer.compute_R()

        if os.path.exists(os.path.join(self.dir, "train_time")):
            train_time = torch.load(os.path.join(self.dir, "train_time"))
        else:
            torch.save(train_time, os.path.join(self.dir, "train_time"))

        return train_time

    def get_hessian_dataset(dir, hessian_dataset_size, train_dataset):
        os.makedirs(dir, exists_ok=True)
        if os.path.exists(os.path.join(dir, "hessian_dataset_indices")):
            hessian_dataset_indices = torch.load(os.path.join(dir, "hessian_dataset_indices"))
        else:
            hessian_dataset_indices = torch.randperm(len(train_dataset))[:hessian_dataset_size]
        return torch.utils.data.Subset(train_dataset, hessian_dataset_indices)

    def explain(self, test_tensor, targets):
        """
        Compute influence scores for the test samples.

        Parameters
        ----------
        test_tensor
            Test samples for which influence scores are computed.
        targets
            Labels for the test samples. This argument is required.

        Returns
        -------
        torch.Tensor
            2D Tensor of shape (test_samples, train_dataset_size) containing the influence scores.
        """
        test_tensor = test_tensor.to(self.device)

        if isinstance(targets, list):
            targets = torch.tensor(targets).to(self.device)
        else:
            targets = targets.to(self.device)

        influence_scores = self.captum_explainer.influence(inputs=(test_tensor, targets))
        return influence_scores

    def self_influence(self, batch_size) -> torch.Tensor:
        """
        Compute self-influence scores.

        Parameters
        ----------
        batch_size : int, optional
            Batch size used for iterating over the dataset. This argument is ignored.

        Returns
        -------
        torch.Tensor
            Self-influence scores for each datapoint in train_dataset.
        """
        influence_scores = self.captum_explainer.self_influence(inputs_dataset=None)
        return influence_scores

