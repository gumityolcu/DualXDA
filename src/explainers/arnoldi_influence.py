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
from captum.influence._core.arnoldi_influence_function import ArnoldiInfluenceFunction, \
                                                              InfluenceFunctionBase, \
                                                                _extract_parameters_from_layers, \
                                                                _params_to_names, \
                                                                _functional_call, \
                                                                _compute_batch_loss_influence_function_base, \
                                                                _parameter_add, \
                                                                _dataset_fn, \
                                                                _parameter_arnoldi, \
                                                                _parameter_distill, \
                                                                _parameter_multiply, \
                                                                _parameter_to
from tqdm import tqdm
                                                                


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
    
    def _retrieve_projections_arnoldi_influence_function(
        self,
        dataloader,
        projection_on_cpu,
        show_progress,
    ):
        """

        Returns the `R` described in the documentation for
        `ArnoldiInfluenceFunction`. The returned `R` represents a set of
        parameters in parameter space. However, since this implementation does *not*
        flatten parameters, each of those parameters is represented as a tuple of
        tensors. Therefore, `R` is represented as a list of tuple of tensors, and
        can be viewed as a linear function that takes in a tuple of tensors
        (representing a parameter), and returns a vector, where the i-th entry is
        the dot-product (as it would be defined over tuple of tensors) of the parameter
        (i.e. the input to the linear function) with the i-th entry of `R`.

        Can specify that projection should always be saved on cpu. if so, gradients are
        always moved to same device as projections before multiplying (moving
        projections to gpu when multiplying would defeat the purpose of moving them to
        cpu to save gpu memory).

        Returns:
            R (list of tuple of tensors): List of tuple of tensors of length
                    `projection_dim` (initialization argument). Each element
                    corresponds to a parameter in parameter-space, is represented as a
                    tuple of tensors, and together, define a projection that can be
                    applied to parameters (represented as tuple of tensors).
        """
        # create function that computes hessian-vector product, given a vector
        # represented as a tuple of tensors

        # first figure out names of params that require gradients. this is need to
        # create that function, as it replaces params based on their names
        params = tuple(
            self.model.parameters()
            if self.layer_modules is None
            else _extract_parameters_from_layers(self.layer_modules)
        )
        # the same position in `params` and `param_names` correspond to each other
        param_names = _params_to_names(params, self.model)

        # get factory that given a batch, returns a function that given params as
        # tuple of tensors, returns loss over the batch
        # pyre-fixme[3]: Return type must be annotated.
        # pyre-fixme[2]: Parameter must be annotated.
        def tensor_tuple_loss_given_batch(batch):
            # pyre-fixme[53]: Captured variable `param_names` is not annotated.
            # pyre-fixme[53]: Captured variable `batch` is not annotated.
            # pyre-fixme[3]: Return type must be annotated.
            # pyre-fixme[2]: Parameter must be annotated.
            def tensor_tuple_loss(*params):
                # `params` is a tuple of tensors, and assumed to be order specified by
                # `param_names`
                features, labels = tuple(batch[0:-1]), batch[-1]
                features=features.to(self.model_device)
                labels=labels.to(self.model_device)
                _output = _functional_call(
                    self.model, dict(zip(param_names, params)), features
                )

                # compute the total loss for the batch, adjusting the output of
                # `self.loss_fn` based on `self.reduction_type`
                return _compute_batch_loss_influence_function_base(
                    self.loss_fn, _output, labels, self.reduction_type
                )

            return tensor_tuple_loss

        # define function that given batch and vector, returns HVP of loss using the
        # batch and vector
        # pyre-fixme[53]: Captured variable `params` is not annotated.
        # pyre-fixme[3]: Return type must be annotated.
        # pyre-fixme[2]: Parameter must be annotated.
        def batch_HVP(batch, v):
            tensor_tuple_loss = tensor_tuple_loss_given_batch(batch)
            return torch.autograd.functional.hvp(tensor_tuple_loss, params, v=v)[1]

        # define function that returns HVP of loss over `dataloader`, given a
        # specified vector
        # pyre-fixme[3]: Return type must be annotated.
        # pyre-fixme[2]: Parameter must be annotated.
        def HVP(v):
            _hvp = None

            _dataloader = dataloader
            if show_progress:
                _dataloader = tqdm(
                    dataloader, desc="processing `hessian_dataset` batch"
                )

            # the HVP of loss using the entire `dataloader` is the sum of the
            # per-batch HVP's
            return _dataset_fn(_dataloader, batch_HVP, _parameter_add, v)

            for batch in _dataloader:
                hvp = batch_HVP(batch, v)
                if _hvp is None:
                    _hvp = hvp
                else:
                    _hvp = _parameter_add(_hvp, hvp)
            return _hvp

        # now that can compute the hessian-vector product (of loss over `dataloader`),
        # can perform arnoldi iteration

        # we always perform the HVP computations on the device where the model is.
        # effectively this means we do the computations on gpu if available. this
        # is necessary because the HVP is computationally expensive.

        # get initial random vector, and place it on the same device as the model.
        # `_parameter_arnoldi` needs to know which device the model is on, and
        # will infer it through the device of this random vector
        b = _parameter_to(
            tuple(torch.randn_like(param) for param in params),
            device=self.model_device,
        )

        # perform the arnoldi iteration, see its documentation for what its return
        # values are.  note that `H` is *not* the Hessian.
        qs, H = _parameter_arnoldi(
            HVP,
            b,
            self.arnoldi_dim,
            self.arnoldi_tol,
            torch.device("cpu") if projection_on_cpu else self.model_device,
            show_progress,
        )

        # `ls`` and `vs`` are (approximately) the top eigenvalues / eigenvectors of the
        # matrix used (implicitly) to compute Hessian-vector products by the `HVP`
        # input to `_parameter_arnoldi`. this matrix is the Hessian of the loss,
        # summed over the examples in `dataloader`. note that because the vectors in
        # the Hessian-vector product are actually tuples of tensors representing
        # parameters, `vs`` is a list of tuples of tensors.  note that here, `H` is
        # *not* the Hessian (`qs` and `H` together define the Krylov subspace of the
        # Hessian)

        ls, vs = _parameter_distill(
            qs, H, self.projection_dim, self.hessian_reg, self.hessian_inverse_tol
        )

        # if `vs` were a 2D tensor whose columns contain the top eigenvectors of the
        # aforementioned hessian, then `R` would be `vs @ torch.diag(ls ** -0.5)`, i.e.
        # scaling each column of `vs` by the corresponding entry in `ls ** -0.5`.
        # however, since `vs` is instead a list of tuple of tensors, `R` should be
        # a list of tuple of tensors, where each entry in the list is scaled by the
        # corresponding entry in `ls ** 0.5`, which we first compute.
        # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
        ls = (1.0 / ls) ** 0.5

        # then, scale each entry in `vs` by the corresponding entry in `ls ** 0.5`
        # since each entry in `vs` is a tuple of tensors, we use a helper function
        # that takes in a tuple of tensors, and a scalar, and multiplies every tensor
        # by the scalar.
        return [_parameter_multiply(v, l) for (v, l) in zip(vs, ls)]


class ArnoldiInfluenceFunctionExplainer(Explainer):
    name="ArnoldiInfluenceFunctionExplainer"
    def __init__(
        self,
        model,
        dataset,
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
        dataset : torch.utils.data.Dataset
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
            dataset=dataset,
            device=device
        )

        self.dir=dir

        hessian_dataset = ArnoldiInfluenceFunctionExplainer.get_hessian_dataset(dir, hessian_dataset_size, dataset)

        explainer_kwargs = {
                "model": model,
                "train_dataset": dataset,
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

    def train(self):
        if os.path.exists(os.path.join(self.dir, "R")):
            self.captum_explainer.R = torch.load(os.path.join(self.dir, "R"), map_location=self.device)
        else:
            train_time = self.captum_explainer.compute_R()
            torch.save(self.captum_explainer.R, os.path.join(self.dir, "R"))

        if os.path.exists(os.path.join(self.dir, "train_time")):
            train_time = torch.load(os.path.join(self.dir, "train_time"))
        else:
            torch.save(train_time, os.path.join(self.dir, "train_time"))
        self.self_influences() #compute to save in cache
        return train_time

    def get_hessian_dataset(dir, hessian_dataset_size, train_dataset):
        os.makedirs(dir, exist_ok=True)
        if os.path.exists(os.path.join(dir, "hessian_dataset_indices")):
            hessian_dataset_indices = torch.load(os.path.join(dir, "hessian_dataset_indices"))
        else:
            hessian_dataset_indices = torch.randperm(len(train_dataset))[:hessian_dataset_size]
            torch.save(hessian_dataset_indices, os.path.join(dir, "hessian_dataset_indices"))
        return torch.utils.data.Subset(train_dataset, hessian_dataset_indices)

    def explain(self, x, xpl_targets):
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
        x = x.to(self.device)

        if isinstance(xpl_targets, list):
            xpl_targets = torch.tensor(xpl_targets).to(self.device)
        else:
            xpl_targets = xpl_targets.to(self.device)

        influence_scores = self.captum_explainer.influence(inputs=(x, xpl_targets))
        return influence_scores

    def self_influences(self) -> torch.Tensor:
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
        if os.path.exists(os.path.join(self.dir, "self_influences")):
            self_inf = torch.load(os.path.join(self.dir, "self_influences"))
        else:
            self_inf = self.captum_explainer.self_influence(inputs_dataset=None)
            torch.save(self_inf, os.path.join(self.dir, "self_influences"))
        return self_inf

