"""Main file to modify for submissions.

Once renamed or symlinked as `main.py`, it will be used by `petric.py` as follows:

>>> from main import Submission, submission_callbacks
>>> from petric import data, metrics
>>> algorithm = Submission(data)
>>> algorithm.run(np.inf, callbacks=metrics + submission_callbacks)
"""

import math
import sirf.STIR as STIR
from cil.optimisation.algorithms import Algorithm
from cil.optimisation.utilities import callbacks
from sirf.contrib.partitioner import partitioner

from collections.abc import Callable

import numpy as np
import importlib.util
import warnings

if importlib.util.find_spec("cupy") is not None:
    import array_api_compat.cupy as xp
else:
    warnings.warn("cupy not found!. Using numpy as fallback", UserWarning)
    import array_api_compat.numpy as xp

from array_api_compat import to_device

# import pure python re-implementation of the RDP -> only used to get diagonal of the RDP Hessian!
from rdp import RDP

from petric import Dataset
import torch
import torch.nn.functional as F
from scipy.ndimage import correlate

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_divisors(n):
    """Returns a sorted list of all divisors of a positive integer n."""
    divisors = set()
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            divisors.add(i)
            divisors.add(n // i)
    return sorted(divisors)


def step_size_rule_1(update: int) -> float:
    if update <= 8:
        new_step_size = 3.0
    elif update <= (40):
        new_step_size = 2.0
    elif update <= 60:
        new_step_size = 1.5
    else:
        new_step_size = 1.0

    return new_step_size


def convolution_torch(tensor, kernel):
    """
    3D cross-correlation of tensor (C, H, W) with kernel (c, h, w).
    Uses torch.nn.functional.conv3d.
    """
    C, H, W = tensor.shape
    c, h, w = kernel.shape

    input_5d = tensor.unsqueeze(0).unsqueeze(0)     # (1, 1, C, H, W)
    kernel_5d = kernel.unsqueeze(0).unsqueeze(0)     # (1, 1, c, h, w)

    pad_c = c - 1
    pad_h = h - 1
    pad_w = w - 1

    padding = (
        pad_w // 2, pad_w - pad_w // 2,
        pad_h // 2, pad_h - pad_h // 2,
        pad_c // 2, pad_c - pad_c // 2,
    )
    input_5d = F.pad(input_5d, padding)
    output = F.conv3d(input_5d, kernel_5d)
    return output.squeeze(0).squeeze(0)           # (C, H, W)

def convolution_scipy(tensor_np: np.ndarray, kernel_np: np.ndarray) -> np.ndarray:
    """
    3D cross-correlation of tensor (C, H, W) with kernel (c, h, w).
    Uses scipy.ndimage.correlate (cross-correlation, no flip) with zero-padding.
    """
    c, h, w = kernel_np.shape

    origin = (
        (c - 1) // 2 - c // 2,
        (h - 1) // 2 - h // 2,
        (w - 1) // 2 - w // 2,
    )

    return correlate(tensor_np, kernel_np, mode="constant", cval=0.0, origin=origin)

class MaxIteration(callbacks.Callback):
    """
    The organisers try to `Submission(data).run(inf)` i.e. for infinite iterations (until timeout).
    This callback forces stopping after `max_iteration` instead.
    """

    def __init__(self, max_iteration: int, verbose: int = 1):
        super().__init__(verbose)
        self.max_iteration = max_iteration

    def __call__(self, algorithm: Algorithm):
        if algorithm.iteration >= self.max_iteration:
            raise StopIteration


class Submission(Algorithm):
    """
    Better preconditioned SVRG for PETRIC (MaGeZ ALG1)
    """

    def __init__(
        self,
        data: Dataset,
        approx_num_subsets: int = 20,  # approximate number of subsets, closest divisor of num_views will be used
        update_objective_interval: int | None = None,
        complete_gradient_epochs: tuple[int, ...] = tuple(
            [x for x in range(0, 1000, 1)]
        ),
        step_size_update_function: Callable[[int], float] = step_size_rule_1,
        precond_update_epochs: tuple[int, ...] = (1, 2, 3),
        pre_filter_fwhm_mm: float = 6.0,
        precond_hessian_factor: float = 2.0,
        precond_filter_fwhm_mm: float = 3.0,
        verbose: bool = False,
        seed: int = 42,
        do_torch_convolution: bool = True,
        **kwargs,
    ):
        """better pre-conditioned SVRG for PETRIC
           where the preconditioner is based on the voxel-wise harmonic mean
           of the inverse of the diagonal of the Hessian of the RDP and x/(A^T 1)

        Parameters
        ----------
        data : Dataset
            a PETRIC dataset
        approx_num_subsets : int, optional
            the approximate number of subsets to use,
            the actual number is chosen as the closeset divisor of thenumber of views,
            by default 20
        update_objective_interval : int | None, optional
            interval by which the objective / metrics are updated, by default None
        complete_gradient_epochs : tuple[int, ...], optional
            epoch numbers in which all subset gradients are calculated for SVRG,
            by default tuple([x for x in range(0, 1000, 1)])
        step_size_update_function : Callable[[int], float], optional
            function that return the step size given the update (iteration) number,
            by default step_size_rule_1
        precond_update_epochs : tuple[int, ...], optional
            epoch numbers in which the preconditioner is updated,
            by default (1, 2, 3)
        precond_hessian_factor : float, optional
            fudge factor in the harmonic mean that accounts for non-diag RDP Hessian elements,
            by default 2.0
        precond_filter_fwhm_mm : float, optional
            FWHM (mm) of Gaussian filter that is applied to the image before preconditioner is calculated,
            by default 3.0
        pre_filter_fwhm_mm : float, optional
            FWHM (mm) of Gaussian filter that is applied to the image before any processing,
            by default 6.0
        verbose : bool, optional
            print verbose outout, by default False
        seed : int, optional
            seed for numpy random generator (used to choose subsets), by default 1
        """


        self._do_torch_convolution = do_torch_convolution

        if self._do_torch_convolution:
            self._all_kernels = torch.load("all_kernels.pt").to(device)
        else:
            self._all_kernels = torch.load("all_kernels.pt").cpu().numpy()

        np.random.seed(seed)

        self._verbose = verbose
        self.subset = 0

        # --- setup the number of subsets

        num_views = data.mult_factors.dimensions()[2]
        num_views_divisors = np.array(get_divisors(num_views))
        self._num_subsets = num_views_divisors[
            np.argmin(np.abs(num_views_divisors - approx_num_subsets))
        ]

        self._kernel_iterations = min([self._num_subsets, self._all_kernels.shape[0] - 1])

        if self._num_subsets not in num_views_divisors:
            raise ValueError(
                f"Number of subsets {self._num_subsets} is not a divisor of {num_views}. Divisors are {num_views_divisors}"
            )

        if self._verbose:
            print(f"num_subsets: {self._num_subsets}")

        # --- setup the initial image as a slightly smoothed version of the OSEM image
        self.x = data.OSEM_image.clone()

        pre_filter = STIR.SeparableGaussianImageFilter()
        pre_filter.set_fwhms(
            [pre_filter_fwhm_mm, pre_filter_fwhm_mm, pre_filter_fwhm_mm]
        )
        pre_filter.set_up(self.x)

        self.x = pre_filter.process(self.x.clone())

        self._update = 0
        self._step_size_update_function = step_size_update_function
        self._step_size = self._step_size_update_function(self._update)
        self._subset_number_list = []
        self._precond_hessian_factor = precond_hessian_factor

        self._data_sub, self._acq_models, self._subset_likelihood_funcs = (
            partitioner.data_partition(
                data.acquired_data,
                data.additive_term,
                data.mult_factors,
                self._num_subsets,
                initial_image=data.OSEM_image,
                mode="staggered",
            )
        )

        penalization_factor = data.prior.get_penalisation_factor()

        # WARNING: modifies prior strength with 1/num_subsets (as currently needed for BSREM implementations)
        data.prior.set_penalisation_factor(penalization_factor / self._num_subsets)
        data.prior.set_up(data.OSEM_image)

        self._subset_prior_fct = data.prior

        self._adjoint_ones = self.x.get_uniform_copy(0)

        for i in range(self._num_subsets):
            if self._verbose:
                print(f"Calculating subset {i} sensitivity")
            self._adjoint_ones += self._subset_likelihood_funcs[
                i
            ].get_subset_sensitivity(0)

        self._fov_mask = data.FOV_mask

        # add a small number in the adjoint ones outside the FOV to avoid NaN in division
        self._adjoint_ones.maximum(1e-6, out=self._adjoint_ones)

        # initialize list / ImageData for all subset gradients and sum of gradients
        self._summed_subset_gradients = self.x.get_uniform_copy(0)
        self._subset_gradients = []

        self._complete_gradient_epochs = complete_gradient_epochs

        self._precond_update_epochs = precond_update_epochs

        # setup python re-implementation of the RDP
        # only used to get the diagonal of the RDP Hessian for preconditioning!
        # (diag of RDP Hessian is not available in SIRF yet)
        if "cupy" in xp.__name__:
            self._dev = xp.cuda.Device(0)
        else:
            self._dev = "cpu"

        self._python_prior = RDP(
            data.OSEM_image.shape,
            xp,
            self._dev,
            xp.asarray(data.OSEM_image.spacing, device=self._dev),
            eps=data.prior.get_epsilon(),
            gamma=data.prior.get_gamma(),
        )
        self._python_prior.kappa = xp.asarray(
            data.kappa.as_array().astype(xp.float64), device=self._dev
        )
        self._python_prior.scale = penalization_factor

        # small relative number for the preconditioner (to avoid zeros in the preconditioner)
        self._precond_delta_rel = 1e-6

        self._precond_filter = STIR.SeparableGaussianImageFilter()
        self._precond_filter.set_fwhms(
            [precond_filter_fwhm_mm, precond_filter_fwhm_mm, precond_filter_fwhm_mm]
        )
        self._precond_filter.set_up(data.OSEM_image)

        # calculate the initial preconditioner based on the initial image
        self._precond = self.calc_precond(self.x)

        if update_objective_interval is None:
            update_objective_interval = self._num_subsets

        super().__init__(update_objective_interval=update_objective_interval, **kwargs)
        self.configured = True  # required by Algorithm

    @property
    def epoch(self) -> int:
        """Return the current epoch number"""
        return self._update // self._num_subsets

    def calc_precond(
        self,
        x: STIR.ImageData,
    ) -> STIR.ImageData:
        """calculate the preconditioner based on the current image

        Parameters
        ----------
        x : STIR.ImageData
            current image

        Returns
        -------
        STIR.ImageData
            preconditioner
        """

        if self._verbose:
            print("Updating preconditioner")

        # generate a smoothed version of the input image
        # to avoid high values, especially in first and last slices
        x_sm = self._precond_filter.process(x)

        prior_diag_hess = x_sm.get_uniform_copy(0)
        prior_diag_hess.fill(
            to_device(
                self._python_prior.diag_hessian(
                    xp.asarray(x_sm.as_array().astype(xp.float64), device=self._dev)
                ),
                "cpu",
            )
        )

        if self._precond_delta_rel > 0:
            x_sm += self._precond_delta_rel * x_sm.max()

        precond = (
            self._fov_mask * x_sm
            / (
                self._adjoint_ones
                + self._precond_hessian_factor * prior_diag_hess * x_sm
            )
        )

        return precond

    def update_all_subset_gradients(self) -> None:
        """calculate all subset gradients and the sum of gradients based on the current image"""

        if self._verbose:
            print("recalculating all subset gradients")

        self._summed_subset_gradients = self.x.get_uniform_copy(0)
        self._subset_gradients = []

        subset_prior_gradient = self._subset_prior_fct.gradient(self.x)

        # remember that the objective has to be maximized
        # posterior = log likelihood - log prior ("minus" instead of "plus"!)
        for i in range(self._num_subsets):
            self._subset_gradients.append(
                self._subset_likelihood_funcs[i].gradient(self.x)
                - subset_prior_gradient
            )
            self._summed_subset_gradients += self._subset_gradients[i]

    def update(self):
        """perform a single pre-conditioned SVRG update (iteration) step"""

        update_all_subset_gradients = (
            self._update % self._num_subsets == 0
        ) and self.epoch in self._complete_gradient_epochs

        update_precond = (
            self._update % self._num_subsets == 0
        ) and self.epoch in self._precond_update_epochs

        if self._verbose:
            print(self._update, self._step_size)

        if update_precond:
            self._precond = self.calc_precond(self.x)

        if update_all_subset_gradients:
            self.update_all_subset_gradients()
            approximated_gradient = self._summed_subset_gradients
        else:
            if self._subset_number_list == []:
                self.create_subset_number_list()

            self.subset = self._subset_number_list.pop()
            if self._verbose:
                print(f" {self._update}, {self.subset}, subset gradient update")

            subset_prior_gradient = self._subset_prior_fct.gradient(self.x)

            # remember that the objective has to be maximized
            # posterior = log likelihood - log prior ("minus" instead of "plus"!)
            approximated_gradient = (
                (
                    (
                        self._subset_likelihood_funcs[self.subset].gradient(self.x)
                        - subset_prior_gradient
                    )
                    - self._subset_gradients[self.subset]
                ) * self._num_subsets
                + self._summed_subset_gradients
            )

        # update the step size based on the current update number and the current step size
        if self._update > self._kernel_iterations:
            self._step_size = self._step_size_update_function(self._update)
            ### Objective has to be maximized -> "+" for gradient ascent
            self.x = self.x + self._step_size * self._precond * approximated_gradient
        else:
            if self._do_torch_convolution:
                new_kernel = self._all_kernels[self._update]
                init_update = self._precond * approximated_gradient
                init_update_gpu = torch.from_numpy(init_update.asarray()).to(device)
                print(f"Doing torch conv, they are on device: {init_update_gpu.device}, {new_kernel.device}")
                update_gpu = convolution_torch(init_update_gpu, new_kernel)
                init_update.fill(update_gpu.cpu().numpy())
                self.x = self.x + init_update
            else:
                new_kernel = self._all_kernels[self._update]
                init_update = self._precond * approximated_gradient
                print(f"Doing scipy conv, they are on device: {init_update.asarray().device}, {new_kernel.device}")
                update_np = convolution_scipy(init_update.asarray(), new_kernel)
                init_update.fill(update_np)
                self.x = self.x + init_update
        # enforce non-negative constraint
        self.x.maximum(0, out=self.x)

        self._update += 1

    def update_objective(self) -> None:
        """
        NB: The objective value is not required by OSEM nor by PETRIC, so this returns `0`.
        NB: It should be `sum(prompts * log(acq_model.forward(self.x)) - self.x * sensitivity)` across all subsets.
        """

        self.loss.append(0)

    def create_subset_number_list(self) -> None:
        """create a list of all subset numbers (in shuffled order)"""

        tmp = np.arange(self._num_subsets)
        np.random.shuffle(tmp)
        self._subset_number_list = tmp.tolist()


submission_callbacks = []
# submission_callbacks = [MaxIteration(660)]