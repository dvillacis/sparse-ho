import time

import numpy as np
from numpy.linalg import norm

from sparse_ho.optimizers.base import BaseOptimizer


class NonMonotoneLineSearch(BaseOptimizer):
    """Polyak nonmonotone bilevel (PoNoBi) line search.

    This implements Algorithm (PoNoBi) provided by the user request.

    Notes
    -----
    - The iterate is the hyperparameter in *log* space (same convention as the
      rest of the optimizers in this package).
    - For line-search trial evaluations, we call `_get_val_grad(..., monitor=None)`
      to avoid adding extra entries to the user-provided `monitor`.

    Parameters
    ----------
    n_outer: int, optional (default=100)
            Maximum number of outer iterations.
    alpha_max: float, optional (default=1.0)
            Maximum step size.
    phi_star: float, optional (default=0.0)
            Lower bound / target value used in the Polyak step.
    use_best_so_far: bool, optional (default=False)
            If True, uses the best objective value seen so far (from previous outer
            iterations) as a dynamic `phi_star` in the Polyak step.
    c: float, optional (default=1e-4)
            Armijo-like parameter in the acceptance condition.
    c_p: float, optional (default=1.0)
            Polyak scaling parameter.
    eta: float, optional (default=0.5)
            Backtracking factor in (0, 1).
    xi: float, optional (default=0.9)
            Nonmonotone averaging parameter in [0, 1).
    max_ls: int, optional (default=50)
            Maximum number of backtracking steps per outer iteration.
    verbose: bool, optional (default=False)
            Verbosity.
    tol : float, optional (default=1e-5)
            Tolerance for the inner optimization solver.
    tol_decrease: bool, optional (default=None)
            If not None, uses a geometric schedule from 1e-2 to `tol`.
    t_max: float, optional (default=10_000)
            Maximum running time threshold in seconds.
    """

    def __init__(
        self,
        n_outer=100,
        alpha_max=1.0,
        phi_star=0.0,
        use_best_so_far=False,
        c=1e-4,
        c_p=1.0,
        eta=0.5,
        xi=0.9,
        max_ls=50,
        verbose=False,
        tol=1e-5,
        tol_decrease=None,
        t_max=10_000,
    ):
        self.n_outer = n_outer
        self.alpha_max = alpha_max
        self.phi_star = phi_star
        self.use_best_so_far = use_best_so_far
        self.c = c
        self.c_p = c_p
        self.eta = eta
        self.xi = xi
        self.max_ls = max_ls
        self.verbose = verbose
        self.tol = tol
        self.tol_decrease = tol_decrease
        self.t_max = t_max

    def _grad_search(self, _get_val_grad, proj_hyperparam, log_alpha0, monitor):

        is_multiparam = isinstance(log_alpha0, np.ndarray)
        if is_multiparam:
            theta = log_alpha0.copy()
        else:
            theta = log_alpha0

        theta = proj_hyperparam(theta)

        if self.tol_decrease is not None:
            tols = np.geomspace(1e-2, self.tol, num=self.n_outer)
        else:
            tols = np.ones(self.n_outer) * self.tol

        Q = 0.0
        bar_l = 0
        C_prev = None
        best_value = np.inf

        value_outer = np.inf
        grad_outer = None

        for k, tol in enumerate(tols):
            best_prev = best_value
            value_outer, grad_outer = _get_val_grad(theta, tol=tol, monitor=monitor)

            if C_prev is None:
                C_prev = value_outer

            denom = self.xi * Q + 1.0
            Ck = (self.xi * Q * C_prev + value_outer) / denom

            g_norm = norm(grad_outer)
            if (not np.isfinite(g_norm)) or g_norm <= 1e-14:
                break

            if self.use_best_so_far and np.isfinite(best_prev):
                phi_star_k = best_prev
            else:
                phi_star_k = self.phi_star

            gap = value_outer - phi_star_k
            if not np.isfinite(gap):
                break
            # With `use_best_so_far=True`, `phi_star_k` is an empirical *upper*
            # bound (best value seen so far), hence `gap` can be <= 0.
            # In that case, the Polyak step would be zero/negative; we fall
            # back to a conservative maximal step instead of stopping.
            if gap <= 0.0:
                alpha0 = self.alpha_max
            else:
                alpha_tilde = gap / (self.c_p * (g_norm**2) + 1e-18)
                alpha0 = min(alpha_tilde, self.alpha_max)

            if alpha0 <= 0.0:
                break

            alpha = alpha0
            l_k = 0
            accepted = False
            value_trial = np.inf

            while l_k <= self.max_ls:
                theta_trial = proj_hyperparam(theta - alpha * grad_outer)
                value_trial, _ = _get_val_grad(theta_trial, tol=tol, monitor=None)

                rhs = Ck - self.c * alpha * (g_norm**2)
                if value_trial <= rhs:
                    accepted = True
                    theta = theta_trial
                    break

                l_k += 1
                # IMPORTANT: `_get_val_grad` updates criterion warm-start state.
                # If we reject the trial step, we must reset that state to the
                # current accepted iterate before trying another step size.
                _get_val_grad(theta, tol=tol, monitor=None)
                alpha = alpha0 * (self.eta ** (bar_l + l_k))
                if alpha <= 1e-30:
                    break

                if monitor is not None and (time.time() - monitor.t0) > self.t_max:
                    break

            bar_l = max(bar_l + l_k - 1, 0)
            Q = self.xi * Q + 1.0
            C_prev = Ck
            if accepted:
                best_value = min(best_value, value_trial)
            else:
                best_value = min(best_value, value_outer)

            if self.verbose:
                print(
                    "Iteration %i/%i || " % (k + 1, self.n_outer)
                    + "Value outer criterion: %.2e || " % value_outer
                    + "norm grad %.2e || " % g_norm
                    + "alpha %.2e || " % alpha
                    + "accepted %s || " % accepted
                    + "ls iters %i || " % l_k
                    + "bar_l %i" % bar_l
                )

            if monitor is not None:
                if len(monitor.times) > 0 and monitor.times[-1] > self.t_max:
                    break
                if (time.time() - monitor.t0) > self.t_max:
                    break

        return theta, value_outer, grad_outer
