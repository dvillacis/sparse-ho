import numpy as np
from numpy.linalg import norm

from sparse_ho.optimizers.base import BaseOptimizer


def _dot(a, b):
	try:
		return float(np.dot(a.ravel(), b.ravel()))
	except Exception:
		return float(a * b)


class TrustRegion(BaseOptimizer):
	"""Trust-region optimizer for the outer problem.

	This method builds a first-order (linear) local model of the objective and
	proposes steps constrained in an Euclidean trust region.

	Parameters
	----------
	n_outer: int, optional (default=100)
		Maximum number of outer iterations.
	radius0: float, optional (default=1.0)
		Initial trust-region radius (in log-hyperparameter space).
	radius_min: float, optional (default=1e-12)
		Minimal trust-region radius.
	radius_max: float, optional (default=1e2)
		Maximal trust-region radius.
	eta_accept: float, optional (default=0.1)
		Acceptance threshold for the ratio of actual/predicted decrease.
	eta_expand: float, optional (default=0.75)
		Threshold above which the trust-region radius can be expanded.
	gamma_dec: float, optional (default=0.25)
		Multiplicative decrease factor for the trust-region radius.
	gamma_inc: float, optional (default=2.0)
		Multiplicative increase factor for the trust-region radius.
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
			self, n_outer=100, radius0=1.0, radius_min=1e-12, radius_max=1e2,
			eta_accept=0.1, eta_expand=0.75, gamma_dec=0.25, gamma_inc=2.0,
			verbose=False, tol=1e-5, tol_decrease=None, t_max=10_000):
		self.n_outer = n_outer
		self.radius0 = radius0
		self.radius_min = radius_min
		self.radius_max = radius_max
		self.eta_accept = eta_accept
		self.eta_expand = eta_expand
		self.gamma_dec = gamma_dec
		self.gamma_inc = gamma_inc
		self.verbose = verbose
		self.tol = tol
		self.tol_decrease = tol_decrease
		self.t_max = t_max

	def _grad_search(
			self, _get_val_grad, proj_hyperparam, log_alpha0, monitor):

		is_multiparam = isinstance(log_alpha0, np.ndarray)
		if is_multiparam:
			log_alphak = log_alpha0.copy()
		else:
			log_alphak = log_alpha0

		log_alphak = proj_hyperparam(log_alphak)

		if self.tol_decrease is not None:
			tols = np.geomspace(1e-2, self.tol, num=self.n_outer)
		else:
			tols = np.ones(self.n_outer) * self.tol

		radius = float(self.radius0)
		value_outer, grad_outer = _get_val_grad(log_alphak, tols[0], monitor)

		for i, tol in enumerate(tols):
			if i > 0:
				value_outer, grad_outer = _get_val_grad(log_alphak, tol, monitor)

			grad_norm = norm(grad_outer)
			if not np.isfinite(grad_norm) or grad_norm <= 1e-14:
				break

			step = -(radius / grad_norm) * grad_outer
			trial = proj_hyperparam(log_alphak + step)
			step_eff = trial - log_alphak

			pred = -_dot(grad_outer, step_eff)
			if (not np.isfinite(pred)) or pred <= 1e-18:
				radius = max(self.radius_min, self.gamma_dec * radius)
				if self.verbose:
					print(
						"Iteration %i/%i || " % (i + 1, self.n_outer) +
						"rejected (nonpositive predicted decrease) || " +
						"radius %.2e" % radius
					)
				if len(monitor.times) > 0 and monitor.times[-1] > self.t_max:
					break
				continue

			value_trial, grad_trial = _get_val_grad(trial, tol, monitor)
			ared = value_outer - value_trial
			rho = ared / pred

			if rho < self.eta_accept:
				radius = max(self.radius_min, self.gamma_dec * radius)
				accepted = False
			else:
				accepted = True
				log_alphak = trial
				value_outer = value_trial
				grad_outer = grad_trial

				if (rho > self.eta_expand) and (norm(step_eff) >= 0.95 * radius):
					radius = min(self.radius_max, self.gamma_inc * radius)

			if self.verbose:
				print(
					"Iteration %i/%i || " % (i + 1, self.n_outer) +
					"Value outer criterion: %.2e || " % value_outer +
					"norm grad %.2e || " % grad_norm +
					"rho %.2e || " % rho +
					"accepted %s || " % accepted +
					"radius %.2e" % radius
				)

			if len(monitor.times) > 0 and monitor.times[-1] > self.t_max:
				break

		return log_alphak, value_outer, grad_outer
