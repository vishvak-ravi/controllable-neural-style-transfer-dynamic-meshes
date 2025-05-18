# pip install scikit-sparse torch scipy
import torch
from torch.autograd import Function
from sksparse.cholmod import cholesky  # CHOLMOD via SuiteSparse
import scipy.sparse as sp


def build_chol_factor(A_csr: sp.csr_matrix):
    """
    A_csr must be SPD in CSR format.
    Returns a CHOLMOD factor object that lives on the CPU.
    Call once and reuse; factor is immutable.
    """
    return cholesky(A_csr)  # numeric factorisation L such that A = L Lᵀ


class CholmodSolve(Function):
    """
    y = A⁻¹ x using a pre-factored sparse A.
    Gradients:  ∂ℓ/∂x = A⁻¹ (∂ℓ/∂y)
    No gradient w.r.t. A (constant).
    """

    @staticmethod
    def forward(ctx, factor, x):
        # x: (n, …) dense torch tensor on CPU, requires_grad=True
        x_np = x.detach().cpu().numpy()
        y_np = factor.solve_A(x_np)  # sparse triangular solves, no dense A⁻¹ built
        y = torch.from_numpy(y_np).to(x.dtype)

        ctx.factor = factor  # saved for backward, not registered as tensor
        return y

    @staticmethod
    def backward(ctx, grad_out):
        # need grad_out on CPU
        g_np = grad_out.detach().cpu().numpy()
        dg_dx_np = ctx.factor.solve_A(g_np)  # same sparse solve
        dg_dx = torch.from_numpy(dg_dx_np).to(grad_out.dtype)
        return None, dg_dx  # None for factor (no grad)


def cholmod_solve(x, factor):
    """Convenience wrapper -- matches torch.cholesky_solve signature order."""
    return CholmodSolve.apply(factor, x)
