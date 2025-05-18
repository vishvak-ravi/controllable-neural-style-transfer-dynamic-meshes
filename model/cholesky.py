import torch
from torch.autograd import Function
from sksparse.cholmod import cholesky
import scipy.sparse as sp
import numpy as np

def build_chol_factor(A_csr: sp.csr_matrix):
    return cholesky(A_csr)              # CPU factor, reuse across steps

class CholmodSolve(Function):
    @staticmethod
    def forward(ctx, factor, x):
        dev = x.device
        x_cpu = x.detach().to('cpu')    # move RHS to CPU
        y_cpu = torch.from_numpy(factor.solve_A(x_cpu.numpy())).to(x.dtype)
        ctx.factor, ctx.dev = factor, dev
        return y_cpu.to(dev)            # send result back to original device

    @staticmethod
    def backward(ctx, grad_out):
        g_cpu = grad_out.detach().to('cpu')
        dg_dx_cpu = torch.from_numpy(ctx.factor.solve_A(g_cpu.numpy())).to(grad_out.dtype)
        return None, dg_dx_cpu.to(ctx.dev)  # same device as incoming grad

def cholmod_solve(x, factor):           # convenience wrapper
    return CholmodSolve.apply(factor, x)
