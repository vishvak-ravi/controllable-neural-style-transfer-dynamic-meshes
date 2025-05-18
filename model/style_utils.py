import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.transforms.transforms import (
    ToTensor,
    CenterCrop,
    Compose,
    Resize,
    ConvertImageDtype,
)
from torchvision.transforms.functional import rotate
from pytorch3d.structures import Meshes
from sksparse.cholmod import cholesky
import scipy.sparse as sp

from enum import Enum
from typing import Optional
import time

VGG_INPUT_SHAPE = (224, 224)
style_transform = Compose([Resize(VGG_INPUT_SHAPE[0]), ToTensor(), CenterCrop(VGG_INPUT_SHAPE)])
render_transform = Compose([ConvertImageDtype(torch.float32), Resize(VGG_INPUT_SHAPE)])

def get_rotated_style_tensors(img: torchvision.utils.Image.Image, num_rots: int) -> torch.Tensor:
    img = style_transform(img)
    rotated_imgs = torch.stack([rotate(img, 360.0 * i / num_rots) for i in range(num_rots)])
    return rotated_imgs

class VGGStyleExtractor(nn.Module):
    def __init__(self, output_size=(56, 56)):
        super().__init__()
        vgg = torchvision.models.vgg16(
            weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1
        )
        # take layers up to ReLU after conv4_3 (indices 0–22)
        self.features = nn.ModuleList(list(vgg.features.children())[:23])
        self.output_size = output_size

    def forward(self, x):
        interpolated_features = []
        for layer in self.features:
            x = layer(x)

            if isinstance(layer, nn.ReLU):
                interpolated_features.append(
                    F.interpolate(
                        x, size=self.output_size, mode="bilinear", align_corners=False
                    )
                )
        interpolated_features = torch.cat(interpolated_features, dim=1)
        interpolated_features = interpolated_features - torch.mean(
            interpolated_features, dim=(-1, -2)
        ).unsqueeze(-1).unsqueeze(-1)
        return interpolated_features


def hard_nearest_neighbor_replacement(ref_feats: torch.Tensor,
                                 extracted_feats: torch.Tensor) -> torch.Tensor:
    """
    Each feature vector in `extracted_feats` (B,C,H,W) is replaced by the
    closest feature (ℓ2) in `ref_feats` (R,C,H,W).  R is usually 1.
    Output shape = (B,C,H,W).
    """
    R, C, H, W = ref_feats.shape            # reference
    B          = extracted_feats.shape[0]    # extracted batches

    # (R·H·W, C) and (B·H·W, C)
    ref_flat = ref_feats.permute(0,2,3,1).reshape(-1, C)
    ext_flat = extracted_feats.permute(0,2,3,1).reshape(-1, C)

    # pair-wise distances: (BHW, RHW)  -> argmin over references
    idx      = torch.cdist(ext_flat, ref_flat).argmin(dim=1)

    # gather nearest reference vectors
    nearest  = ref_flat[idx]                # (BHW, C)

    # reshape back to (B,C,H,W)
    return nearest.reshape(B, H, W, C).permute(0,3,1,2)

def soft_nearest_neighbor(
    style: torch.Tensor,          # (1,C,Hs,Ws) or (C,Hs,Ws)
    content: torch.Tensor,        # (N,C,H,W)
    tau: float = 10.0,
    eps: float = 1e-8,
    chunk: Optional[int] = None,  # max #content patches per block (memory cap)
) -> torch.Tensor:
    """
    Differentiable, batched, cosine-based soft NN replacement.

    Returns: (N,C,H,W)
    """
    if style.dim() == 4:                        # squeeze batch dim if present
        style = style.squeeze(0)               # (C,Hs,Ws)

    C, Hs, Ws = style.shape
    N, _, H, W = content.shape
    Ps, P = Hs * Ws, H * W                     # #style / #content patches

    # --- flatten ---
    style_flat = style.permute(1, 2, 0).reshape(Ps, C)          # (Ps,C)
    cont_flat  = content.permute(0, 2, 3, 1).reshape(N, P, C)   # (N,P,C)

    # --- ℓ2-normalise (cosine sim) ---
    style_n = style_flat / (style_flat.norm(dim=1, keepdim=True) + eps)     # (Ps,C)
    cont_n  = cont_flat  / (cont_flat.norm(dim=2, keepdim=True) + eps)      # (N,P,C)

    style_n_T  = style_n.t().unsqueeze(0)        # (1,C,Ps)
    style_flat_T = style_flat.unsqueeze(0)       # (1,Ps,C)

    # helper to compute block [i:j] to fit memory
    def _replace(block: torch.Tensor) -> torch.Tensor:          # block: (N,b,C)
        sim = torch.bmm(block, style_n_T.expand(N, -1, -1))     # (N,b,Ps)
        w   = torch.softmax(sim * tau, dim=2)                   # (N,b,Ps)
        return torch.bmm(w, style_flat_T.expand(N, -1, -1))     # (N,b,C)

    if chunk is None or P <= chunk:
        recon = _replace(cont_n)                                # (N,P,C)
    else:                                                       # chunked to save mem
        outs = []
        for i in range(0, P, chunk):
            j = min(i + chunk, P)
            outs.append(_replace(cont_n[:, i:j]))
        recon = torch.cat(outs, dim=1)                          # (N,P,C)

    return recon.reshape(N, H, W, C).permute(0, 3, 1, 2).contiguous()

def nearest_neighbor_replacement(
    style_features: torch.Tensor,          # (C, Hs, Ws) or (B, C, Hs, Ws)
    content_features: torch.Tensor,        # (N, C, H, W)
    tau: float = 10.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Soft-nearest-neighbor feature replacement with stable gradients.
    Returns (N, C, H, W).
    """
    # ---------- flatten style patches ----------
    if style_features.dim() == 4:          # batched styles
        B, C, Hs, Ws = style_features.shape
        style_features = (style_features      # (B, C, Hs, Ws)
                          .permute(1, 0, 2, 3) # → (C, B, Hs, Ws)
                          .reshape(C, B * Hs * Ws))
    else:                                  # single style image
        C, Hs, Ws = style_features.shape
        style_features = style_features.reshape(C, Hs * Ws)  # (C, Ps)

    # ---------- flatten content patches ----------
    N, C_c, H, W = content_features.shape
    assert C_c == style_features.size(0), "channel mismatch"
    P = H * W
    cf = content_features.reshape(N, C, P)          # (N, C, P)

    # ---------- cosine similarities ----------
    cf_norm = cf / (cf.norm(dim=1, keepdim=True) + eps)          # (N, C, P)
    sf_norm = style_features / (style_features.norm(dim=0, keepdim=True) + eps)  # (C, Ps)
    sims = torch.bmm(cf_norm.transpose(1, 2),                    # (N, P, C)
                     sf_norm.expand(N, -1, -1))                  # (N, P, Ps)

    # ---------- soft assignment ----------
    weights = torch.softmax(sims * tau, dim=2)                   # (N, P, Ps)

    # ---------- reconstruction ----------
    recon = torch.bmm(weights,                                   # (N, P, Ps)
                      style_features.t().expand(N, -1, -1))      # (N, P, C)
    return recon.permute(0, 2, 1).reshape(N, C, H, W)



class LaplacianRoutine(Enum):
    TORCH3D = 0
    CUSTOM = 1


def get_combinatorial_laplacian(
    mesh: Meshes, routine: LaplacianRoutine, log_time: bool
) -> torch.Tensor:
    
    if log_time:
        t0 = time.time()

    # build undirected edge list
    # packed verts/faces
    verts = mesh.verts_packed()  # (V, 3)
    faces = mesh.faces_packed()  # (F, 3)
    V = verts.size(0)
    fe = torch.cat(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], dim=0
    )  # (3F, 2)
    edges = torch.cat([fe, fe[:, [1, 0]]], dim=0)  # both directions
    edges = torch.unique(edges, dim=0)  # (E, 2)
    i, j = edges.t()
    # off-diag: -1 for each edge
    off_vals = -torch.ones(i.size(0), device=verts.device)

    # degree = sum of weights (1) over neighbors
    deg = torch.zeros(V, device=verts.device)
    deg.scatter_add_(0, i, -off_vals)

    # assemble sparse Laplacian in COO format
    idx = torch.cat(
        [edges, torch.stack([torch.arange(V, device=verts.device)] * 2, dim=1)], dim=0
    ).t()  # (2, E + V)
    vals = torch.cat([off_vals, deg], dim=0)  # (E + V,)

    # convert COO tensor to CSC format by first converting to CSR then transposing
    L_coo = torch.sparse_coo_tensor(idx, vals, (V, V))
    
    if log_time:
        print(f"Got laplacian in {time.time() - t0}s")
    
    return L_coo


class CholeskyFactorRoutine(Enum):
    TORCH = 0
    CHOLMOD = 1
    MAGMA_DENSE = 2
    MAGMA_SPARSE = 3


def cholesky_factor(
    A: torch.Tensor,
    routine: CholeskyFactorRoutine,
    log_time: bool,
    device: torch.device,
    dense: bool,
):
    """
    A must be COO sparse and may return a dense or sparse lower cholesky factor based on the routine.
    Assumption is A is too large to fit onto GPU, so L is always returned on CPU or sparse (CSC) on GPU as torch.Tensor

    TORCH: since A is large, factor is computed on CPU
    CHOLMOD: cholmod is CPU only

    device: location of L
    dense: True if L is returned as dense else False
    """

    if device == "cuda":
        assert not dense

    if log_time:
        t0 = time.time()

    if routine == CholeskyFactorRoutine.TORCH:  # returns on same device as argument
        A = A.to(device).to_dense()
        print(f"Cholesky factored in %f3.2 s", time.time() - t0)
        factor = torch.linalg.cholesky(A)
    if routine == CholeskyFactorRoutine.CHOLMOD:  # returns as CPU only
        # Ensure A is coalesced and convert it to CSR to obtain proper indices for scipy CSC conversion
        A = A.coalesce().to_sparse_csr()
        indptr = A.crow_indices().cpu().numpy()
        indices = A.col_indices().cpu().numpy()
        data = A.values().cpu().numpy()
        n = A.size(0)
        spA = sp.csc_matrix((data, indices, indptr), shape=(n, n))
        factor = cholesky(spA)
        print(f"Cholesky factored in %f3.2 s", time.time() - t0)
        return factor


class CholeskySolveRoutine(Enum):
    TORCH = 0
    CHOLMOD = 0
    MAGMA_DENSE = 2
    MAGMA_SPARSE = 3


if __name__ == "__main__":
    verts = torch.tensor([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]])
    faces = torch.tensor([[0, 1, 2]])
    L = get_combinatorial_laplacian(Meshes(verts[None], faces[None])).to_dense()

    truth = torch.tensor([[2, -1, -1], [-1, 2, -1], [-1, -1, 2.0]])
    assert torch.equal(L, truth)
