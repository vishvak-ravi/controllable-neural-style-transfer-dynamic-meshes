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
style_transform = Compose(
    [Resize(VGG_INPUT_SHAPE[0]), ToTensor(), CenterCrop(VGG_INPUT_SHAPE)]
)
render_transform = Compose([ConvertImageDtype(torch.float32), Resize(VGG_INPUT_SHAPE)])


def get_rotated_style_tensors(
    img: torchvision.utils.Image.Image, num_rots: int
) -> torch.Tensor:
    """
    Generates a tensor containing multiple rotated versions of a given image.

    Args:
        img (torchvision.utils.Image.Image): The input image to be transformed and rotated.
        num_rots (int): The number of rotations to apply. The image will be rotated
                        evenly across 360 degrees.

    Returns:
        torch.Tensor: A tensor containing the rotated versions of the input image.
                      The tensor has shape (num_rots, C, H, W), where C is the
                      number of channels (3 after conversion), and H and W are
                      the height and width of the image.

    Notes:
        - The input image is first transformed using `style_transform`.
        - If the input image has 4 channels, it is converted to 3 channels by
          taking the first 3 channels.
        - The `rotate` function is used to apply the rotations.
    """
    img = style_transform(img)
    if img.shape[0] == 4:  # Check if the image has 4 channels
        img = img[:3]  # Convert to 3 channels by taking the first 3
    rotated_imgs = torch.stack(
        [rotate(img, 360.0 * i / num_rots) for i in range(num_rots)]
    )
    return rotated_imgs


class VGGStyleExtractor(nn.Module):
    """
    A PyTorch module for extracting style features from an input image using a pre-trained VGG16 network.

    This module extracts features from the VGG16 network up to the ReLU activation after the `conv4_3` layer.
    The extracted features are interpolated to a specified output size and normalized by subtracting their mean.

    Attributes:
        features (nn.ModuleList): A list of layers from the VGG16 network up to the ReLU after `conv4_3`.
        output_size (tuple): The target spatial size (height, width) for interpolating the extracted features.

    Methods:
        forward(x):
            Passes the input tensor through the VGG16 layers, extracts features after each ReLU activation,
            interpolates them to the specified output size, and normalizes the features.

    Args:
        output_size (tuple, optional): The target spatial size for interpolating the extracted features.
            Defaults to (56, 56).
    """

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


def hard_nearest_neighbor_replacement(
    ref_feats: torch.Tensor, extracted_feats: torch.Tensor
) -> torch.Tensor:
    """
    Replaces each feature vector in the `extracted_feats` tensor with the closest
    feature vector (based on ℓ2 distance) from the `ref_feats` tensor.

    Args:
        ref_feats (torch.Tensor): A reference tensor of shape (R, C, H, W), where:
            - R: Number of reference feature maps (usually 1).
            - C: Number of channels in the feature maps.
            - H: Height of the feature maps.
            - W: Width of the feature maps.
        extracted_feats (torch.Tensor): A tensor of extracted features of shape
            (B, C, H, W), where:
            - B: Batch size.
            - C: Number of channels in the feature maps.
            - H: Height of the feature maps.
            - W: Width of the feature maps.

    Returns:
        torch.Tensor: A tensor of shape (B, C, H, W) where each feature vector
        in `extracted_feats` is replaced by the closest feature vector from
        `ref_feats` based on ℓ2 distance.

    Notes:
        - This is nondifferentiable used for testing purposes
        - The function computes pairwise ℓ2 distances between all feature vectors
        in `extracted_feats` and `ref_feats`.
        - The closest feature vector from `ref_feats` is selected for each feature
        vector in `extracted_feats`.
        - The output tensor has the same shape as the input `extracted_feats` tensor.
    """
    R, C, H, W = ref_feats.shape  # reference
    B = extracted_feats.shape[0]  # extracted batches

    # (R·H·W, C) and (B·H·W, C)
    ref_flat = ref_feats.permute(0, 2, 3, 1).reshape(-1, C)
    ext_flat = extracted_feats.permute(0, 2, 3, 1).reshape(-1, C)

    # pair-wise distances: (BHW, RHW)  -> argmin over references
    idx = torch.cdist(ext_flat, ref_flat).argmin(dim=1)

    # gather nearest reference vectors
    nearest = ref_flat[idx]  # (BHW, C)

    # reshape back to (B,C,H,W)
    return nearest.reshape(B, H, W, C).permute(0, 3, 1, 2)


def nearest_neighbor_replacement(
    style_features: torch.Tensor,  # (C, Hs, Ws) or (B, C, Hs, Ws)
    content_features: torch.Tensor,  # (N, C, H, W)
    tau: float = 10.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Performs soft-nearest-neighbor feature replacement with stable gradients.

        This function replaces content features with style features by computing
        cosine similarities between the two and applying a soft assignment
        mechanism. It is designed to handle both single and batched style images.

        Args:
            style_features (torch.Tensor): A tensor of style features with shape
                (C, Hs, Ws) for a single style image or (B, C, Hs, Ws) for batched
                style images, where:
                - C: Number of channels.
                - Hs, Ws: Height and width of the style feature map.
                - B: Batch size for batched style images.
            content_features (torch.Tensor): A tensor of content features with shape
                (N, C, H, W), where:
                - N: Batch size of content images.
                - C: Number of channels (must match the channels in style_features).
                - H, W: Height and width of the content feature map.
            tau (float, optional): A temperature parameter controlling the sharpness
                of the softmax distribution. Default is 10.0.
            eps (float, optional): A small value added to the denominator for numerical
                stability during normalization. Default is 1e-8.

        Returns:
            torch.Tensor: A tensor of reconstructed features with shape (N, C, H, W),
            where the content features are replaced by style features based on
            soft-nearest-neighbor matching.

        Raises:
            AssertionError: If the number of channels in content_features does not
            match the number of channels in style_features.

        Notes:
            - The function normalizes both content and style features to compute
              cosine similarities.
            - Softmax is applied along the style feature dimension to compute
              weights for reconstruction.
            - The gradients are stable due to the use of soft assignments.
    """
    # ---------- flatten style patches ----------
    if style_features.dim() == 4:  # batched styles
        B, C, Hs, Ws = style_features.shape
        style_features = style_features.permute(  # (B, C, Hs, Ws)
            1, 0, 2, 3
        ).reshape(  # → (C, B, Hs, Ws)
            C, B * Hs * Ws
        )
    else:  # single style image
        C, Hs, Ws = style_features.shape
        style_features = style_features.reshape(C, Hs * Ws)  # (C, Ps)

    # ---------- flatten content patches ----------
    N, C_c, H, W = content_features.shape
    assert C_c == style_features.size(0), "channel mismatch"
    P = H * W
    cf = content_features.reshape(N, C, P)  # (N, C, P)

    # ---------- cosine similarities ----------
    cf_norm = cf / (cf.norm(dim=1, keepdim=True) + eps)  # (N, C, P)
    sf_norm = style_features / (
        style_features.norm(dim=0, keepdim=True) + eps
    )  # (C, Ps)
    sims = torch.bmm(
        cf_norm.transpose(1, 2), sf_norm.expand(N, -1, -1)  # (N, P, C)
    )  # (N, P, Ps)

    # ---------- soft assignment ----------
    weights = torch.softmax(sims * tau, dim=2)  # (N, P, Ps)

    # ---------- reconstruction ----------
    recon = torch.bmm(
        weights, style_features.t().expand(N, -1, -1)  # (N, P, Ps)
    )  # (N, P, C)
    return recon.permute(0, 2, 1).reshape(N, C, H, W)


def get_combinatorial_laplacian(mesh: Meshes, log_time: bool) -> torch.Tensor:
    """
    Computes the combinatorial Laplacian matrix for a given mesh.

    The combinatorial Laplacian is a sparse matrix representation of the mesh's
    connectivity. It is constructed using the vertices and edges of the mesh,
    where the off-diagonal entries represent the negative weights of edges,
    and the diagonal entries represent the degree of each vertex.

    Args:
        mesh (Meshes): A PyTorch3D `Meshes` object containing the mesh data.
            The mesh should have packed vertices and faces.
        log_time (bool): If True, logs the time taken to compute the Laplacian.

    Returns:
        torch.Tensor: A sparse tensor in COO format representing the
        combinatorial Laplacian matrix of the mesh.

    Notes:
        - The Laplacian is constructed as a sparse tensor in COO format.
        - The function ensures that the edge list is undirected and unique.
        - If `log_time` is enabled, the computation time is printed to the console.
    """
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


def cholesky_factor(
    A: torch.Tensor,
    routine: CholeskyFactorRoutine,
    log_time: bool,
    device: torch.device,
    dense: bool,
):
    """
       Computes the Cholesky factorization of a given sparse matrix `A`.

    This function supports two routines for Cholesky factorization:
    - `TORCH`: Uses PyTorch's built-in Cholesky factorization.
    - `CHOLMOD`: Uses the CHOLMOD library via SciPy for factorization.

    The input matrix `A` must be in COO sparse format. The function assumes that
    `A` is too large to fit onto the GPU, so the factorization is performed on the CPU.
    The resulting Cholesky factor `L` can be returned as either a dense or sparse matrix,
    depending on the `dense` parameter.

    Args:
        A (torch.Tensor): The input sparse matrix in COO format.
        routine (CholeskyFactorRoutine): The routine to use for Cholesky factorization.
            Options are `CholeskyFactorRoutine.TORCH` or `CholeskyFactorRoutine.CHOLMOD`.
        log_time (bool): If `True`, logs the time taken for the factorization.
        device (torch.device): The device where the resulting Cholesky factor `L` should be located.
            If `device` is "cuda", the `dense` parameter must be `False`.
        dense (bool): If `True`, returns the Cholesky factor `L` as a dense matrix.
            If `False`, returns `L` as a sparse matrix.

    Returns:
        torch.Tensor or scipy.sparse.csc_matrix: The Cholesky factor `L`.
            - If `routine` is `TORCH`, returns a PyTorch tensor.
            - If `routine` is `CHOLMOD`, returns a SciPy CSC sparse matrix.

    Raises:
        AssertionError: If `device` is "cuda" and `dense` is `True`.

    Notes:
        - For the `TORCH` routine, the input matrix `A` is converted to a dense format
          before factorization.
        - For the `CHOLMOD` routine, the input matrix `A` is converted to a SciPy CSC
          sparse matrix before factorization.
        - The function assumes that the input matrix `A` is symmetric positive definite.

    Example:
        >>> A = torch.sparse_coo_tensor(indices, values, size)
        >>> L = cholesky_factor(A, CholeskyFactorRoutine.TORCH, log_time=True, device="cpu", dense=True)
    """

    if device == "cuda":
        assert not dense

    if log_time:
        t0 = time.time()

    if routine == CholeskyFactorRoutine.TORCH:  # returns on same device as argument
        A = A.to(device).to_dense()
        print(f"Cholesky factored in {time.time() - t0} s")
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
