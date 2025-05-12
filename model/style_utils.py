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
from pytorch3d.structures import Meshes

VGG_INPUT_SHAPE = (224, 224)
style_transform = Compose([ToTensor(), CenterCrop(VGG_INPUT_SHAPE)])
render_transform = Compose([ConvertImageDtype(torch.float32), Resize(VGG_INPUT_SHAPE)])


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


def nearest_neighbor_replacement(
    style_features: torch.Tensor,
    content_features: torch.Tensor,
    tau: float = 10.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Soft-nearest-neighbor replacement that preserves gradients.

    Args
    ----
    style_features   : (C, Hs, Ws) or (1, C, Hs, Ws) tensor
    content_features : (N, C, H, W) tensor
    tau              : temperature; larger → harder assignment
    eps              : numerical stability

    Returns
    -------
    Tensor of shape (N, C, H, W)
    """
    # Flatten spatial dims
    if style_features.dim() == 4:  # (1,C,Hs,Ws) → (C,Hs,Ws)
        style_features = style_features.squeeze(0)
    N, C, H, W = content_features.shape
    P = H * W  # #content patches
    cf = content_features.view(N, C, P)  # (N, C, P)
    sf = style_features.view(C, -1)  # (C, Ps)

    # Cosine similarity
    cf_norm = cf / (cf.norm(dim=1, keepdim=True) + eps)
    sf_norm = sf / (sf.norm(dim=0, keepdim=True) + eps)
    sims = torch.bmm(
        cf_norm.transpose(1, 2), sf_norm.unsqueeze(0).expand(N, -1, -1)  # (N, P, C)
    )  # (N, C, Ps) → (N, P, Ps)

    # Soft assignment over style patches
    weights = torch.softmax(sims * tau, dim=2)  # (N, P, Ps)

    # Weighted sum of style patches
    sf_T = sf.t().unsqueeze(0).expand(N, -1, -1)  # (N, Ps, C)
    recon = torch.bmm(weights, sf_T)  # (N, P, C)

    return recon.permute(0, 2, 1).contiguous().view(N, C, H, W)


def get_combinatorial_laplacian(mesh: Meshes) -> torch.Tensor:
    # packed verts/faces
    verts = mesh.verts_packed()  # (V, 3)
    faces = mesh.faces_packed()  # (F, 3)
    V = verts.size(0)

    # build undirected edge list
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

    # assemble sparse Laplacian
    idx = torch.cat(
        [edges, torch.stack([torch.arange(V, device=verts.device)] * 2, dim=1)], dim=0
    ).t()  # (2, E + V)
    vals = torch.cat([off_vals, deg], dim=0)  # (E + V,)

    L = torch.sparse_coo_tensor(idx, vals, (V, V))
    return L


if __name__ == "__main__":
    verts = torch.tensor([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]])
    faces = torch.tensor([[0, 1, 2]])
    L = get_combinatorial_laplacian(Meshes(verts[None], faces[None])).to_dense()

    truth = torch.tensor([[2, -1, -1], [-1, 2, -1], [-1, -1, 2.0]])
    assert torch.equal(L, truth)
