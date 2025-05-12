import PIL.Image
import torch, torchvision
from torch.optim.adamw import AdamW
from torch.nn import MSELoss
from torch.utils.tensorboard import SummaryWriter
from pytorch3d.structures import Meshes
from pytorch3d.io.obj_io import load_objs_as_meshes

from style_utils import (
    VGGStyleExtractor,
    style_transform,
    render_transform,
    nearest_neighbor_replacement,
    get_combinatorial_laplacian,
)
from rendering_utils import render_mono_texture_from_meshes, render_in_pose
import PIL

N_ITERS = 100
LAMBDAS = [20, 5, 0.5]
MASK_RATIOS = [0.2, 0.1, 0]
LR = [10.0e-3, 5.0e-3, 1.5e-3]

device = "cuda" if torch.cuda.is_available() else "cpu"


def transfer_style(style_reference_path, input_mesh_path, cfg: dict = {}):

    batch_size = cfg.get("batch_size", 8)
    writer = SummaryWriter()  # ← 1-line init
    global_step = 0

    # load style ref
    ref_style = torchvision.utils.Image.open(style_reference_path)
    ref_style = style_transform(ref_style).unsqueeze(0).to(device)

    mse = MSELoss().to(device)
    style_extractor = VGGStyleExtractor().to(device)
    with torch.no_grad():
        ref_style_features = style_extractor(ref_style)  # 1 x 2688 x H/4, W/4

    # set up batched meshes and normalize
    orig_mesh = load_objs_as_meshes(
        [input_mesh_path], device=device, load_textures=False
    ).to(device)
    verts = orig_mesh.verts_packed()
    center = verts.mean(0)
    verts = verts - center
    scale = 2.0 / (verts.abs().max())
    verts = verts * scale

    # set up optimizer
    verts = verts.requires_grad_(True)
    opt = AdamW([verts])
    laplace_beltrami = get_combinatorial_laplacian(orig_mesh).to(device)

    # assume: verts (V,3), laplace_beltrami (V,V sparse), style_extractor, renderer cfg set

    V = verts.size(0)
    I = torch.eye(V, device=device)
    x_hat = verts.clone()
    x_prev = x_hat.clone()

    for lam, mask_ratio, lr in zip(LAMBDAS, MASK_RATIOS, LR):
        mask = (torch.rand(V, device=device) < mask_ratio).to(device)
        A = (I + lam * laplace_beltrami).to_dense()
        L = torch.linalg.cholesky(A)
        x_star = (A @ x_hat).detach().requires_grad_(True)
        opt = AdamW([x_star], lr=lr)

        for i in range(N_ITERS):
            opt.zero_grad()

            x_hat = torch.cholesky_solve(x_star, L)

            imgs: torch.Tensor = render_mono_texture_from_meshes(
                Meshes(
                    [x_hat for _ in range(batch_size)],
                    faces=[orig_mesh.faces_list()[0] for _ in range(batch_size)],
                ),
                batch_size=batch_size,
                poisson_radius=0.25,
                save_name=f"data/renders/{lam}_{i}",
            )

            imgs = imgs.permute(0, 3, 1, 2)
            feats = style_extractor(render_transform(imgs))
            repl = nearest_neighbor_replacement(ref_style_features, feats)
            loss = mse(repl, feats)
            loss.backward()

            if global_step % 10 == 0:  # ← log every 10 iters
                writer.add_scalar("loss", loss.item(), global_step)
                writer.add_scalar("grad_norm", x_star.grad.norm().item(), global_step)

            x_star.grad[mask] = 0
            opt.step()
            global_step += 1

        x_hat = torch.cholesky_solve(x_star.detach(), L)
        meshes = Meshes(
            [x_hat],
            faces=[orig_mesh.faces_list()[0]],
        )
        render_in_pose(
            meshes,
            color=torch.tensor([1, 56, 37]) / 255,
            save_name=f"data/renders/{lam}.png",
        )
    writer.close()
    return x_hat


if __name__ == "__main__":
    style_img = "data/styles/swirly.jpg"
    obj = "data/merlion.obj"
    output_mesh = transfer_style(style_img, obj)
