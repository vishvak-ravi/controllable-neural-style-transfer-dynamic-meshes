import PIL.Image
import torch, torchvision
from torch.optim.adamw import AdamW
from torch.nn import MSELoss
from torch.utils.tensorboard import SummaryWriter
from pytorch3d.structures import Meshes
from pytorch3d.io.obj_io import load_objs_as_meshes
from cholesky import cholmod_solve

from style_utils import (
    VGGStyleExtractor,
    style_transform,
    render_transform,
    nearest_neighbor_replacement,
    get_combinatorial_laplacian,
    cholesky_factor,
    CholeskySolveRoutine,
    CholeskyFactorRoutine,
)
from rendering_utils import (
    render_mono_texture_from_meshes,
    render_in_pose,
    vertex_preprocess_from_mesh_path,
)
import PIL

N_ITERS = 100
LAMBDAS = torch.tensor([20, 5, 0.5])
MASK_RATIOS = torch.tensor([0.2, 0.1, 0])
LR = torch.tensor([2.0e-3, 1.0e-3, 0.5e-3])

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
        # vert_wc_translation = torch.Tensor([0.0, -0.25, 0.0])

        orig_mesh, verts = vertex_preprocess_from_mesh_path(input_mesh_path)
        print("getting laplacian")
        laplace_beltrami = get_combinatorial_laplacian(orig_mesh)

        # assume: verts (V,3), laplace_beltrami (V,V sparse), style_extractor, renderer cfg set

        V = verts.size(0)
        x_hat = verts.clone()
        # x_prev = x_hat.clone() # ignore for static meshes

    for lam, mask_ratio, lr in zip(LAMBDAS, MASK_RATIOS, LR):
        with torch.no_grad():
            mask = (torch.rand(V, device=device) < mask_ratio).to(device)
            print("computed mask")
            A = laplace_beltrami.clone()
            A.diagonal().add_(lam)  # still sparse
            sparsity = (A == 0).sum().item() / A.numel()
            print(f"sparsity: ", sparsity)
            print("starting cholesky")
            L = cholesky_factor(A, CholeskyFactorRoutine.TORCH, log_time=True, device="cpu", dense=False)
            print("Finished cholesky")
            x_star = A @ x_hat
            print("Got laplacian parameterization")
            del A
        x_star.requires_grad_(True)
        opt = AdamW([x_star], lr=lr)
        for i in range(N_ITERS):
            opt.zero_grad()
            x_hat = cholmod_solve(x_star L)
            #x_hat = torch.cholesky_solve(x_star, L)
            print("solved cholesky")

            imgs: torch.Tensor = render_mono_texture_from_meshes(
                Meshes(
                    [x_hat for _ in range(batch_size)],
                    faces=[orig_mesh.faces_list()[0] for _ in range(batch_size)],
                ),
                batch_size=batch_size,
                poisson_radius=0.25,
                save_name=f"data/renders/{lam}_{i}",
            )
            print("rendered!")

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
    obj = "data/merlion_200k.obj"
    output_mesh = transfer_style(style_img, obj)
