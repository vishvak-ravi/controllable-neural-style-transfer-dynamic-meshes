import PIL.Image
import torch, torchvision
from torch.optim.adamw import AdamW
from torch.nn import MSELoss
from torch.utils.tensorboard import SummaryWriter
from pytorch3d.structures import Meshes
from pytorch3d.io.obj_io import load_objs_as_meshes
from cholesky import cholmod_solve
import time

from style_utils import (
    VGGStyleExtractor,
    get_rotated_style_tensors,
    render_transform,
    nearest_neighbor_replacement,
    get_combinatorial_laplacian,
    cholesky_factor,
    CholeskyFactorRoutine,
    LaplacianRoutine
)
from rendering_utils import (
    render_mono_texture_from_meshes,
    vertex_preprocess_from_mesh_path,
)
import PIL

N_ITERS = torch.tensor([50, 50, 25])
LAMBDAS = torch.tensor([20, 5, 0.5])
MASK_RATIOS = torch.tensor([0.2, 0.1, 0])
LR = torch.tensor([8.0e-3, 2.0e-3, 0.5e-3])

device = "cuda" if torch.cuda.is_available() else "cpu"


def transfer_style(style_reference_path, input_mesh_path, cfg: dict = {}):

    batch_size = cfg.get("batch_size", 4)
    num_rots = cfg.get("num_rotations", 4)
    writer = SummaryWriter()  # ← 1-line init
    global_step = 0

    # load style ref
    ref_style = torchvision.utils.Image.open(style_reference_path)
    all_rotated_styles = get_rotated_style_tensors(ref_style, num_rots).to(device)

    mse = MSELoss().to(device)
    style_extractor = VGGStyleExtractor().to(device)
    with torch.no_grad():
        unnormalized_ref_style_features = style_extractor(all_rotated_styles)  # num_rots x 2688 x H/4, W/4
        ref_style_features = unnormalized_ref_style_features - torch.mean(unnormalized_ref_style_features, dim=(2, 3), keepdim=True)
        
        # set up batched meshes and normalize
        # vert_wc_translation = torch.Tensor([0.0, -0.25, 0.0])

        orig_mesh, verts = vertex_preprocess_from_mesh_path(input_mesh_path)
        print("getting laplacian")
        laplace_beltrami = get_combinatorial_laplacian(orig_mesh, LaplacianRoutine.CUSTOM, log_time=True)

        # assume: verts (V,3), laplace_beltrami (V,V sparse), style_extractor, renderer cfg set

        V = verts.size(0)
        x_hat = verts.clone()
        # x_prev = x_hat.clone() # ignore for static meshes

    for lam, mask_ratio, lr, iters in zip(LAMBDAS, MASK_RATIOS, LR, N_ITERS):
        with torch.no_grad():
            mask = (torch.rand(V, device=device) < mask_ratio).to(device)
            print("computed mask")
            
            A = laplace_beltrami.clone() * lam # compute A = I * lambda + Laplace
            A = A.coalesce()
            diag_mask = A.indices()[0] == A.indices()[1]
            A.values()[diag_mask] += 1
            
            print("starting cholesky")
            t0 = time.time()
            L = cholesky_factor(A, CholeskyFactorRoutine.CHOLMOD, log_time=True, device="cpu", dense=False)
            print(f'got lower cholesky in {time.time() - t0} s')
            A = A.to(device)
            
            x_star = A @ x_hat
            print("Got laplacian parameterization")
            del A
        x_star.requires_grad_(True)
        opt = AdamW([x_star], lr=lr)
        for i in range(iters):
            opt.zero_grad()
            t0 = time.time()
            print("solving cholesky")
            x_hat = cholmod_solve(x_star, L)
            x_hat = x_hat.to(device)
            #x_hat = torch.cholesky_solve(x_star, L)
            print(f'solved cholesky in {time.time() - t0} s')

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
            
            with torch.no_grad():
                if global_step == 0:
                    mean_content_features = torch.mean(feats, dim=(2, 3), keepdim=True)
                    feature_count = 1
                else:
                    current_mean = torch.mean(feats, dim=(2, 3), keepdim=True)
                    feature_count += 1
                    mean_content_features = (mean_content_features * (feature_count - 1) + current_mean) / feature_count

            normalized_feats = feats - mean_content_features
            repl = nearest_neighbor_replacement(ref_style_features, normalized_feats, tau=1e10)
            loss = mse(repl, feats)
            loss.backward()

            if global_step % 10 == 0:  # ← log every 10 iters
                writer.add_scalar("loss", loss.item(), global_step)
                writer.add_scalar("grad_norm", x_star.grad.norm().item(), global_step)

            x_star.grad[mask] = 0
            opt.step()
            global_step += 1

        with torch.no_grad():
            x_hat = cholmod_solve(x_star, L)

    writer.close()
    return x_hat


if __name__ == "__main__":
    style_img = "data/styles/triangles.png"
    obj = "data/spot_280k.obj"
    output_mesh = transfer_style(style_img, obj)
