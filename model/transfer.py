import PIL.Image
import torch, torchvision
from torch.optim.adamw import AdamW
from torch.nn import MSELoss
from torch.utils.tensorboard import SummaryWriter
from pytorch3d.structures import Meshes
from pytorch3d.io.obj_io import load_objs_as_meshes, save_obj
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
    LaplacianRoutine,
)
from rendering_utils import (
    render_mono_texture_from_meshes,
    vertex_preprocess_from_mesh_path,
)
import PIL

N_ITERS = torch.tensor([30, 30, 50])
LAMBDAS = torch.tensor([20, 5, 1])
MASK_RATIOS = torch.tensor([0.2, 0.1, 0])
LR = torch.tensor([8.0e-3, 2.0e-3, 0.5e-3])
BLUR_RADII = torch.tensor([0, 0, 0])
IMAGE_SCALES = torch.tensor([0.4, 0.8, 1.5])
RESOLUTION = 1000

device = "cuda" if torch.cuda.is_available() else "cpu"


def transfer_style(style_reference_path, input_mesh_path, cfg: dict = {}):

    batch_size = cfg.get("batch_size", 3)
    num_rots = cfg.get("num_rotations", 8)
    writer = SummaryWriter()
    global_step = 0

    # load style extractor
    ref_style = torchvision.utils.Image.open(style_reference_path)
    all_rotated_styles = get_rotated_style_tensors(ref_style, num_rots).to(device)

    mse = MSELoss().to(device)
    style_extractor = VGGStyleExtractor().to(device)
    with torch.no_grad():
        # extract and normalize reference style features
        unnormalized_ref_style_features = style_extractor(
            all_rotated_styles
        )  # num_rots x 2688 x H/4, W/4
        ref_style_features = unnormalized_ref_style_features - torch.mean(
            unnormalized_ref_style_features, dim=(2, 3), keepdim=True
        )
        
        # get mesh and its laplacian
        orig_mesh, verts = vertex_preprocess_from_mesh_path(input_mesh_path)
        print("getting laplacian")
        laplace_beltrami = get_combinatorial_laplacian(
            orig_mesh, LaplacianRoutine.CUSTOM, log_time=True
        )

        V = verts.size(0)
        x_hat = verts.clone()
        # x_prev = x_hat.clone() # ignore for static meshes
    frame_start_time = time.time()
    for lam, mask_ratio, lr, iters, blur_radius, img_scale in zip(
        LAMBDAS, MASK_RATIOS, LR, N_ITERS, BLUR_RADII, IMAGE_SCALES
    ):
        with torch.no_grad():
            mask = (torch.rand(V, device=device) < mask_ratio).to(device) # volume regularization mask
            print("computed mask")

            A = laplace_beltrami.clone() * lam  # compute A = I * lambda + Laplace
            A = A.coalesce()
            diag_mask = A.indices()[0] == A.indices()[1]
            A.values()[diag_mask] += 1

            print("starting cholesky")
            t0 = time.time()
            L = cholesky_factor(
                A,
                CholeskyFactorRoutine.CHOLMOD,
                log_time=True,
                device="cpu",
                dense=False,
            )
            print(f"got lower cholesky in {time.time() - t0} s")
            A = A.to(device)

            x_star = A @ x_hat
            print("Got laplacian parameterization")
            del A
        x_star.requires_grad_(True)
        opt = AdamW([x_star], lr=lr)
        for i in range(iters):
            t_start = time.time()
            opt.zero_grad()
            t0 = time.time()
            print("solving cholesky")
            x_hat = cholmod_solve(x_star, L) # get vertices from implicit paramterization
            x_hat = x_hat.to(device)
            print(f"solved cholesky in {time.time() - t0} s")

            imgs: torch.Tensor = render_mono_texture_from_meshes( # construct and render mesh
                Meshes(
                    [x_hat for _ in range(batch_size)],
                    faces=[orig_mesh.faces_list()[0] for _ in range(batch_size)],
                ),
                batch_size=batch_size,
                poisson_radius=0.20,
                blur_radius=blur_radius,
                resolution=RESOLUTION * img_scale,
                save_name=f"data/renders/{lam}_{i}",
            )
            print("rendered!")

            imgs = imgs.permute(0, 3, 1, 2)
            feats = style_extractor(render_transform(imgs))
            
            # normalize content features with running average
            with torch.no_grad():
                if global_step == 0:
                    mean_content_features = torch.mean(feats, dim=(2, 3), keepdim=True)
                    feature_count = 1
                else:
                    current_mean = torch.mean(feats, dim=(2, 3), keepdim=True)
                    feature_count += 1
                    mean_content_features = (
                        mean_content_features * (feature_count - 1) + current_mean
                    ) / feature_count

            normalized_feats = feats - mean_content_features
            
            # compute MSE between feats and its nearest neighbor from the reference style feats
            repl = nearest_neighbor_replacement(
                ref_style_features, normalized_feats, tau=1e10
            )
            loss = mse(repl, feats)
            loss.backward()

            if global_step % 10 == 0:  # log every 10 iters
                writer.add_scalar("loss", loss.item(), global_step)
                writer.add_scalar("grad_norm", x_star.grad.norm().item(), global_step)

            x_star.grad[mask] = 0 # use the volume regularization mask
            opt.step()
            global_step += 1
            print(f"Step {i} took: {time.time() - t_start}")

        with torch.no_grad():
            x_hat = cholmod_solve(x_star, L)
    print(f"Total time per frame: {time.time() - frame_start_time}")
    writer.close()
    output_mesh = Meshes(
        [x_hat],
        faces=[orig_mesh.faces_list()[0]],
    )
    return output_mesh


if __name__ == "__main__":
    style_img = "data/styles/starry.png"
    obj = "data/spot_210k.obj"
    output_mesh: Meshes = transfer_style(style_img, obj)
    save_obj(
        f"data/mesh_outputs/spot_starry.obj",
        output_mesh.verts_list()[0],
        output_mesh.faces_list()[0],
    )
