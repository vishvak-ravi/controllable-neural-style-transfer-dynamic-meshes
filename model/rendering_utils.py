import os
import torch
from pytorch3d.io.obj_io import load_objs_as_meshes
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    Materials,
    OrthographicCameras,
    PerspectiveCameras,
    RasterizationSettings,
    BlendParams,
    HardPhongShader,
    MeshRenderer,
    MeshRasterizer,
    TexturesVertex,
    look_at_view_transform,
)
from pytorch3d.renderer.lighting import PointLights
from PIL import Image

from scipy.stats import qmc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DISTANCE_TO_OBJ = 10.0


def sample_camera_params(poisson_r: float, n: int, visualize: bool = False):
    with torch.no_grad():
        # 1-D engine kept; just convert to radians before trig
        eng = qmc.PoissonDisk(d=2, radius=poisson_r, hypersphere="volume")
        uv = torch.tensor(eng.random(n), dtype=torch.float32, device=device)  # u,v ∈ (0,1)
        azim_deg = (uv[:, 0] - 0.5) * 360  # (-180°,180°)
        elev_deg = (uv[:, 1] - 0.5) * 30  # (-10°, 10°)

        if visualize:
            import plotly.graph_objects as go

            azim = torch.deg2rad(azim_deg)
            elev = torch.deg2rad(elev_deg)
            x = DISTANCE_TO_OBJ * torch.cos(elev) * torch.sin(azim)
            y = DISTANCE_TO_OBJ * torch.sin(elev)
            z = DISTANCE_TO_OBJ * torch.cos(elev) * torch.cos(azim)
            fig = go.Figure(go.Scatter3d(x=x, y=y, z=z, mode="markers"))
            fig.update_layout(
                scene=dict(
                    aspectmode="cube",
                    xaxis=dict(range=[-DISTANCE_TO_OBJ, DISTANCE_TO_OBJ]),
                    yaxis=dict(range=[-DISTANCE_TO_OBJ, DISTANCE_TO_OBJ]),
                    zaxis=dict(range=[-DISTANCE_TO_OBJ, DISTANCE_TO_OBJ]),
                )
            )
            fig.show()

        R, T = look_at_view_transform(DISTANCE_TO_OBJ, elev_deg, azim_deg, device=device)
        return R, T


def postprocess_pytorch3d_image(
    image_tensor, tone_map=True, gamma=2.2, save_to_disk=False
):
    """
    Args:
        image_tensor: (H, W, 3) or (N, H, W, 3) float tensor, values may be outside [0,1]
        tone_map: apply Reinhard tone mapping
        gamma: gamma correction value

    Returns:
        uint8 image tensor in [0, 255]
    """
    if tone_map:
        image_tensor = image_tensor / (image_tensor + 1.0)
    image_tensor = image_tensor.clamp(0.0, 1.0)
    image_tensor = image_tensor.pow(1.0 / gamma)
    return image_tensor


def render_mono_texture_from_meshes(
    meshes: Meshes,
    poisson_radius: float,
    color: torch.Tensor = torch.tensor([36, 39, 224]) / 255,
    batch_size: int = 1,
    save_name: str = None,
):
    # create mono texture + material properties
    verts = meshes.verts_packed()  # (V, 3)
    merlion_color = (
        torch.ones((batch_size, verts.shape[0] // batch_size, 3)) * color
    ).to(device)
    meshes.textures = TexturesVertex(verts_features=merlion_color)
    materials = Materials(
        device=device,
        ambient_color=[[0.75, 0.75, 0.75]],  # boost ambient
        diffuse_color=[[0.8, 0.2, 0.2]],
        specular_color=[[1.0, 1.0, 1.0]],  # brighter highlights
        shininess=6.0,
    )

    # sample camera params + lighting
    R, T = sample_camera_params(poisson_radius, batch_size)

    cameras = OrthographicCameras(device=device, R=R, T=T)
    camera_pos = cameras.get_camera_center()
    lights = PointLights(
        device=device,
        location=camera_pos,
        ambient_color=[[0.3, 0.3, 0.3]],
        diffuse_color=[[1.0, 1.0, 1.0]],
        specular_color=[[1.0, 1.0, 1.0]],
    )

    # render setup
    raster_settings = RasterizationSettings(
        image_size=700, blur_radius=0.0, faces_per_pixel=25
    )
    blend_params = BlendParams(background_color=(0.0, 0.0, 0.0))
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=HardPhongShader(
            device=device,
            cameras=cameras,
            lights=lights,
            materials=materials,
            blend_params=blend_params,
        ),
    )

    # render + show
    rgb = renderer(meshes).squeeze()[..., :3]
    imgs = postprocess_pytorch3d_image(rgb)

    if save_name is not None:
        for i, img in enumerate(imgs):
            path = f"{save_name}_{i}.png"
            print("Saving to:", os.path.abspath(path))
            Image.fromarray((img * 255).to(torch.uint8).cpu().numpy()).save(path)

    return imgs


def render_in_pose(
    meshes: Meshes,
    color: torch.Tensor,
    save_name: str = None,
):
    # create mono texture + material properties
    verts = meshes.verts_packed()  # (V, 3)
    merlion_color = (torch.ones((1, verts.shape[0], 3)) * color).to(device)
    meshes.textures = TexturesVertex(verts_features=merlion_color)
    materials = Materials(
        device=device,
        ambient_color=[[0.75, 0.75, 0.75]],  # boost ambient
        diffuse_color=[[0.8, 0.2, 0.2]],
        specular_color=[[1.0, 1.0, 1.0]],  # brighter highlights
        shininess=6.0,
    )

    # sample camera params + lighting
    light_pos1 = [-1.0, 0.5, 1.0]
    light_pos2 = [1.0, 0.5, -1.0]
    lights = PointLights(
        device=device,
        location=[light_pos1, light_pos2],
        ambient_color=[[0.3, 0.3, 0.3], [0.3, 0.3, 0.3]],
        diffuse_color=[[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]],
        specular_color=[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
    )
    R, T = look_at_view_transform(3.5, 15, 180)
    cameras = PerspectiveCameras(device=device, R=R, T=T)

    # render setup
    raster_settings = RasterizationSettings(
        image_size=512, blur_radius=0.0, faces_per_pixel=50
    )
    blend_params = BlendParams(background_color=(0.0, 0.0, 0.0))
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=HardPhongShader(
            device=device,
            cameras=cameras,
            lights=lights,
            materials=materials,
            blend_params=blend_params,
        ),
    )

    # render + show
    rgb = renderer(meshes).squeeze()[..., :3]
    img = postprocess_pytorch3d_image(rgb)

    Image.fromarray((img * 255).to(torch.uint8).cpu().numpy()).save(
        f"{save_name}.png"
    )

    return img


if __name__ == "__main__":
    color = torch.tensor([1, 56, 37]) / 255

    input_mesh_path = "./data/merlion.obj"
    batch_size = 1

    orig_mesh = load_objs_as_meshes(
        [input_mesh_path], device=device, load_textures=False
    )
    verts = orig_mesh.verts_packed()
    center = verts.mean(0)
    verts = verts - center
    scale = 2.0 / (verts.abs().max())
    verts = verts * scale
    src_meshes = Meshes(
        verts=[verts for _ in range(batch_size)],
        faces=[orig_mesh.faces_list()[0] for _ in range(batch_size)],
    )

    render_in_pose(src_meshes, color=color, save_name="pose")


def vertex_preprocess_from_mesh_path(
    input_mesh_path: str, translation: torch.Tensor = None
):
    orig_mesh = load_objs_as_meshes(
        [input_mesh_path], device=device, load_textures=False
    ).to(device)
    verts = orig_mesh.verts_packed()
    center = verts.median(0).values
    verts = verts - center
    scale = 2.0 / (verts.abs().max())
    verts = verts * scale
    if translation is not None:
        assert translation.shape[-1] == 3
        translation = translation.reshape(1, 3)
        verts = verts + translation
    return orig_mesh, verts
