import os
import torch
from pytorch3d.io.obj_io import load_objs_as_meshes
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    Materials,
    FoVOrthographicCameras,
    PerspectiveCameras,
    RasterizationSettings,
    BlendParams,
    SoftPhongShader,
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
DISTANCE_TO_OBJ = 11.0  # ignored for orthographic rendering


def sample_camera_params(poisson_r: float, n: int, visualize: bool = False):
    """
    Sample camera parameters using Poisson disk sampling.

    This function generates camera positions distributed on a sphere's surface using Poisson disk sampling
    for more uniform coverage. The cameras are positioned looking at the origin.

    Args:
        poisson_r (float): The minimum distance between samples in the Poisson disk sampling
        n (int): Number of camera positions to generate
        visualize (bool, optional): If True, displays a 3D visualization of camera positions. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - R (torch.Tensor): Rotation matrices for each camera position, shape (n, 3, 3)
            - T (torch.Tensor): Translation vectors for each camera position, shape (n, 3)

    Notes:
        - Camera positions are sampled on a partial sphere with:
            - Azimuth angle range: [-180°, 180°]
            - Elevation angle range: [-35°, 35°]
        - All cameras are positioned at a fixed distance (DISTANCE_TO_OBJ) from the origin
        - Not true Poisson sampling on the sphere surface—samples angles instead but ~Poisson for small elevation
    """
    with torch.no_grad():
        # 1-D engine kept; just convert to radians before trig
        eng = qmc.PoissonDisk(d=2, radius=poisson_r, hypersphere="volume")
        uv = torch.tensor(
            eng.random(n), dtype=torch.float32, device=device
        )  # u,v ∈ (0,1)
        azim_deg = (uv[:, 0] - 0.5) * 360  # (-180°,180°)
        elev_deg = (uv[:, 1] - 0.5) * 70  # (-35°, 35°)

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

        R, T = look_at_view_transform(
            DISTANCE_TO_OBJ, elev_deg, azim_deg, device=device
        )
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
    blur_radius: float = 1e-3,
    resolution: int = 1000,
    save_name: str = None,
):
    """
    Renders a mesh with a single color (monotexture) from different camera angles.

    Args:
        meshes (Meshes): PyTorch3D Meshes object containing the 3D mesh to render
        poisson_radius (float): Radius for Poisson disk sampling to generate camera positions
        color (torch.Tensor, optional): RGB color for the mesh texture. Defaults to blue color [36, 39, 224]/255
        batch_size (int, optional): Number of different views to render. Defaults to 1
        blur_radius (float, optional): Blur radius for rasterization. Defaults to 1e-3
        resolution (int, optional): Output image resolution. Defaults to 1000
        save_name (str, optional): If provided, saves rendered images with this prefix. Defaults to None

    Returns:
        torch.Tensor: Tensor of rendered images with shape (N, H, W, 3) where:
            - N is the batch size (number of views)
            - H, W are the height and width of the rendered image
            - 3 represents RGB channels

    The function:
    - Applies a monotexture with specified color to the mesh
    - Sets up material properties for Phong shading
    - Samples camera positions using Poisson disk sampling
    - Uses orthographic projection with adjustable view volume
    - Renders the mesh using PyTorch3D's MeshRenderer with soft Phong shading
    - Optionally saves the rendered images to disk if save_name is provided

    Notes:
        The mesh is expected to be scaled to [-1, 1] in X-Y coordinates for proper rendering
        Images are saved as PNG files with names {save_name}_{index}.png if save_name is provided
    """
    # create mono texture + material properties
    verts = meshes.verts_packed()
    merlion_color = (
        torch.ones((batch_size, verts.shape[0] // batch_size, 3)) * color
    ).to(device)
    meshes.textures = TexturesVertex(verts_features=merlion_color)
    materials = Materials(
        device=device,
        ambient_color=[[0.7, 0.7, 0.7]],
        diffuse_color=[[0.9, 0.9, 0.9]],
        specular_color=[[0.8, 0.8, 0.8]],  # important for some patterns
        shininess=2.0,
    )

    # sample camera params + lighting
    R, T = sample_camera_params(poisson_radius, batch_size)
    cameras = FoVOrthographicCameras(
        R=R,
        T=T,
        device=device,
        znear=0.01,
        zfar=10.0,
        min_x=-1.0,
        max_x=1.0,
        min_y=-1.0,
        max_y=1.0,
        scale_xyz=((0.5, 0.5, 0.5),),  # raise to zoom out, lower to zoom in
    )

    camera_pos = cameras.get_camera_center()
    lights = PointLights(
        device=device,
        location=camera_pos,
        ambient_color=((0.8, 0.8, 0.8),),
        diffuse_color=((0.5, 0.5, 0.5),),
        specular_color=((0.6, 0.6, 0.6),),
    )

    # render setup
    raster_settings = RasterizationSettings(
        image_size=resolution, blur_radius=blur_radius, faces_per_pixel=15
    )
    blend_params = BlendParams(  # this might need some work...
        sigma=1e-2, gamma=4e-2, background_color=(0, 0, 0)
    )
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(  # maybe another shader?
            device=device,
            cameras=cameras,
            lights=lights,
            materials=materials,
            blend_params=blend_params,
        ),
    )

    # render + show
    rgb = renderer(meshes)
    if rgb.dim() == 4:
        rgb = rgb.squeeze(0)
    rgb = rgb[..., :3]

    imgs = postprocess_pytorch3d_image(rgb)

    # Ensure imgs is always (N, H, W, 3) for consistent iteration
    if imgs.dim() == 3:
        imgs = imgs.unsqueeze(0)

    if save_name is not None:
        for i, img in enumerate(imgs):
            path = f"{save_name}_{i}.png"
            # print("Saving to:", os.path.abspath(path))
            Image.fromarray((img * 255).to(torch.uint8).cpu().numpy()).save(path)

    return imgs


def vertex_preprocess_from_mesh_path(input_mesh_path: str):
    """
    Preprocesses vertices from a given mesh file by centering and scaling.

    This function loads a 3D mesh from a file, centers its vertices around the origin
    by subtracting the median position, and scales the vertices to fit within a
    normalized range.

    Args:
        input_mesh_path (str): Path to the input mesh file (.obj format)

    Returns:
        tuple: A tuple containing:
            - orig_mesh (pytorch3d.structures.Meshes): The loaded mesh object
            - verts (torch.Tensor): Preprocessed vertices after centering and scaling

    Note:
        The vertices are scaled such that the maximum absolute coordinate value is 2.0.
    """
    orig_mesh = load_objs_as_meshes(
        [input_mesh_path], device=device, load_textures=False
    ).to(device)
    verts = orig_mesh.verts_packed()
    center = verts.median(0).values
    verts = verts - center
    scale = 2.0 / (verts.abs().max())
    verts = verts * scale
    return orig_mesh, verts


if __name__ == "__main__":
    orig_mesh, verts = vertex_preprocess_from_mesh_path("data/spot_280k.obj")
    mesh = Meshes([verts], [orig_mesh.faces_list()[0]])
    render_mono_texture_from_meshes(mesh, 0.20, save_name="data/renders/test.png")
