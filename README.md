# Controllable Neural Style Transfer for Dynamic Meshes

Here's an implementation of [Controllable Neural Style Transfer for Dynamic Meshes](https://studios.disneyresearch.com/2024/07/28/controllable-neural-style-transfer-for-dynamic-meshes/) by Haetinger et al. Note that this only supports static meshes.

TL;DR—this jitters 3D vertices to minimize the distance between features of a reference style image and ones extracted from images produced by a differentiable renderer.

## Usage
1. Clone repo and place an .obj file to ```data/``` and a style image in ```data/styles``` (anywhere really is fine)
2. Change corresponding paths in the `main` of `model/transfer.py` and run:
```bash
python model/transfer.py
```
3. (Optional) finetune the learning rate, number of iterations, smoothness factors, and masking ratios as needed. The original paper offers starting points for these.

## Implementation Notes
- I used a compute cluster with 1x A100 40Gb with 200Gb RAM. The original paper used an RTX 3090 with an unknown amount of RAM.
- ```spot_280k.obj``` with the given parameters takes ~4 minutes to optimize, but takes the authors 2, so hyperparameter selection likely explains this. 
- Poisson sampling in 3D is not trivial when trying to sample along the surface of a sphere. Instead, since the elevation range is small, directly poisson sampling azimuth and elevation approximates blue noise on a sphere's surface.
- The Cholesky decomposition of the Laplace-Beltrami of the mesh runs on CPU since it is too large for GPU nor are sparse Cholesky operations supported by ```pytorch```. Instead, a custom backwards operation is used in ```model/cholesky.py``` leveraging ```scikit-sparse```