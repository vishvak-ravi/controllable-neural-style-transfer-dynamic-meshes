import unittest
from model.style_utils import get_combinatorial_laplacian, LaplacianRoutine, nearest_neighbor_replacement, hard_nearest_neighbor_replacement, soft_nearest_neighbor
from model.rendering_utils import vertex_preprocess_from_mesh_path

import torch
from pytorch3d.structures import Meshes
import matplotlib.pyplot as plt

torch.set_grad_enabled(False)


class TestLaplaceCreation(unittest.TestCase):
    def test_creation_with_quad_triangle(self):
        verts = torch.tensor([[0.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        faces = torch.tensor([[0, 1, 2], [0, 2, 3]])
        meshes = Meshes([verts], [faces])
        laplacian = get_combinatorial_laplacian(meshes, LaplacianRoutine.CUSTOM, log_time=True).to_dense()
        true_laplacian = torch.Tensor(
            [[3, -1, -1, -1], [-1, 2, -1, 0], [-1, -1, 3, -1], [-1, 0, -1, 2]]
            )
        self.assertTrue(true_laplacian.equal(laplacian))

class TestNearestNeighbor(unittest.TestCase):
    def test_high_dimensional(self):
        ref_feats = torch.randn((1, 5, 2, 2)) * 100
        extracted_feats = ref_feats + torch.randn((2, 5, 2, 2)) * 0.0001
        replaced_feats = hard_nearest_neighbor_replacement(ref_feats, extracted_feats)
        soft_feats = soft_nearest_neighbor(ref_feats, extracted_feats, tau=1e8)
        custom_feats = nearest_neighbor_replacement(ref_feats, extracted_feats, tau=1e8)
        self.assertTrue(torch.allclose(ref_feats, replaced_feats))
        self.assertTrue(torch.allclose(ref_feats, soft_feats))
        
        pass

    def test_2d_visual(self):
        # Generate features
        ref_feats = torch.randn((1, 2, 4, 4)) * 100
        extracted_feats = ref_feats + torch.randn((2, 2, 4, 4))
        replaced_feats = nearest_neighbor_replacement(ref_feats, extracted_feats)

        # Flatten spatial dims so each feature is a point in R^2.
        # For ref and replaced feats we have one batch.
        ref_points = ref_feats[0].reshape(2, -1).t().numpy()         # (16, 2)
        replaced_points = replaced_feats[0].reshape(2, -1).t().numpy()   # (16, 2)

        # Start plot
        plt.figure(figsize=(8, 6))
        # Plot the reference/replaced features with a red marker.
        plt.scatter(ref_points[:, 0], ref_points[:, 1],
                marker='^', color='red', label='Reference/Replaced')

        # Define markers and colors for the two extracted batches.
        markers = ['o', 's']
        colors = ['blue', 'green']

        # For each extracted batch, plot the features and draw arrows to the replaced features.
        for b in range(extracted_feats.shape[0]):
            ext_points      = extracted_feats[b].reshape(2, -1).t().numpy()
            repl_points_b   = replaced_feats[b].reshape(2, -1).t().numpy()   # <- use matching batch

            plt.scatter(ext_points[:,0], ext_points[:,1],
                        marker=markers[b], color=colors[b],
                        label=f'Extracted Batch {b}')

            for i in range(ext_points.shape[0]):
                start, end = ext_points[i], repl_points_b[i]
                plt.arrow(start[0], start[1],
                        end[0]-start[0], end[1]-start[1],
                        head_width=0.5, length_includes_head=True,
                        color='gray', alpha=0.5)

        plt.title('Nearest Neighbor Replacement Visualization (2D Features)')
        plt.xlabel('Feature Dimension 0')
        plt.ylabel('Feature Dimension 1')
        plt.legend()
        plt.grid(True)
        plt.savefig("2d_nn.png")
        plt.close()
        # TODO: implement test for nearest neighbor in 2d visually
        pass

class TestBackwardsCholeskySolve(unittest.TestCase):
    def test_solve(self):
        # TODO: implement test for backwards cholesky solve
        pass

if __name__ == '__main__':
    unittest.main()