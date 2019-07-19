import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Threshold


class GProjection(nn.Module):
    """
    Graph Projection layer, which pool 2D features to mesh

    The layer projects a vertex of the mesh to the 2D image and use
    bi-linear interpolation to get the corresponding feature.
    """

    def __init__(self, mesh_pos, camera_f, camera_c, bound=0):
        super(GProjection, self).__init__()
        self.mesh_pos, self.camera_f, self.camera_c = mesh_pos, camera_f, camera_c
        self.threshold = None
        self.bound = 0
        if self.bound != 0:
            self.threshold = Threshold(bound, bound)

    def bound_val(self, x):
        """
        given x, return min(threshold, x), in case threshold is not None
        """
        if self.bound < 0:
            return -self.threshold(-x)
        elif self.bound > 0:
            return self.threshold(x)
        return x

    def forward(self, img_features, inputs):
        # map to [-1, 1]
        # not sure why they render to negative x
        positions = inputs + torch.tensor(self.mesh_pos, device=inputs.device, dtype=torch.float)
        w = (-self.camera_f[0] * (positions[:, :, 0] / self.bound_val(positions[:, :, 2]))) / self.camera_c[0]
        h = (self.camera_f[1] * (positions[:, :, 1] / self.bound_val(positions[:, :, 2]))) / self.camera_c[1]

        # clamp to [-1, 1]
        w = torch.clamp(w, min=-1, max=1)
        h = torch.clamp(h, min=-1, max=1)

        feats = [inputs]
        for i in range(4):
            feats.append(self.project(img_features[i], torch.stack([w, h], dim=-1)))

        output = torch.cat(feats, 2)

        return output

    def project(self, img_feat, sample_points):
        """
        :param img_feat: [batch_size x channel x h x w]
        :param sample_points: [batch_size x num_points x 2], in range [-1, 1]
        :return: [batch_size x num_points x feat_dim]
        """
        output = F.grid_sample(img_feat, sample_points.unsqueeze(1))
        output = torch.transpose(output.squeeze(2), 1, 2)
        return output
