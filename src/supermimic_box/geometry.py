"""Analytic geometry of the box objects: signed distance, corners and lowest point.

Everything is in the object's body frame, fixed-shape and differentiable in the query points.
"""

from __future__ import annotations

import torch

CORNERS = tuple((x, y, z) for x in (1.0, -1.0) for y in (1.0, -1.0) for z in (1.0, -1.0))


class ObjectGeometry:
    def __init__(self, half: torch.Tensor):
        """``half`` (objects, 3): half extents of every object."""
        self.half = half
        self.signs = torch.tensor(CORNERS, device=half.device)

    def surface(self, obj, points):
        """Signed distance (negative inside) and unit direction away from the surface at ``points``
        (..., K, 3), given in the frame of the objects ``obj`` (...,). Returns (..., K, 4)."""
        half = self.half[obj].unsqueeze(-2)
        excess = points.abs() - half
        outside = excess.clamp_min(0)
        outside_distance = outside.square().sum(-1, keepdim=True).add(1e-12).sqrt()
        deepest, axis = excess.max(-1, keepdim=True)
        is_outside = deepest > 0
        distance = torch.where(is_outside, outside_distance, deepest)
        inward = torch.zeros_like(points).scatter(-1, axis, 1.0)
        direction = torch.where(is_outside, outside / outside_distance, inward) * torch.where(points < 0, -1.0, 1.0)
        return torch.cat((distance, direction), -1)

    def corners(self, obj):
        """(..., 8, 3) corners of the objects ``obj`` (...,) in their own frame."""
        return self.signs * self.half[obj].unsqueeze(-2)

    def bottom(self, obj, position, rotation):
        """Height of the lowest point of the objects ``obj`` (...,) at the given poses."""
        return position[..., 2] - (rotation[..., 2, :].abs() * self.half[obj]).sum(-1)
