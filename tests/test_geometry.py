import torch

from supermimic_box.geometry import ObjectGeometry


def test_box_distance_direction_and_bottom():
    geometry = ObjectGeometry(torch.tensor([[0.1, 0.2, 0.3], [0.5, 0.5, 0.5]]))
    obj = torch.tensor([0, 1])
    points = torch.tensor([[[0.3, 0.0, 0.0], [0.0, 0.0, 0.0], [0.2, 0.3, 0.3]],
                           [[0.0, 0.0, -0.9], [0.4, 0.0, 0.0], [0.0, 0.0, 0.0]]])  # fmt: skip
    field = geometry.surface(obj, points)
    assert torch.allclose(field[0, :, 0], torch.tensor([0.2, -0.1, 2**0.5 * 0.1]), atol=1e-6)
    assert torch.allclose(field[0, 0, 1:], torch.tensor([1.0, 0.0, 0.0])) and torch.allclose(field[1, 0], torch.tensor([0.4, 0.0, 0.0, -1.0]))
    assert torch.allclose(field[1, 1], torch.tensor([-0.1, 1.0, 0.0, 0.0]))
    assert torch.allclose(field[..., 1:].norm(dim=-1), torch.ones(2, 3), atol=1e-5)
    # The direction is the gradient of the distance.
    probe = points.clone().requires_grad_(True)
    geometry.surface(obj, probe)[..., 0].sum().backward()
    assert torch.allclose(probe.grad[0, [0, 2]], field[0, [0, 2], 1:], atol=1e-5)
    assert geometry.corners(obj).shape == (2, 8, 3) and torch.allclose(geometry.corners(obj)[0].abs().amax(0), geometry.half[0])
    upright, on_side = torch.eye(3), torch.tensor([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]])
    bottom = geometry.bottom(torch.tensor([0, 0]), torch.tensor([[0.0, 0, 1], [0, 0, 1]]), torch.stack((upright, on_side)))
    assert torch.allclose(bottom, torch.tensor([0.7, 0.8]))
