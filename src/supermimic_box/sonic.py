"""The pre-trained SONIC whole-body tracker (NVIDIA GEAR-SONIC, G1 mode) as a frozen base policy.

SONIC tracks a joint-space reference: an encoder turns ten future reference frames into two
quantized motion tokens, and a decoder maps the tokens and a ten-step proprioceptive history to
joint targets around the default pose. It knows nothing about objects; a learned residual on its
action adds the interaction.

The weights are the release checkpoint ported to MuJoCo joint order (``mocke``'s
``pretrained/sonic/last_ported.pt``), and the two input layouts follow that port:

  tokenizer (640): the reference at the current frame and nine more, 0.1 s apart. Joint positions
      of all ten frames, then joint velocities, cut into ten rows of 58 (SONIC's own layout, which
      does not align rows with frames), each row followed by the reference pelvis rotation of that
      frame relative to the robot's pelvis (first two matrix columns).
  proprio (930): ten-step histories, oldest first, of base angular velocity, joint positions
      relative to the default pose, joint velocities, previous actions and projected gravity.

The proprioceptive history is read from the actor's observation history, whose first entries are
the same quantities with the same noise. SONIC was trained with this project's drives except for
the hip pitch joints, which it drives with the larger 7520-22 actuator (``drives: sonic``).
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn

FUTURE_FRAMES, FRAME_SKIP, HISTORY = 10, 5, 10
TOKENS, TOKEN_DIM, LEVELS = 2, 32, 32
ENCODER_HIDDEN, DECODER_HIDDEN = (2048, 1024, 512, 512), (2048, 2048, 1024, 1024, 512, 512)


def mlp(inp: int, hidden, out: int) -> nn.Sequential:
    """Linear layers at even indices, as in the checkpoint."""
    layers, d = [], inp
    for h in hidden:
        layers += [nn.Linear(d, h), nn.SiLU()]
        d = h
    layers.append(nn.Linear(d, out))
    return nn.Sequential(*layers)


def fsq(z):
    """Finite scalar quantization to 32 levels per dimension in [-1, 1) (vector_quantize_pytorch.FSQ),
    with a straight-through gradient."""
    half = (LEVELS - 1) * (1 + 1e-3) / 2
    bounded = torch.tanh(z + math.atanh(0.5 / half)) * half - 0.5
    return (bounded + (bounded.round() - bounded).detach()) / (LEVELS // 2)


class SonicBase(nn.Module):
    """Frozen SONIC. ``forward`` returns its action: joint targets are default + scale * action."""

    def __init__(self, checkpoint: str | Path, motion, joint_names, num_joints: int):
        super().__init__()
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        meta = saved["meta"]
        if tuple(meta["joint_order"]) != tuple(joint_names):
            raise ValueError("The SONIC checkpoint is not in this robot's joint order")
        if (meta["num_future_frames"], meta["history_length"], meta["tokenizer_dim"], meta["proprio_dim"]) != (
                FUTURE_FRAMES, HISTORY, FUTURE_FRAMES * (2 * num_joints + 6), HISTORY * (3 * num_joints + 6)):  # fmt: skip
            raise ValueError("Unexpected SONIC checkpoint layout")
        self.J, self.motion = num_joints, motion
        self.encoder = mlp(meta["tokenizer_dim"], ENCODER_HIDDEN, TOKENS * TOKEN_DIM)
        self.decoder = mlp(TOKENS * TOKEN_DIM + meta["proprio_dim"], DECODER_HIDDEN, num_joints)
        self.load_state_dict(saved["model_state_dict"])
        self.requires_grad_(False).eval()
        self.register_buffer("offsets", torch.arange(FUTURE_FRAMES) * FRAME_SKIP, persistent=False)

    def tokenizer(self, root_R, f):
        """Encoder input for robot root rotations ``root_R`` (B, 3, 3) at reference frames ``f`` (B,)."""
        m = self.motion
        future = torch.minimum(f.unsqueeze(-1) + self.offsets, m.clip_end[m.clip_id[f]].unsqueeze(-1))
        rows = torch.cat((m.q[future].flatten(1), m.qd[future].flatten(1)), -1).unflatten(-1, (FUTURE_FRAMES, 2 * self.J))
        relative = root_R.transpose(-1, -2).unsqueeze(1) @ m.root_R[future]
        return torch.cat((rows, relative[..., :2].flatten(-2)), -1).flatten(1)

    def proprio(self, history):
        """Decoder input from the actor's observation history (B, steps, obs), newest last."""
        J, h = self.J, history[:, -HISTORY:]
        q, qd, w, gravity, action = h[..., :J], h[..., J : 2 * J], h[..., 2 * J : 2 * J + 3], h[..., 2 * J + 3 : 2 * J + 6], h[..., 2 * J + 6 : 3 * J + 6]
        return torch.cat([x.flatten(1) for x in (w, q, qd, action, gravity)], -1)

    def forward(self, history, root_R, f):
        """Runs in full precision: the tokens are rounded, and SONIC was not trained in half precision.
        The tokens carry no gradient; the action is differentiable in the observation history."""
        with torch.no_grad():
            tokens = fsq(self.encoder(self.tokenizer(root_R, f)).unflatten(-1, (TOKENS, TOKEN_DIM))).flatten(1)
        return self.decoder(torch.cat((tokens, self.proprio(history)), -1))
