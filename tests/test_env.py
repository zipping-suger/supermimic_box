import torch

from supermimic_box.env import make_task
from supermimic_box.mdp import CONTROL_DT
from supermimic_box.state import pack


def test_reference_library(data, robot, registry, clips):
    from supermimic_box.motion import Motion

    for mode in ("flow", "full"):
        m = Motion(data / "box", robot, registry["objects"], clips=clips, goal_mode=mode)
        assert m.num_frames == int(m.clip_len.sum()) and m.tasks == ["carry", "push", "carry"]
        assert (m.q >= m.kin.q_lo - 1e-6).all() and (m.q <= m.kin.q_hi + 1e-6).all()
        assert m.goal(torch.arange(7), 10).shape == (7, m.goal_dim(10))
        # The goal of a clip's last frame repeats that frame instead of reading the next clip.
        last = m.clip_end[:1]
        assert torch.equal(m.next_frame(last), last)
        goal = m.goal(last, 3).unflatten(-1, (3, -1))
        assert torch.allclose(goal[0, 0], goal[0, 2])
    # Carrying and pushing use the hands, not the feet; every clip begins without contact.
    assert m.contact[m.clip_start].sum() == 0
    per_clip = torch.stack([m.contact[a : b + 1].sum(0) for a, b in zip(m.clip_start, m.clip_end)])
    assert (per_clip[:, 2:] > 0).all() and (per_clip[:, :2] == 0).all()
    table, count = m.precontact_frames()
    for clip in range(3):
        frame = int(table[clip, 0])
        assert m.clip_start[clip] <= frame < m.clip_end[clip] and m.contact[frame].sum() == 0 and m.contact[frame + 10].sum() > 0


def test_reset_reproduces_the_reference_and_rewards_peak_there(tiny_config):
    task = make_task(tiny_config)
    env, m, mdp = task.env, task.motion, task.mdp
    everyone = torch.ones(env.N, dtype=torch.bool)
    frames = env.reset_map[m.clip_start[torch.tensor([0, 0, 1, 2])] + torch.tensor([0, 40, 0, 40])]
    env.reset(everyone, frames)
    state = env.physics.state()
    assert (pack(state) - pack(m.ref_state(frames))).abs().max() < 1e-5
    assert env.age.sum() == 0 and torch.allclose(env.tables, m.tables(frames))
    # The platforms are where the clip has them.
    for world in range(env.N):
        k = env.physics.assignment[world]
        assert abs(env.physics.data[world].mocap_pos[env.physics.mocap[k]].reshape(-1) - m.tables(frames)[world].numpy()).max() < 1e-6
    # At the reference state, with the reference's contacts, every tracking bonus is at its weight.
    c = tiny_config.reward
    action = task.controller.reference_action(state[4])
    still = torch.zeros(env.N, 2)
    reward, terms = mdp.reward(state, frames, action, action, m.contact[frames], still, still, torch.ones(env.N, dtype=torch.bool), env.context)
    for name in ("anchor_pos", "anchor_ori", "body_pos", "body_ori", "hand_pos", "hand_ori", "object_pos", "object_ori",
                 "interact_pos", "interact_ori", "contact"):  # fmt: skip
        assert torch.allclose(terms[name], torch.full((env.N,), getattr(c, name) * CONTROL_DT), atol=2e-4), name
    assert mdp.errors(state, frames).abs().max() < 2e-2 and not mdp.hard_termination(state, frames)[0].any()
    assert mdp.soft_termination(state, frames).max() < 1e-3
    # Observations and model inputs have the advertised sizes.
    assert mdp.model_features(state, frames, env.context).shape[-1] == mdp.model_feature_dim
    assert env.history.shape == (env.N, tiny_config.history, mdp.obs(state, frames, action, env.context).shape[-1])


def test_steps_follow_the_clip_and_resets_exchange_objects(tiny_config):
    task = make_task(tiny_config)
    env, m = task.env, task.motion
    everyone = torch.ones(env.N, dtype=torch.bool)
    frames = env.reset_map[m.clip_start[torch.tensor([0, 1, 2, 0])]]
    env.reset(everyone, frames)
    row = env.step(task.controller.reference_action(m.q[m.next_frame(env.f)]))
    assert torch.equal(env.f, frames + 1) and row["valid"].all() and torch.isfinite(row["reward"]).all()
    assert row["contact"].shape == (env.N, 4) and row["foot_contact"].shape == (env.N, 2)
    # Standing on the ground at the first frame of a clip.
    assert row["foot_contact"].sum(-1).min() >= 1
    # A world takes the box of the clip it is reset to.
    swapped = env.reset_map[m.clip_start[torch.tensor([1, 2, 0, 0])]]
    env.reset(everyone, swapped)
    assert env.physics.assignment == m.object_of(swapped).tolist()
    assert (pack(env.physics.state()) - pack(m.ref_state(swapped))).abs().max() < 1e-5
