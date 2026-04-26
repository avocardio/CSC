"""Unit + integration tests for rl/continual_world.py.

These tests are designed to catch the regression we just fixed: the
`importance` parameter being decoupled from the forward pass.
"""

import os, sys, math, copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
import pytest

from rl.continual_world import (
    QuantizedLinear, CWActor, CWCritic, SACAgent, ReplayBuffer,
    TaskReplayStore, CW10_TASKS, OBS_DIM, ACT_DIM, HIDDEN, N_CORE_LAYERS,
)
from models.quantization import quantize


# ============================================================
# Section 1: Quantization primitive
# ============================================================
def test_quantize_b8_near_identity():
    """At b=8 (256 levels) with appropriate exponent, quantization is near-identity."""
    W = torch.randn(8, 16) * 0.1
    b = torch.full((8,), 8.0).view(8, 1)
    e = torch.full((8,), -4.0).view(8, 1)
    Wq = quantize(W, b, e)
    assert torch.allclose(W, Wq, atol=2 ** -4)            # within one quant level


def test_quantize_b0_zero():
    """b=0 zeros the weight (channel removed)."""
    W = torch.randn(4, 8)
    b = torch.zeros(4).view(4, 1)
    e = torch.full((4,), -4.0).view(4, 1)
    Wq = quantize(W, b, e)
    assert (Wq == 0).all()


def test_ste_round_passes_gradient():
    """Round STE: gradient passes as identity."""
    x = torch.randn(10, requires_grad=True)
    b = torch.full((10,), 8.0)
    e = torch.full((10,), -4.0)
    y = quantize(x, b, e)
    y.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


# ============================================================
# Section 2: QuantizedLinear forward + grad coupling
# ============================================================
def test_quantized_linear_forward_shape():
    layer = QuantizedLinear(16, 8)
    x = torch.randn(4, 16)
    y = layer(x)
    assert y.shape == (4, 8)


def test_quantized_linear_b_change_changes_output():
    """Critical invariant: forward depends on bit_depth.

    This is exactly the bug we found: the old `importance` parameter
    was disconnected from the forward pass. Verify our QuantizedLinear
    is properly coupled.
    """
    torch.manual_seed(0)
    layer = QuantizedLinear(16, 8, init_bit=8.0)
    x = torch.randn(4, 16)
    y_full = layer(x).clone()
    # Reduce bit-depth to 1 -> very lossy quantization
    with torch.no_grad():
        layer.quantizer.bit_depth.fill_(1.0)
    y_lossy = layer(x)
    assert not torch.allclose(y_full, y_lossy, atol=1e-4), (
        'Bit-depth must affect forward output - if not, the importance '
        'signal is decoupled (decoy parameter bug).'
    )


def test_quantized_linear_grad_to_bit_depth():
    """Loss on output produces gradient on bit_depth (via STE through quantize)."""
    layer = QuantizedLinear(16, 8, init_bit=8.0)
    x = torch.randn(4, 16)
    target = torch.randn(4, 8)
    y = layer(x)
    loss = ((y - target) ** 2).mean()
    loss.backward()
    assert layer.quantizer.bit_depth.grad is not None
    assert torch.isfinite(layer.quantizer.bit_depth.grad).all()


# ============================================================
# Section 3: CSC actor — importance signal is real
# ============================================================
def test_csc_actor_compression_loss_pushes_b_down():
    """Pure compression pressure should drive bit-depths from 8 toward 0."""
    torch.manual_seed(0)
    actor = CWActor(num_tasks=4, quantize=True)
    quant_params = []
    for layer in actor.core_layers():
        quant_params.extend(layer.quantizer.parameters())
    opt = torch.optim.Adam(quant_params, lr=0.5)
    initial_b = actor.fc1.channel_bit_depths().mean().item()
    for _ in range(50):
        opt.zero_grad()
        loss = actor.compression_loss()
        loss.backward()
        opt.step()
    final_b = actor.fc1.channel_bit_depths().mean().item()
    assert final_b < initial_b - 1.0, (
        f'compression loss didn\'t push b down: {initial_b}->{final_b}'
    )


def test_csc_task_loss_grad_to_b_when_quant_active():
    """Per-channel bit-depth gradients differ when clamping is active.

    Mechanism check: bit-depth gradient flows through the clamp boundary, so
    once `b` enters the active regime (clamping happens), channels with more
    clamped weights should see larger |grad b|.
    """
    torch.manual_seed(0)
    layer = QuantizedLinear(8, 4, init_bit=2.0, init_exp=0.0)
    # b=2, e=0 => range = [-2, 1] (one clamped on top, two on bottom)
    with torch.no_grad():
        # row 0: large magnitudes -> heavy clamping
        layer.weight[0] = torch.tensor([3.0, -3.0, 3.0, -3.0, 3.0, -3.0, 3.0, -3.0])
        # row 1: well within range -> no clamping
        layer.weight[1] = torch.tensor([0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.1])
        # row 2: large magnitudes -> heavy clamping
        layer.weight[2] = torch.tensor([5.0, -5.0, 5.0, -5.0, 5.0, -5.0, 5.0, -5.0])
        # row 3: well within range -> no clamping
        layer.weight[3] = torch.tensor([0.05, 0.05, -0.05, -0.05] * 2)
    x = torch.randn(32, 8)
    y = layer(x)
    loss = y.pow(2).mean()
    loss.backward()
    g = layer.quantizer.bit_depth.grad.abs()
    # Heavily clamped rows must see larger gradient than non-clamped rows
    assert g[0] > g[1] * 10, f'g[0]={g[0]:.4f} g[1]={g[1]:.6f}'
    assert g[2] > g[3] * 10, f'g[2]={g[2]:.4f} g[3]={g[3]:.6f}'


def test_csc_extended_training_differentiates_channels():
    """Extended training drives b into the active regime and creates per-channel variance."""
    torch.manual_seed(0)
    actor = CWActor(num_tasks=1, quantize=True)
    weight_params, quant_params = actor.core_param_groups()
    opt_w = torch.optim.Adam(weight_params, lr=1e-3)
    opt_q = torch.optim.Adam(quant_params, lr=0.5, eps=1e-3)

    x = torch.randn(64, OBS_DIM)
    target_W = torch.randn(ACT_DIM, OBS_DIM) * 0.3
    y_target = x @ target_W.t()

    for _ in range(800):
        opt_w.zero_grad(); opt_q.zero_grad()
        mu, _ = actor(x, 0)
        task_loss = F.mse_loss(mu, y_target)
        comp_loss = 0.05 * actor.compression_loss()      # strong compression pressure
        (task_loss + comp_loss).backward()
        opt_w.step(); opt_q.step()

    # By now bit-depths should have entered the active regime: nonzero variance
    b_all = torch.cat([l.channel_bit_depths() for l in actor.core_layers()])
    b_std = b_all.std().item()
    assert b_std > 0.05, f'bit-depths uniform after 800 steps: std={b_std}'


# ============================================================
# Section 4: Multi-head routing
# ============================================================
def test_actor_heads_are_per_task():
    """Different task IDs route to different heads -> different outputs."""
    torch.manual_seed(0)
    actor = CWActor(num_tasks=4, quantize=False)
    obs = torch.randn(8, OBS_DIM)
    mu0, _ = actor(obs, 0)
    mu3, _ = actor(obs, 3)
    assert not torch.allclose(mu0, mu3, atol=1e-4)


def test_actor_per_sample_task_ids():
    """Per-sample task_id should select the matching scalar-task output."""
    torch.manual_seed(0)
    actor = CWActor(num_tasks=4, quantize=False)
    obs = torch.randn(4, OBS_DIM)
    t = torch.tensor([0, 1, 2, 3])
    mu_batch, _ = actor(obs, t)
    for i in range(4):
        mu_i, _ = actor(obs[i:i + 1], int(t[i].item()))
        assert torch.allclose(mu_batch[i], mu_i.squeeze(0), atol=1e-5), (
            f'sample {i}: vector vs scalar dispatch mismatch')


def test_critic_heads_are_per_task():
    torch.manual_seed(0)
    critic = CWCritic(num_tasks=3)
    obs = torch.randn(4, OBS_DIM); act = torch.randn(4, ACT_DIM)
    q1_a, _ = critic(obs, act, 0)
    q1_b, _ = critic(obs, act, 2)
    assert not torch.allclose(q1_a, q1_b, atol=1e-4)


def test_critic_returns_twin_q():
    """CWCritic returns (q1, q2) — the twin Q values for SAC double Q-learning."""
    critic = CWCritic(num_tasks=2)
    obs = torch.randn(3, OBS_DIM); act = torch.randn(3, ACT_DIM)
    out = critic(obs, act, 0)
    assert isinstance(out, tuple) and len(out) == 2
    assert out[0].shape == (3, 1) and out[1].shape == (3, 1)


# ============================================================
# Section 5: Replay buffer + task store
# ============================================================
def test_replay_buffer_basic():
    buf = ReplayBuffer(capacity=128, device='cpu')
    for _ in range(10):
        buf.add(np.zeros(OBS_DIM), np.zeros(ACT_DIM), 0.0,
                np.zeros(OBS_DIM), 0.0, 0)
    assert buf.size == 10
    s, a, r, ns, d, t = buf.sample(5)
    assert s.shape == (5, OBS_DIM); assert t.shape == (5,)


def test_replay_store_remainder_distribution():
    """Per-task remainder should be distributed - we should NOT lose samples."""
    buf = ReplayBuffer(capacity=128, device='cpu')
    for _ in range(50):
        buf.add(np.zeros(OBS_DIM), np.zeros(ACT_DIM), 0.0,
                np.zeros(OBS_DIM), 0.0, 0)
    store = TaskReplayStore(device='cpu')
    store.add_task(buf, n=20); store.add_task(buf, n=20); store.add_task(buf, n=20)
    out = store.sample(31)
    assert out[0].shape == (31, OBS_DIM), (
        f'TaskReplayStore lost samples: got {out[0].shape[0]}, expected 31')


# ============================================================
# Section 6: SAC agent end-to-end smoke test
# ============================================================
@pytest.mark.parametrize('method', ['finetune', 'replay', 'ewc', 'l2', 'mas', 'csc'])
def test_agent_update_runs(method):
    """Smoke: each method runs one training step without exception."""
    torch.manual_seed(0)
    agent = SACAgent(method=method, n_tasks=2, batch_size=8, device='cpu',
                     replay_per_task=64)
    buf = ReplayBuffer(capacity=256, device='cpu')
    for _ in range(64):
        buf.add(np.random.randn(OBS_DIM).astype(np.float32),
                np.random.randn(ACT_DIM).astype(np.float32),
                np.random.randn(), np.random.randn(OBS_DIM).astype(np.float32),
                0.0, 0)
    agent.reset_for_new_task(0)
    for _ in range(3):
        agent.update(buf)
    # If replay/csc, simulate a task end so the next update mixes replay
    if agent.replay_store is not None:
        agent.on_task_end(buf, 0)
        agent.reset_for_new_task(1)
        for _ in range(3):
            agent.update(buf)


# ============================================================
# Section 7: Soft protection actually scales gradients
# ============================================================
def test_csc_grad_scaling_active_after_task_end():
    """After task 0 with non-zero acc_bits, core grads must shrink for protected channels."""
    torch.manual_seed(0)
    agent = SACAgent(method='csc', n_tasks=2, batch_size=8, device='cpu',
                     replay_per_task=32, grad_scale_beta=10.0)
    buf = ReplayBuffer(capacity=128, device='cpu')
    for _ in range(64):
        buf.add(np.random.randn(OBS_DIM).astype(np.float32),
                np.random.randn(ACT_DIM).astype(np.float32),
                np.random.randn(), np.random.randn(OBS_DIM).astype(np.float32),
                0.0, 0)
    agent.reset_for_new_task(0)
    # Manually set acc_bits to a heterogeneous pattern
    with torch.no_grad():
        agent.acc_bits[0].zero_()
        agent.acc_bits[0][:128] = 8.0                      # half "important"
    agent.reset_for_new_task(1)

    # Capture pre-scaling gradient by skipping the scaling
    s, a, r, ns, d, t = buf.sample(8)
    alpha_per_sample = agent.log_alpha.detach().exp()[t].unsqueeze(-1)
    na2, lp2, _ = agent.actor.sample(s, t)
    q1pi, q2pi = agent.critic(s, na2, t)
    actor_loss = (alpha_per_sample * lp2 - torch.min(q1pi, q2pi)).mean()
    actor_loss = actor_loss + agent.gamma_comp * agent.actor.compression_loss()
    agent.actor_opt.zero_grad(); agent.imp_opt.zero_grad()
    actor_loss.backward()

    # Pre-scaling gradient
    g_pre = agent.actor.fc1.weight.grad.clone()
    agent._scale_core_grads_by_acc_bits()
    g_post = agent.actor.fc1.weight.grad.clone()
    # First 128 rows: should be scaled by 1/(1+10*8) = 1/81
    # Last 128 rows: should be unchanged (acc=0 -> scale=1)
    expected_first = g_pre[:128] / 81.0
    expected_last = g_pre[128:]
    assert torch.allclose(g_post[:128], expected_first, atol=1e-6)
    assert torch.allclose(g_post[128:], expected_last, atol=1e-6)


# ============================================================
# Section 8: EWC Fisher sanity
# ============================================================
def test_ewc_fisher_nonzero_after_task():
    """Fisher must be non-trivial after computing on data."""
    torch.manual_seed(0)
    agent = SACAgent(method='ewc', n_tasks=2, batch_size=8, device='cpu',
                     replay_per_task=32)
    buf = ReplayBuffer(capacity=512, device='cpu')
    for _ in range(300):
        buf.add(np.random.randn(OBS_DIM).astype(np.float32),
                np.random.randn(ACT_DIM).astype(np.float32),
                np.random.randn(), np.random.randn(OBS_DIM).astype(np.float32),
                0.0, 0)
    agent.reset_for_new_task(0)
    agent.on_task_end(buf, 0)
    # Some Fisher value should be substantially above the floor of 1e-5
    max_f = max(f.max().item() for f in agent.ewc_fisher.values())
    assert max_f > 1e-3, f'Fisher all near floor: max={max_f}'


def test_ewc_penalty_zero_at_snapshot():
    """At the moment of snapshot, EWC penalty is exactly zero."""
    torch.manual_seed(0)
    agent = SACAgent(method='ewc', n_tasks=2, batch_size=8, device='cpu',
                     replay_per_task=32)
    buf = ReplayBuffer(capacity=512, device='cpu')
    for _ in range(300):
        buf.add(np.random.randn(OBS_DIM).astype(np.float32),
                np.random.randn(ACT_DIM).astype(np.float32),
                np.random.randn(), np.random.randn(OBS_DIM).astype(np.float32),
                0.0, 0)
    agent.reset_for_new_task(0)
    agent.on_task_end(buf, 0)
    pen = sum((agent.ewc_fisher[n] * (p - agent.ewc_params[n]).pow(2)).sum()
              for n, p in agent.actor.named_parameters() if n in agent.ewc_fisher)
    assert pen.item() < 1e-8, f'penalty at snapshot was {pen.item()}'


# ============================================================
# Section 9: MAS — must include log_std term
# ============================================================
def test_mas_uses_log_std_term():
    """Sanity: removing log_std from the policy shouldn't shrink importance to 0."""
    # A weak indirect test: produce data where the log_std head is the dominant
    # gradient pathway for some param, then check MAS importance > floor.
    torch.manual_seed(0)
    agent = SACAgent(method='mas', n_tasks=2, batch_size=8, device='cpu',
                     replay_per_task=32)
    buf = ReplayBuffer(capacity=512, device='cpu')
    for _ in range(200):
        buf.add(np.random.randn(OBS_DIM).astype(np.float32),
                np.random.randn(ACT_DIM).astype(np.float32),
                np.random.randn(), np.random.randn(OBS_DIM).astype(np.float32),
                0.0, 0)
    agent.reset_for_new_task(0)
    # Spike log_std head weights -> log_std term will dominate ||mu||^2
    with torch.no_grad():
        agent.actor.heads_ls_W[0].normal_(mean=0, std=2.0)
    agent.on_task_end(buf, 0)
    # Importance should be non-trivial on the log_std head we spiked
    n = 'heads_ls_W'
    assert n in agent.mas_imp
    imp_ls = agent.mas_imp[n][0].abs().mean().item()
    imp_mu = agent.mas_imp.get('heads_mu_W', torch.zeros(1))[0].abs().mean().item()
    # log_std-driven importance > mu-driven importance after spiking
    assert imp_ls > imp_mu, (
        f'MAS appears to ignore log_std: imp_ls={imp_ls} <= imp_mu={imp_mu}')


# ============================================================
# Section 10: PackNet ownership
# ============================================================
def test_packnet_ownership_init():
    agent = SACAgent(method='packnet', n_tasks=4, batch_size=8, device='cpu')
    for layer in agent.actor.core_layers():
        owner = agent.pn_owner[id(layer.weight)]
        assert (owner == 0).all()


def test_packnet_grad_mask_zeros_other_owners():
    agent = SACAgent(method='packnet', n_tasks=2, batch_size=8, device='cpu')
    # Mark some weights as owned by task 1
    layer = agent.actor.fc1
    owner = agent.pn_owner[id(layer.weight)]
    owner[:128] = 1                                         # half owned by task 1
    # Synthetic gradient
    layer.weight.grad = torch.ones_like(layer.weight)
    agent._current_task = 0
    agent._packnet_mask_grads()
    assert (layer.weight.grad[:128] == 0).all()
    assert (layer.weight.grad[128:] == 1).all()


# ============================================================
# Section 11: Per-task log_alpha
# ============================================================
def test_log_alpha_is_per_task():
    agent = SACAgent(method='finetune', n_tasks=5, device='cpu')
    assert agent.log_alpha.shape == (5,)


# ============================================================
# Section 12: Checkpoint round-trip
# ============================================================
def test_checkpoint_round_trip(tmp_path):
    from rl.continual_world import save_checkpoint, load_checkpoint
    torch.manual_seed(0)
    agent = SACAgent(method='csc', n_tasks=2, batch_size=8, device='cpu',
                     replay_per_task=32)
    buf = ReplayBuffer(capacity=128, device='cpu')
    for _ in range(64):
        buf.add(np.zeros(OBS_DIM, dtype=np.float32),
                np.zeros(ACT_DIM, dtype=np.float32),
                0.0, np.zeros(OBS_DIM, dtype=np.float32), 0.0, 0)
    agent.reset_for_new_task(0)
    for _ in range(3):
        agent.update(buf)
    agent.on_task_end(buf, 0)
    path = str(tmp_path / 'ckpt.pt')
    save_checkpoint(path, agent, task_idx=0,
                    eval_history=[{'a': 1}], results={'b': 2},
                    rng_state={'torch': torch.get_rng_state(),
                               'numpy': np.random.get_state()})

    agent2 = SACAgent(method='csc', n_tasks=2, batch_size=8, device='cpu',
                      replay_per_task=32)
    t_idx, hist, res, rng = load_checkpoint(path, agent2)
    assert t_idx == 0; assert hist == [{'a': 1}]; assert res == {'b': 2}
    # Actor state matches
    for (n1, p1), (n2, p2) in zip(agent.actor.named_parameters(),
                                  agent2.actor.named_parameters()):
        assert torch.allclose(p1, p2)
    # acc_bits restored
    for a1, a2 in zip(agent.acc_bits, agent2.acc_bits):
        assert torch.allclose(a1, a2)


# ============================================================
# Section 13: bit-floor prevents permanent collapse
# ============================================================
def test_bit_floor_prevents_collapse():
    """With bit_floor=1.0, no channel should be at exactly 0 after heavy compression."""
    torch.manual_seed(0)
    agent = SACAgent(method='csc', n_tasks=1, batch_size=8, device='cpu',
                     gamma_comp=1.0, lr_quant=10.0, bit_floor=1.0,    # huge pressure
                     replay_per_task=32)
    buf = ReplayBuffer(capacity=128, device='cpu')
    for _ in range(64):
        buf.add(np.random.randn(OBS_DIM).astype(np.float32),
                np.random.randn(ACT_DIM).astype(np.float32),
                np.random.randn(), np.random.randn(OBS_DIM).astype(np.float32),
                0.0, 0)
    agent.reset_for_new_task(0)
    for _ in range(50):
        agent.update(buf)
    for layer in agent.actor.core_layers():
        b = layer.channel_bit_depths().detach()
        assert (b >= 1.0).all(), (
            f'bit-depth dropped below floor: min={b.min().item()}')


def test_quantization_in_forward_pass():
    """Verify forward pass DEPENDS on bit_depth - regression check for the
    decoupled-importance bug.
    """
    torch.manual_seed(0)
    actor = CWActor(num_tasks=1, quantize=True)
    obs = torch.randn(8, OBS_DIM)
    mu_b8, _ = actor(obs, 0)

    with torch.no_grad():
        for layer in actor.core_layers():
            layer.quantizer.bit_depth.data.fill_(1.0)
    mu_b1, _ = actor(obs, 0)
    assert not torch.allclose(mu_b8, mu_b1, atol=1e-3), (
        'Output unchanged when b dropped from 8 to 1 — '
        'quantization not active in forward (decoupled-importance bug!)')


def test_action_in_valid_range():
    """Sampled actions must be in tanh-squashed [-1, 1]."""
    torch.manual_seed(0)
    actor = CWActor(num_tasks=2, quantize=False)
    obs = torch.randn(16, OBS_DIM)
    a, _, _ = actor.sample(obs, 0)
    assert (a >= -1).all() and (a <= 1).all()


def test_per_task_alpha_gradient_routing():
    """Only the log_alpha entries used in this batch should accumulate gradient."""
    torch.manual_seed(0)
    agent = SACAgent(method='replay', n_tasks=4, batch_size=16, device='cpu',
                     replay_per_task=64)
    buf = ReplayBuffer(capacity=256, device='cpu')
    for _ in range(80):
        buf.add(np.random.randn(OBS_DIM).astype(np.float32),
                np.random.randn(ACT_DIM).astype(np.float32),
                np.random.randn(), np.random.randn(OBS_DIM).astype(np.float32),
                0.0, 1)             # all transitions belong to task 1
    agent.reset_for_new_task(1)
    a_before = agent.log_alpha.data.clone()
    agent.update(buf)
    a_after = agent.log_alpha.data
    diff = (a_after - a_before).abs()
    # Only entry 1 should have moved (only task 1 in the batch since replay_store is empty)
    assert diff[1].item() > 0
    for i in (0, 2, 3):
        assert diff[i].item() < 1e-9, (
            f'log_alpha[{i}] moved when task {i} was not in batch: diff={diff[i].item()}')


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-x', '-v']))
