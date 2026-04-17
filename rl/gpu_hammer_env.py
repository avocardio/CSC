"""GPU-accelerated Sawyer hammer environment using MuJoCo Warp.

Ports the MetaWorld hammer-v3 task to run ~1024 parallel envs on GPU at 80k+ sps.
All state stays on GPU (as PyTorch tensors via wp.to_torch).

Key design decisions:
- Reuses MetaWorld's XML (sawyer_hammer.xml) as-is
- Reward, reset, and observation logic reimplemented in vectorized PyTorch
- Mocap body control for arm XYZ (matches MetaWorld convention)
- Frame-skip of 5 physics steps per control step (CW default)
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import mujoco
import warp as wp
import mujoco_warp as mjw

import metaworld as _mw
HAMMER_XML = os.path.join(os.path.dirname(_mw.__file__), 'assets', 'sawyer_xyz', 'sawyer_hammer.xml')

# Action/obs dims (matching MetaWorld)
ACT_DIM = 4  # [dx, dy, dz, gripper]
OBS_DIM = 39

# Control limits
ACTION_SCALE = 1.0 / 100.0  # 1 cm per unit action
MOCAP_LOW = torch.tensor([-0.2, 0.5, 0.06])
MOCAP_HIGH = torch.tensor([0.2, 0.7, 0.6])

# Task-specific constants
HAND_INIT_POS = torch.tensor([0.0, 0.4, 0.2])
HAMMER_INIT_LOW = torch.tensor([-0.1, 0.4, 0.0])
HAMMER_INIT_HIGH = torch.tensor([0.1, 0.5, 0.0])
GOAL_POS = torch.tensor([0.24, 0.74, 0.11])  # Nail target
NAIL_HEAD_OFFSET = torch.tensor([0.16, 0.06, 0.0])  # Hammer origin -> head
NAIL_JOINT_IDX = 16  # qpos index of NailSlideJoint
HAMMER_QPOS_START = 9  # qpos index where hammer free joint begins (pos 9:12, quat 12:16)
HAMMER_QVEL_START = 9  # qvel index (free joint has 6 DOF: 3 lin + 3 ang)

FRAME_SKIP = 5
MAX_EP_LEN = 200
SUCCESS_THRESHOLD = 0.09  # NailSlideJoint > 0.09 = success


def _tolerance(x: torch.Tensor, bounds=(0.0, 0.02), margin: float = 0.2) -> torch.Tensor:
    """Vectorized version of dm_control reward_utils.tolerance (long_tail sigmoid)."""
    lo, hi = bounds
    in_bounds = ((x >= lo) & (x <= hi)).float()
    # Distance outside bounds
    d = torch.where(x < lo, lo - x, torch.where(x > hi, x - hi, torch.zeros_like(x)))
    # long_tail sigmoid: value = 1 at d=0, value = ~0.1 at d=margin
    # long_tail: 1 / (d/margin * scale^2 + 1) where scale picks value=0.1 at d=margin
    scale = torch.sqrt(torch.tensor(1.0 / 0.1 - 1.0))  # solve 1/(s^2+1) = 0.1
    x_scaled = d / margin * scale
    sigmoid_val = 1.0 / (x_scaled * x_scaled + 1.0)
    return in_bounds + (1 - in_bounds) * sigmoid_val


class GPUHammerEnv:
    """Batched GPU Sawyer hammer environment."""

    def __init__(self, n_envs: int = 1024, device: str = 'cuda:0'):
        self.n_envs = n_envs
        self.device = torch.device(device)

        # Load MuJoCo model (CPU side)
        self.mjm = mujoco.MjModel.from_xml_path(HAMMER_XML)

        # Reset mocap welds (MetaWorld convention): set weld data to identity
        # This is required for the mocap body to properly actuate the hand
        for i in range(self.mjm.eq_data.shape[0]):
            if self.mjm.eq_type[i] == mujoco.mjtEq.mjEQ_WELD:
                self.mjm.eq_data[i] = np.array(
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 5.0]
                )

        # Initialize Warp
        wp.init()
        with wp.ScopedDevice(device):
            self.m = mjw.put_model(self.mjm)
            self.d = mjw.make_data(self.mjm, nworld=n_envs)

        # Grab GPU tensor views (zero-copy via DLPack)
        self._sync_torch_views()

        # Move constants to device
        self.mocap_low = MOCAP_LOW.to(self.device)
        self.mocap_high = MOCAP_HIGH.to(self.device)
        self.hand_init_pos = HAND_INIT_POS.to(self.device)
        self.hammer_init_low = HAMMER_INIT_LOW.to(self.device)
        self.hammer_init_high = HAMMER_INIT_HIGH.to(self.device)
        self.goal_pos = GOAL_POS.to(self.device)
        self.nail_head_offset = NAIL_HEAD_OFFSET.to(self.device)

        # Find site ids
        self.end_effector_sid = mujoco.mj_name2id(
            self.mjm, mujoco.mjtObj.mjOBJ_SITE, 'endEffector')
        self.right_ee_sid = mujoco.mj_name2id(
            self.mjm, mujoco.mjtObj.mjOBJ_SITE, 'rightEndEffector')
        self.left_ee_sid = mujoco.mj_name2id(
            self.mjm, mujoco.mjtObj.mjOBJ_SITE, 'leftEndEffector')
        # hammer body id for getting xpos
        self.hammer_bid = mujoco.mj_name2id(
            self.mjm, mujoco.mjtObj.mjOBJ_BODY, 'hammer')
        self.nail_bid = mujoco.mj_name2id(
            self.mjm, mujoco.mjtObj.mjOBJ_BODY, 'nail_link')

        # Episode state (per env)
        self.ep_len = torch.zeros(n_envs, dtype=torch.int32, device=self.device)
        self.prev_obs = torch.zeros(n_envs, 18, device=self.device)
        self.success_once = torch.zeros(n_envs, dtype=torch.bool, device=self.device)
        self.return_ = torch.zeros(n_envs, device=self.device)
        self.ep_return = torch.zeros(n_envs, device=self.device)

    def _sync_torch_views(self):
        """Get PyTorch views of the Warp data arrays (zero-copy)."""
        # These are torch tensors on GPU, shape (nworld, ...)
        self.qpos = wp.to_torch(self.d.qpos)           # (nworld, nq=17)
        self.qvel = wp.to_torch(self.d.qvel)           # (nworld, nv=16)
        self.ctrl = wp.to_torch(self.d.ctrl)           # (nworld, nu=2)
        self.mocap_pos = wp.to_torch(self.d.mocap_pos)   # (nworld, 1, 3)
        self.mocap_quat = wp.to_torch(self.d.mocap_quat) # (nworld, 1, 4)
        # site_xpos shape (nworld, nsite, 3), body_xpos (nworld, nbody, 3)
        self.site_xpos = wp.to_torch(self.d.site_xpos)
        self.xpos = wp.to_torch(self.d.xpos)
        self.xquat = wp.to_torch(self.d.xquat)

    @torch.no_grad()
    def _physics_step(self, n_frames: int = 1):
        """Step Warp physics n_frames times."""
        with wp.ScopedDevice(str(self.device)):
            for _ in range(n_frames):
                mjw.step(self.m, self.d)
        # Views should auto-update since they're zero-copy

    @torch.no_grad()
    def _forward(self):
        """Compute kinematics (needed after manually setting qpos)."""
        with wp.ScopedDevice(str(self.device)):
            mjw.forward(self.m, self.d)

    @torch.no_grad()
    def _get_hand_pos(self) -> torch.Tensor:
        """Get endEffector site xpos: (nworld, 3)."""
        return self.site_xpos[:, self.end_effector_sid, :]

    @torch.no_grad()
    def _get_gripper_distance(self) -> torch.Tensor:
        """Get normalized gripper open amount: (nworld,) in [0,1]."""
        right = self.site_xpos[:, self.right_ee_sid, :]
        left = self.site_xpos[:, self.left_ee_sid, :]
        dist = torch.norm(right - left, dim=-1)
        return (dist / 0.1).clamp(0.0, 1.0)

    @torch.no_grad()
    def _get_hammer_pos(self) -> torch.Tensor:
        return self.xpos[:, self.hammer_bid, :]

    @torch.no_grad()
    def _get_hammer_quat(self) -> torch.Tensor:
        return self.xquat[:, self.hammer_bid, :]

    @torch.no_grad()
    def _get_nail_pos(self) -> torch.Tensor:
        return self.xpos[:, self.nail_bid, :]

    @torch.no_grad()
    def _get_nail_quat(self) -> torch.Tensor:
        return self.xquat[:, self.nail_bid, :]

    @torch.no_grad()
    def _compute_curr_obs(self) -> torch.Tensor:
        """Returns (nworld, 18): [hand(3), gripper(1), hammer_pos(3), hammer_quat(4),
        nail_pos(3), nail_quat(4)]"""
        hand = self._get_hand_pos()
        grip = self._get_gripper_distance().unsqueeze(-1)
        hp = self._get_hammer_pos()
        hq = self._get_hammer_quat()
        np_ = self._get_nail_pos()
        nq = self._get_nail_quat()
        return torch.cat([hand, grip, hp, hq, np_, nq], dim=-1)

    @torch.no_grad()
    def _get_obs(self) -> torch.Tensor:
        """Returns (nworld, 39): current + previous + goal."""
        curr = self._compute_curr_obs()
        goal = self.goal_pos.unsqueeze(0).expand(self.n_envs, 3)
        obs = torch.cat([curr, self.prev_obs, goal], dim=-1)
        self.prev_obs = curr
        return obs

    @torch.no_grad()
    def reset(self) -> torch.Tensor:
        """Reset all envs. Mimics MetaWorld reset: set qpos defaults, then run
        physics for 50 steps with mocap at hand_init_pos to let the weld pull
        the Sawyer arm into position."""
        # Reset qpos to model default (qpos0), reset velocities
        qpos0 = torch.as_tensor(self.mjm.qpos0, device=self.device).float()
        self.qpos.copy_(qpos0.unsqueeze(0).expand(self.n_envs, -1))
        self.qvel.zero_()

        # Randomize hammer XYZ position
        rand = torch.rand(self.n_envs, 3, device=self.device)
        hammer_pos = self.hammer_init_low + rand * (
            self.hammer_init_high - self.hammer_init_low)
        self.qpos[:, HAMMER_QPOS_START:HAMMER_QPOS_START + 3] = hammer_pos
        # Nail joint starts at 0
        self.qpos[:, NAIL_JOINT_IDX] = 0.0

        # Set mocap to hand initial position and quaternion
        self.mocap_pos[:, 0, :] = self.hand_init_pos
        mocap_q = torch.tensor([1.0, 0.0, 1.0, 0.0], device=self.device)
        self.mocap_quat[:, 0, :] = mocap_q

        # Set gripper ctrl to closed (-1, 1) per MetaWorld reset
        self.ctrl[:, 0] = -1.0
        self.ctrl[:, 1] = 1.0

        # Relax physics: 50 steps of 5 frame-skip each = 250 physics steps total
        # This lets the mocap weld pull the Sawyer arm into position
        self._physics_step(50 * FRAME_SKIP)

        # Reset episode state
        self.ep_len.zero_()
        self.success_once.zero_()
        self.ep_return.zero_()
        self.prev_obs.zero_()

        # Get initial observation
        obs = self._compute_curr_obs()
        self.prev_obs = obs.clone()  # init prev with current
        goal = self.goal_pos.unsqueeze(0).expand(self.n_envs, 3)
        return torch.cat([obs, obs, goal], dim=-1)  # prev=curr at reset

    @torch.no_grad()
    def step(self, action: torch.Tensor):
        """Step all envs with action (n_envs, 4). Returns (obs, reward, done, info)."""
        action = action.clamp(-1, 1)

        # Update mocap position by action delta
        new_mocap = self.mocap_pos[:, 0, :] + action[:, :3] * ACTION_SCALE
        new_mocap = torch.max(torch.min(new_mocap, self.mocap_high), self.mocap_low)
        self.mocap_pos[:, 0, :] = new_mocap

        # Gripper: ctrl[0] = +gripper, ctrl[1] = -gripper (matching MetaWorld)
        self.ctrl[:, 0] = action[:, 3]
        self.ctrl[:, 1] = -action[:, 3]

        # Physics step (frame skip)
        self._physics_step(FRAME_SKIP)

        # Get new observation
        obs = self._get_obs()

        # Compute reward
        reward = self._compute_reward(action, obs)

        # Check success: nail joint qpos > 0.09
        nail_pos = self.qpos[:, NAIL_JOINT_IDX]
        success = nail_pos > SUCCESS_THRESHOLD
        self.success_once = self.success_once | success

        # Episode length tracking
        self.ep_len += 1
        self.ep_return += reward
        truncated = self.ep_len >= MAX_EP_LEN
        done = truncated  # No early termination

        # Info dict
        info = {
            'success': success.float(),
            'success_once': self.success_once.float(),
            'ep_return': self.ep_return.clone(),
        }

        return obs, reward, done, info

    @torch.no_grad()
    def _compute_reward(self, action: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        """Vectorized hammer reward (MetaWorld v2 version, simplified).

        reward = (2 * grab_reward + 6 * in_place_reward) * quat_reward
        Success bonus: +10 when nail > threshold and reward > 5.
        """
        hand = obs[:, :3]
        hammer_pos = obs[:, 4:7]
        hammer_quat = obs[:, 7:11]
        hammer_head = hammer_pos + self.nail_head_offset

        # Quaternion reward: distance from [1,0,0,0]
        ideal = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        quat_err = torch.norm(hammer_quat - ideal, dim=-1)
        quat_reward = (1.0 - quat_err / 0.4).clamp(min=0.0)

        # Grab reward (simplified caging): proximity of hand to hammer + gripper closed
        hand_to_hammer = torch.norm(hand - hammer_pos, dim=-1)
        grab_proximity = (1.0 - (hand_to_hammer / 0.1).clamp(0.0, 1.0))
        gripper_closed = (1.0 - obs[:, 3])  # gripper_distance small = closed
        grab_reward = 0.5 * grab_proximity + 0.5 * grab_proximity * gripper_closed

        # In-place reward: hammer_head close to target
        pos_err = torch.norm(self.goal_pos.unsqueeze(0) - hammer_head, dim=-1)
        lifted = (hammer_head[:, 2] > 0.02).float()
        in_place = 0.1 * lifted + 0.9 * _tolerance(
            pos_err, bounds=(0.0, 0.02), margin=0.2)

        reward = (2.0 * grab_reward + 6.0 * in_place) * quat_reward

        # Success bonus
        nail_pos = self.qpos[:, NAIL_JOINT_IDX]
        success = (nail_pos > SUCCESS_THRESHOLD) & (reward > 5.0)
        reward = torch.where(success, torch.tensor(10.0, device=self.device), reward)

        return reward

    @torch.no_grad()
    def auto_reset_step(self, action: torch.Tensor):
        """Step + auto-reset done envs. Returns (obs, reward, done, info).
        obs is the post-reset observation for done envs, same as SB3/gymnasium
        vector env convention. The 'real' next obs (before reset) is in info.
        """
        obs, reward, done, info = self.step(action)
        info['real_next_obs'] = obs.clone()
        if done.any():
            self.reset_done(done)
            # Recompute observation after reset for done envs
            new_obs = self._compute_curr_obs()
            goal = self.goal_pos.unsqueeze(0).expand(self.n_envs, 3)
            # Mix: done envs get reset obs, others keep their current
            reset_obs = torch.cat([new_obs, new_obs, goal], dim=-1)
            obs = torch.where(done.unsqueeze(-1), reset_obs, obs)
        return obs, reward, done, info

    @torch.no_grad()
    def reset_done(self, done: torch.Tensor):
        """Reset only the envs where done==True. Essential for async rollout.

        Note: since mocap_pos and qpos are PyTorch views of GPU memory,
        we can assign to them directly without a Warp kernel. We then run
        a few relaxation physics steps on ALL envs (the ones not being reset
        will just continue moving, which is typically fine since we're not
        storing those transitions)."""
        if not done.any():
            return

        idx = done.nonzero(as_tuple=False).squeeze(-1)
        n_done = len(idx)

        # Reset qpos and qvel for done envs
        qpos0 = torch.as_tensor(self.mjm.qpos0, device=self.device).float()
        self.qpos[idx] = qpos0.unsqueeze(0).expand(n_done, -1)
        self.qvel[idx] = 0.0

        # Randomize hammer position for done envs
        rand = torch.rand(n_done, 3, device=self.device)
        hammer_pos = self.hammer_init_low + rand * (
            self.hammer_init_high - self.hammer_init_low)
        self.qpos[idx, HAMMER_QPOS_START:HAMMER_QPOS_START + 3] = hammer_pos
        self.qpos[idx, NAIL_JOINT_IDX] = 0.0

        # Reset mocap for done envs
        self.mocap_pos[idx, 0, :] = self.hand_init_pos
        mocap_q = torch.tensor([1.0, 0.0, 1.0, 0.0], device=self.device)
        self.mocap_quat[idx, 0, :] = mocap_q

        # Episode state
        self.ep_len[idx] = 0
        self.success_once[idx] = False
        self.ep_return[idx] = 0.0
        self.prev_obs[idx] = 0.0

        # Note: we don't run relaxation steps here to keep step fast.
        # The first few steps after reset will have the hand drifting
        # toward init position. This is a tradeoff for speed.


def benchmark():
    """Quick benchmark of env throughput."""
    import time

    for n_envs in [64, 256, 1024, 4096]:
        env = GPUHammerEnv(n_envs=n_envs)
        obs = env.reset()
        print(f'n_envs={n_envs}: obs shape = {obs.shape}')

        # Warmup
        action = torch.zeros(n_envs, ACT_DIM, device='cuda:0')
        for _ in range(10):
            env.step(action)
        torch.cuda.synchronize()

        # Benchmark
        n_steps = 200
        t0 = time.time()
        for _ in range(n_steps):
            action = torch.rand(n_envs, ACT_DIM, device='cuda:0') * 2 - 1
            obs, reward, done, info = env.step(action)
        torch.cuda.synchronize()
        elapsed = time.time() - t0
        total = n_envs * n_steps * FRAME_SKIP  # physics steps
        ctrl = n_envs * n_steps  # control steps
        print(f'  {ctrl/elapsed:,.0f} control_sps, '
              f'{total/elapsed:,.0f} physics_sps ({elapsed:.2f}s)')
        print(f'  mean reward: {reward.mean().item():.3f}')
        # Reset
        env.reset()
        # Try a full episode
        ep_rewards = []
        for _ in range(MAX_EP_LEN):
            action = torch.rand(n_envs, ACT_DIM, device='cuda:0') * 2 - 1
            obs, reward, done, info = env.step(action)
            ep_rewards.append(reward.mean().item())
        print(f'  mean ep reward: {sum(ep_rewards)/len(ep_rewards):.3f}')
        print(f'  success_once rate: {info["success_once"].mean().item():.3f}')

        del env
        torch.cuda.empty_cache()


if __name__ == '__main__':
    benchmark()
