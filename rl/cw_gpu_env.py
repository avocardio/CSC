"""Unified GPU Continual World environment (MuJoCo Warp based).

All CW10 tasks share the same Sawyer robot, action space, and obs format.
They differ in:
- The XML scene file loaded
- The object body/site names
- The reward computation
- The success criterion
- The reset logic (object placements)

This module provides a base class that handles everything common, and
per-task subclasses for the 10 CW tasks.

Key design:
- All state stays on GPU via wp.to_torch views
- Action: 4D [dx, dy, dz, gripper] in [-1, 1]
- Observation: 39D = [hand(3), grip(1), obj_info(14), prev(18), goal(3)]
- Frame skip: 5 physics steps per control step
- Max episode length: 200 (CW convention, NOT 500 MW default)
"""

import os
import numpy as np
import torch
import mujoco
import warp as wp
import mujoco_warp as mjw

from rl.cw_reward_utils import tolerance, hamacher_product, gripper_caging_reward

# Constants matching MetaWorld's SawyerXYZEnv
ACT_DIM = 4
OBS_DIM = 39
FRAME_SKIP = 5
MAX_EP_LEN = 200  # CW convention
ACTION_SCALE = 1.0 / 100.0
MOCAP_LOW = torch.tensor([-0.2, 0.5, 0.06])
MOCAP_HIGH = torch.tensor([0.2, 0.7, 0.6])

import metaworld as _mw
METAWORLD_XML_DIR = os.path.join(os.path.dirname(_mw.__file__), 'assets', 'sawyer_xyz')


class CWGPUEnvBase:
    """Base class for batched GPU CW environments.

    Subclasses must set:
        xml_file: name of xml in METAWORLD_XML_DIR
        obj_body_names: list of body names for objects (positions in obs)
        init_hand_pos: torch.tensor (3,) initial hand position
        hand_init_ctrl: [0]=right_gripper, [1]=left_gripper start values
        max_obj_obs_dim: 14 (padded)
        goal_low, goal_high: sampling range for goal
        obj_low, obj_high: sampling range for initial object position

    Subclasses must implement:
        _randomize_state(idx): reset task-specific state for envs in idx
        _compute_reward(action, obs): return (N,) reward tensor
        _compute_success(): return (N,) bool tensor
    """

    xml_file: str = 'sawyer_hammer.xml'
    max_obj_obs_dim: int = 14

    def __init__(self, n_envs: int = 1024, device: str = 'cuda:0',
                 relaxation_steps: int = 50):
        self.n_envs = n_envs
        self.device = torch.device(device)
        self.relaxation_steps = relaxation_steps

        # Load MuJoCo model
        xml_path = os.path.join(METAWORLD_XML_DIR, self.xml_file)
        self.mjm = mujoco.MjModel.from_xml_path(xml_path)

        # Reset mocap welds (MetaWorld convention)
        for i in range(self.mjm.eq_data.shape[0]):
            if self.mjm.eq_type[i] == mujoco.mjtEq.mjEQ_WELD:
                self.mjm.eq_data[i] = np.array(
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 5.0])

        # Fix for MuJoCo Warp CCD: set all geom margins to 0
        # (MuJoCo Warp doesn't support non-zero margins with NATIVECCD)
        for i in range(self.mjm.ngeom):
            self.mjm.geom_margin[i] = 0.0

        # Set up Warp
        # Increase njmax to avoid nefc overflow warnings during contact-heavy states
        self.mjm.opt.disableflags = 0
        # Ensure enough constraint budget (default is ~64)
        if hasattr(self.mjm.opt, 'njmax'):
            self.mjm.opt.njmax = max(256, self.mjm.opt.njmax)
        # Increase CCD iterations to avoid collision detection warnings
        if hasattr(self.mjm.opt, 'ccd_iterations'):
            self.mjm.opt.ccd_iterations = max(64, self.mjm.opt.ccd_iterations)
        wp.init()
        with wp.ScopedDevice(device):
            self.m = mjw.put_model(self.mjm)
            # make_data supports nconmax/njmax per-world
            try:
                self.d = mjw.make_data(
                    self.mjm, nworld=n_envs, nconmax=512, njmax=512)
            except TypeError:
                self.d = mjw.make_data(self.mjm, nworld=n_envs)

        self._sync_torch_views()

        # Find site/body IDs (set by subclass or common)
        self._setup_ids()

        # Per-env state
        self.mocap_low = MOCAP_LOW.to(self.device)
        self.mocap_high = MOCAP_HIGH.to(self.device)
        self.ep_len = torch.zeros(n_envs, dtype=torch.int32, device=self.device)
        self.prev_obs = torch.zeros(n_envs, 18, device=self.device)
        self.success_once = torch.zeros(n_envs, dtype=torch.bool, device=self.device)
        self.ep_return = torch.zeros(n_envs, device=self.device)
        # For gripper_caging_reward: need init_tcp and obj_init_pos per env
        self.init_tcp = torch.zeros(n_envs, 3, device=self.device)
        self.obj_init_pos = torch.zeros(n_envs, 3, device=self.device)
        self.target_pos = torch.zeros(n_envs, 3, device=self.device)

    def _sync_torch_views(self):
        self.qpos = wp.to_torch(self.d.qpos)
        self.qvel = wp.to_torch(self.d.qvel)
        self.ctrl = wp.to_torch(self.d.ctrl)
        self.mocap_pos = wp.to_torch(self.d.mocap_pos)
        self.mocap_quat = wp.to_torch(self.d.mocap_quat)
        self.site_xpos = wp.to_torch(self.d.site_xpos)
        self.xpos = wp.to_torch(self.d.xpos)
        self.xquat = wp.to_torch(self.d.xquat)

    def _setup_ids(self):
        """Look up body/site IDs. Subclasses can override to add task-specific ones."""
        self.ee_sid = mujoco.mj_name2id(
            self.mjm, mujoco.mjtObj.mjOBJ_SITE, 'endEffector')
        self.right_ee_sid = mujoco.mj_name2id(
            self.mjm, mujoco.mjtObj.mjOBJ_SITE, 'rightEndEffector')
        self.left_ee_sid = mujoco.mj_name2id(
            self.mjm, mujoco.mjtObj.mjOBJ_SITE, 'leftEndEffector')
        # leftpad / rightpad bodies (for gripper caging reward)
        self.leftpad_bid = mujoco.mj_name2id(
            self.mjm, mujoco.mjtObj.mjOBJ_BODY, 'leftpad')
        self.rightpad_bid = mujoco.mj_name2id(
            self.mjm, mujoco.mjtObj.mjOBJ_BODY, 'rightpad')

    # ---- Accessors (GPU views) ----

    def _get_hand_pos(self) -> torch.Tensor:
        return self.site_xpos[:, self.ee_sid, :]

    def _get_gripper_distance(self) -> torch.Tensor:
        right = self.site_xpos[:, self.right_ee_sid, :]
        left = self.site_xpos[:, self.left_ee_sid, :]
        return (torch.norm(right - left, dim=-1) / 0.1).clamp(0.0, 1.0)

    def _get_tcp_center(self) -> torch.Tensor:
        right = self.site_xpos[:, self.right_ee_sid, :]
        left = self.site_xpos[:, self.left_ee_sid, :]
        return (right + left) / 2.0

    def _get_leftpad(self) -> torch.Tensor:
        return self.xpos[:, self.leftpad_bid, :]

    def _get_rightpad(self) -> torch.Tensor:
        return self.xpos[:, self.rightpad_bid, :]

    @torch.no_grad()
    def _physics_step(self, n: int = 1):
        with wp.ScopedDevice(str(self.device)):
            for _ in range(n):
                mjw.step(self.m, self.d)

    @torch.no_grad()
    def _forward(self):
        with wp.ScopedDevice(str(self.device)):
            mjw.forward(self.m, self.d)

    # ---- Task-specific methods (subclasses override) ----

    def _get_obj_positions(self) -> torch.Tensor:
        """Return object positions (N, 3) for observation. Subclass override."""
        raise NotImplementedError

    def _get_obj_quaternions(self) -> torch.Tensor:
        """Return object quaternions (N, 4) for observation. Subclass override."""
        raise NotImplementedError

    def _get_obj_obs(self) -> torch.Tensor:
        """Return (N, 14) padded object info: flat [pos, quat, pos, quat, ...]."""
        raise NotImplementedError

    def _randomize_state(self, idx: torch.Tensor):
        """Randomize per-task state for the given env indices."""
        raise NotImplementedError

    def _post_relax_init(self, idx: torch.Tensor):
        """Hook called after physics relaxation. Default no-op.

        Subclasses override to read actual site/body positions that depend
        on the relaxed physics state."""
        pass

    def _compute_reward(self, action: torch.Tensor,
                        obs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _compute_success(self) -> torch.Tensor:
        raise NotImplementedError

    def _get_initial_hand_pos(self) -> torch.Tensor:
        """Subclass can override. Default: (0, 0.6, 0.2)."""
        return torch.tensor([0.0, 0.6, 0.2], device=self.device)

    def _get_goal_obs(self) -> torch.Tensor:
        """Return (N, 3) goal for observation (uses self.target_pos by default)."""
        return self.target_pos

    # ---- Main API ----

    def _compute_curr_obs(self) -> torch.Tensor:
        hand = self._get_hand_pos()
        grip = self._get_gripper_distance().unsqueeze(-1)
        obj_obs = self._get_obj_obs()  # (N, 14)
        return torch.cat([hand, grip, obj_obs], dim=-1)  # (N, 18)

    def _get_obs(self) -> torch.Tensor:
        curr = self._compute_curr_obs()
        goal = self._get_goal_obs()
        obs = torch.cat([curr, self.prev_obs, goal], dim=-1)
        self.prev_obs = curr
        return obs

    @torch.no_grad()
    def reset(self) -> torch.Tensor:
        """Reset all envs."""
        qpos0 = torch.as_tensor(self.mjm.qpos0, device=self.device).float()
        self.qpos.copy_(qpos0.unsqueeze(0).expand(self.n_envs, -1))
        self.qvel.zero_()

        # Task-specific randomization
        all_idx = torch.arange(self.n_envs, device=self.device)
        self._randomize_state(all_idx)

        # Set mocap to initial hand position
        init_hand = self._get_initial_hand_pos()
        self.mocap_pos[:, 0, :] = init_hand
        self.mocap_quat[:, 0, :] = torch.tensor(
            [1.0, 0.0, 1.0, 0.0], device=self.device)

        # Start with gripper open (-1, 1 per MetaWorld convention)
        self.ctrl[:, 0] = -1.0
        self.ctrl[:, 1] = 1.0

        # Relax physics so the mocap weld pulls the arm into position
        self._physics_step(self.relaxation_steps * FRAME_SKIP)

        # Snapshot init_tcp for gripper_caging_reward
        self.init_tcp = self._get_tcp_center().clone()

        # Hook for subclasses to do task-specific post-relaxation setup
        # (e.g., reading actual site positions for target_pos)
        self._post_relax_init(torch.arange(self.n_envs, device=self.device))

        # Reset episode state
        self.ep_len.zero_()
        self.success_once.zero_()
        self.ep_return.zero_()
        self.prev_obs.zero_()

        curr = self._compute_curr_obs()
        self.prev_obs = curr.clone()
        return torch.cat([curr, curr, self._get_goal_obs()], dim=-1)

    @torch.no_grad()
    def step(self, action: torch.Tensor):
        action = action.clamp(-1, 1)

        # Update mocap position
        new_mocap = self.mocap_pos[:, 0, :] + action[:, :3] * ACTION_SCALE
        new_mocap = torch.max(torch.min(new_mocap, self.mocap_high), self.mocap_low)
        self.mocap_pos[:, 0, :] = new_mocap

        # Gripper control (MetaWorld: ctrl = [action[-1], -action[-1]])
        self.ctrl[:, 0] = action[:, 3]
        self.ctrl[:, 1] = -action[:, 3]

        # Physics step with frame skip
        self._physics_step(FRAME_SKIP)

        obs = self._get_obs()
        reward = self._compute_reward(action, obs)

        success = self._compute_success()
        self.success_once = self.success_once | success

        self.ep_len += 1
        self.ep_return += reward
        done = self.ep_len >= MAX_EP_LEN

        info = {
            'success': success.float(),
            'success_once': self.success_once.float(),
            'ep_return': self.ep_return.clone(),
        }
        return obs, reward, done, info

    @torch.no_grad()
    def auto_reset_step(self, action: torch.Tensor):
        obs, reward, done, info = self.step(action)
        info['real_next_obs'] = obs.clone()
        if done.any():
            self.reset_done(done)
            new_curr = self._compute_curr_obs()
            reset_obs = torch.cat([new_curr, new_curr, self._get_goal_obs()], dim=-1)
            obs = torch.where(done.unsqueeze(-1), reset_obs, obs)
        return obs, reward, done, info

    def close(self):
        """Release resources (compatibility with gym API)."""
        pass

    @torch.no_grad()
    def reset_done(self, done: torch.Tensor):
        if not done.any():
            return
        idx = done.nonzero(as_tuple=False).squeeze(-1)

        # Reset qpos to default + randomize task state
        qpos0 = torch.as_tensor(self.mjm.qpos0, device=self.device).float()
        self.qpos[idx] = qpos0.unsqueeze(0).expand(len(idx), -1)
        self.qvel[idx] = 0.0

        self._randomize_state(idx)

        # Reset mocap for done envs
        init_hand = self._get_initial_hand_pos()
        self.mocap_pos[idx, 0, :] = init_hand
        mocap_q = torch.tensor([1.0, 0.0, 1.0, 0.0], device=self.device)
        self.mocap_quat[idx, 0, :] = mocap_q

        # Episode state
        self.ep_len[idx] = 0
        self.success_once[idx] = False
        self.ep_return[idx] = 0.0
        self.prev_obs[idx] = 0.0

        # Task-specific post-reset (no relaxation here for speed)
        self._post_relax_init(idx)
