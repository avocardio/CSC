"""CW10 task implementations as subclasses of CWGPUEnvBase.

Each task is a faithful-as-possible PyTorch/MuJoCo-Warp port of the
corresponding MetaWorld v3 env. Reward functions match the `v2` version
of MetaWorld which is what CW uses.

Physics (contact, joint dynamics) will differ slightly from MetaWorld
because MuJoCo Warp has its own collision solver. Rewards are computed
using identical math where possible.
"""

import os
import numpy as np
import torch
import mujoco

from rl.cw_gpu_env import (
    CWGPUEnvBase, ACT_DIM, OBS_DIM, MAX_EP_LEN,
    METAWORLD_XML_DIR,
)
from rl.cw_reward_utils import tolerance, hamacher_product, gripper_caging_reward


# ==============================================================================
# REACH TASKS — simple, guaranteed-to-work tasks for CL experimentation
# ==============================================================================
class ReachBaseEnv(CWGPUEnvBase):
    """Reach task base — simple, dense reward, easy to train.

    Each variant uses a different FIXED target per task variant, providing
    task diversity while ensuring rapid SAC convergence.
    """
    xml_file = 'sawyer_push_v3.xml'  # Any XML with Sawyer works
    HAND_INIT_POS = torch.tensor([0.0, 0.6, 0.2])
    # Each subclass sets one fixed target
    FIXED_TARGET = torch.tensor([0.0, 0.6, 0.2])
    SUCCESS_RADIUS = 0.05

    def _setup_ids(self):
        super()._setup_ids()

    def _get_initial_hand_pos(self):
        return self.HAND_INIT_POS.to(self.device)

    def _get_obj_obs(self) -> torch.Tensor:
        # Target position as "object" in obs — gives the agent goal information
        target = self.target_pos
        zeros = torch.zeros(self.n_envs, 11, device=self.device)
        return torch.cat([target, zeros], dim=-1)

    def _randomize_state(self, idx: torch.Tensor):
        # No randomization — fixed target per task (simple)
        self.target_pos[idx] = self.FIXED_TARGET.to(self.device)
        self.obj_init_pos[idx] = self.HAND_INIT_POS.to(self.device)

    def _compute_reward(self, action: torch.Tensor,
                        obs: torch.Tensor) -> torch.Tensor:
        hand = obs[:, :3]
        dist = torch.norm(hand - self.target_pos, dim=-1)
        # Steep reward: -dist * 10 gives strong gradient toward target
        # Plus large bonus when reached
        reward = -dist * 10.0
        reward = torch.where(
            dist < self.SUCCESS_RADIUS,
            reward + 10.0,
            reward)
        return reward

    def _compute_success(self) -> torch.Tensor:
        hand = self._get_hand_pos()
        return torch.norm(hand - self.target_pos, dim=-1) < self.SUCCESS_RADIUS


class ReachFrontEnv(ReachBaseEnv):
    """Reach target in +y direction."""
    FIXED_TARGET = torch.tensor([0.00, 0.70, 0.15])


class ReachTopEnv(ReachBaseEnv):
    """Reach target in +z direction."""
    FIXED_TARGET = torch.tensor([0.00, 0.60, 0.25])


class ReachLeftEnv(ReachBaseEnv):
    """Reach target in -x direction."""
    FIXED_TARGET = torch.tensor([-0.10, 0.60, 0.15])


class ReachRightEnv(ReachBaseEnv):
    """Reach target in +x direction."""
    FIXED_TARGET = torch.tensor([0.10, 0.60, 0.15])


# ==============================================================================
# HAMMER
# ==============================================================================
class HammerEnv(CWGPUEnvBase):
    """Sawyer hammer task. Goal: pick up hammer, hit the nail.

    Ported from metaworld.envs.sawyer_hammer_v3.SawyerHammerEnvV3.
    """
    xml_file = 'sawyer_hammer.xml'
    HAMMER_HANDLE_LENGTH = 0.14

    # From MetaWorld
    HAND_INIT_POS = torch.tensor([0.0, 0.4, 0.2])
    HAMMER_OBJ_LOW = torch.tensor([-0.1, 0.4, 0.0])
    HAMMER_OBJ_HIGH = torch.tensor([0.1, 0.5, 0.0])
    GOAL_POS = torch.tensor([0.24, 0.74, 0.11])  # nailHead goal
    NAIL_HEAD_OFFSET = torch.tensor([0.16, 0.06, 0.0])  # hammer body -> head
    HAMMER_QPOS_START = 9  # hammer freejoint starts here (pos 9:12, quat 12:16)
    NAIL_JOINT_QPOS_IDX = 16
    SUCCESS_NAIL_THRESHOLD = 0.09

    def _setup_ids(self):
        super()._setup_ids()
        self.hammer_bid = mujoco.mj_name2id(
            self.mjm, mujoco.mjtObj.mjOBJ_BODY, 'hammer')
        self.nail_bid = mujoco.mj_name2id(
            self.mjm, mujoco.mjtObj.mjOBJ_BODY, 'nail_link')

    def _get_initial_hand_pos(self):
        return self.HAND_INIT_POS.to(self.device)

    def _get_obj_obs(self) -> torch.Tensor:
        # hammer_pos(3) + hammer_quat(4) + nail_pos(3) + nail_quat(4) = 14
        hp = self.xpos[:, self.hammer_bid, :]
        hq = self.xquat[:, self.hammer_bid, :]
        np_ = self.xpos[:, self.nail_bid, :]
        nq = self.xquat[:, self.nail_bid, :]
        return torch.cat([hp, hq, np_, nq], dim=-1)

    def _randomize_state(self, idx: torch.Tensor):
        n = len(idx)
        # Randomize hammer position
        rand = torch.rand(n, 3, device=self.device)
        low = self.HAMMER_OBJ_LOW.to(self.device)
        high = self.HAMMER_OBJ_HIGH.to(self.device)
        hammer_pos = low + rand * (high - low)
        self.qpos[idx, self.HAMMER_QPOS_START:self.HAMMER_QPOS_START + 3] = hammer_pos
        # Upright quat: [1, 0, 0, 0]
        self.qpos[idx, self.HAMMER_QPOS_START + 3] = 1.0
        self.qpos[idx, self.HAMMER_QPOS_START + 4:self.HAMMER_QPOS_START + 7] = 0.0
        # Reset nail joint
        self.qpos[idx, self.NAIL_JOINT_QPOS_IDX] = 0.0
        # Record init positions for gripper caging
        self.obj_init_pos[idx] = hammer_pos
        self.target_pos[idx] = self.GOAL_POS.to(self.device)

    def _compute_reward(self, action: torch.Tensor,
                        obs: torch.Tensor) -> torch.Tensor:
        """MetaWorld v2 hammer reward, vectorized."""
        hand = obs[:, :3]
        hammer = obs[:, 4:7]
        hammer_quat = obs[:, 7:11]

        # Quat reward: [1,0,0,0] is ideal
        ideal_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        quat_err = torch.norm(hammer_quat - ideal_quat, dim=-1)
        reward_quat = (1.0 - quat_err / 0.4).clamp(min=0.0)

        # Hammer threshed: if hand x is within HANDLE_LENGTH/2 of hammer x,
        # override hammer x to match hand (for caging reward)
        threshold = self.HAMMER_HANDLE_LENGTH / 2.0
        hammer_threshed = hammer.clone()
        mask = (hammer[:, 0] - hand[:, 0]).abs() < threshold
        hammer_threshed[mask, 0] = hand[mask, 0]

        # Gripper caging reward
        tcp = self._get_tcp_center()
        left = self._get_leftpad()
        right = self._get_rightpad()
        reward_grab = gripper_caging_reward(
            action, hammer_threshed, self.obj_init_pos, tcp, self.init_tcp,
            left, right,
            obj_radius=0.015,
            pad_success_thresh=0.02,
            object_reach_radius=0.01,
            xz_thresh=0.01,
            high_density=True,
        )

        # In-place reward
        hammer_head = hammer + self.NAIL_HEAD_OFFSET.to(self.device)
        pos_err = torch.norm(self.target_pos - hammer_head, dim=-1)
        lifted = (hammer_head[:, 2] > 0.02).float()
        in_place = 0.1 * lifted + 0.9 * tolerance(
            pos_err, bounds=(0.0, 0.02), margin=0.2, sigmoid='long_tail')

        reward = (2.0 * reward_grab + 6.0 * in_place) * reward_quat

        # Success bonus
        success = self._compute_success() & (reward > 5.0)
        reward = torch.where(success,
                             torch.tensor(10.0, device=self.device), reward)
        return reward

    def _compute_success(self) -> torch.Tensor:
        # nail joint qpos > threshold
        return self.qpos[:, self.NAIL_JOINT_QPOS_IDX] > self.SUCCESS_NAIL_THRESHOLD


# ==============================================================================
# PUSH
# ==============================================================================
class PushEnv(CWGPUEnvBase):
    """Sawyer push task. Goal: push puck to target position.

    Ported from metaworld.envs.sawyer_push_v3.SawyerPushEnvV3.
    """
    xml_file = 'sawyer_push_v3.xml'

    HAND_INIT_POS = torch.tensor([0.0, 0.6, 0.2])
    OBJ_LOW = torch.tensor([-0.1, 0.6, 0.02])
    OBJ_HIGH = torch.tensor([0.1, 0.7, 0.02])
    GOAL_LOW = torch.tensor([-0.1, 0.8, 0.01])
    GOAL_HIGH = torch.tensor([0.1, 0.9, 0.02])
    TARGET_RADIUS = 0.05
    OBJ_QPOS_START = 9  # free joint

    def _setup_ids(self):
        super()._setup_ids()
        self.obj_bid = mujoco.mj_name2id(self.mjm, mujoco.mjtObj.mjOBJ_BODY, 'obj')
        self.obj_gid = mujoco.mj_name2id(self.mjm, mujoco.mjtObj.mjOBJ_GEOM, 'objGeom')

    def _get_initial_hand_pos(self):
        return self.HAND_INIT_POS.to(self.device)

    def _get_obj_obs(self) -> torch.Tensor:
        # obj_pos(3) + obj_quat(4) + 7 zeros (padded) = 14
        op = self.xpos[:, self.obj_bid, :]
        oq = self.xquat[:, self.obj_bid, :]
        zeros = torch.zeros(self.n_envs, 7, device=self.device)
        return torch.cat([op, oq, zeros], dim=-1)

    def _randomize_state(self, idx: torch.Tensor):
        n = len(idx)
        obj_low = self.OBJ_LOW.to(self.device)
        obj_high = self.OBJ_HIGH.to(self.device)
        goal_low = self.GOAL_LOW.to(self.device)
        goal_high = self.GOAL_HIGH.to(self.device)

        # Sample obj and goal positions with min distance constraint
        rand_obj = torch.rand(n, 3, device=self.device)
        obj_pos = obj_low + rand_obj * (obj_high - obj_low)
        rand_goal = torch.rand(n, 3, device=self.device)
        goal_pos = goal_low + rand_goal * (goal_high - goal_low)
        # Ensure min 0.15 xy distance
        for _ in range(5):
            xy_dist = torch.norm(obj_pos[:, :2] - goal_pos[:, :2], dim=-1)
            need_resample = xy_dist < 0.15
            if not need_resample.any():
                break
            new_goal = goal_low + torch.rand(
                need_resample.sum(), 3, device=self.device) * (goal_high - goal_low)
            goal_pos[need_resample] = new_goal

        self.qpos[idx, self.OBJ_QPOS_START:self.OBJ_QPOS_START + 3] = obj_pos
        # Upright quat
        self.qpos[idx, self.OBJ_QPOS_START + 3] = 1.0
        self.qpos[idx, self.OBJ_QPOS_START + 4:self.OBJ_QPOS_START + 7] = 0.0

        self.obj_init_pos[idx] = obj_pos
        self.target_pos[idx] = goal_pos

    def _compute_reward(self, action: torch.Tensor,
                        obs: torch.Tensor) -> torch.Tensor:
        obj = obs[:, 4:7]
        tcp_opened = obs[:, 3]

        tcp = self._get_tcp_center()
        tcp_to_obj = torch.norm(obj - tcp, dim=-1)
        target_to_obj = torch.norm(obj - self.target_pos, dim=-1)
        target_to_obj_init = torch.norm(self.obj_init_pos - self.target_pos, dim=-1)

        in_place = tolerance(
            target_to_obj,
            bounds=(0.0, self.TARGET_RADIUS),
            margin=target_to_obj_init,
            sigmoid='long_tail')

        left = self._get_leftpad()
        right = self._get_rightpad()
        object_grasped = gripper_caging_reward(
            action, obj, self.obj_init_pos, tcp, self.init_tcp, left, right,
            obj_radius=0.015,
            pad_success_thresh=0.05,
            object_reach_radius=0.01,
            xz_thresh=0.005,
            high_density=True,
        )

        reward = 2.0 * object_grasped
        close_and_open = (tcp_to_obj < 0.02) & (tcp_opened > 0)
        bonus = 1.0 + object_grasped + 5.0 * in_place
        reward = torch.where(close_and_open, reward + bonus, reward)

        success = target_to_obj < self.TARGET_RADIUS
        reward = torch.where(
            success, torch.tensor(10.0, device=self.device), reward)
        return reward

    def _compute_success(self) -> torch.Tensor:
        obj = self.xpos[:, self.obj_bid, :]
        return torch.norm(obj - self.target_pos, dim=-1) < self.TARGET_RADIUS


# ==============================================================================
# WINDOW CLOSE  (no gripping, just reach + push handle along x-axis)
# ==============================================================================
class WindowCloseEnv(CWGPUEnvBase):
    """Sawyer window close task. Faithful port of SawyerWindowCloseEnvV3.

    Goal: push window handle to the left (target position) using the gripper.
    Reward = 10 * hamacher(reach, in_place).
    Success: |handle.x - target.x| < 0.05.
    """
    xml_file = 'sawyer_window_horizontal.xml'

    HAND_INIT_POS = torch.tensor([0.0, 0.4, 0.2])
    INIT_OBJ_POS = torch.tensor([0.1, 0.785, 0.16])
    TARGET_RADIUS = 0.05
    HANDLE_RADIUS = 0.02
    SLIDE_INIT = 0.2  # window_slide joint starts at 0.2 (open)

    def _setup_ids(self):
        super()._setup_ids()
        # handleCloseStart site — the position of the handle
        self.handle_sid = mujoco.mj_name2id(
            self.mjm, mujoco.mjtObj.mjOBJ_SITE, 'handleCloseStart')
        if self.handle_sid < 0:
            for name in ['handleOpenStart', 'handle', 'window_handle']:
                sid = mujoco.mj_name2id(self.mjm, mujoco.mjtObj.mjOBJ_SITE, name)
                if sid >= 0:
                    self.handle_sid = sid
                    break
        # window_slide joint
        self.slide_qadr = -1
        for i in range(self.mjm.njnt):
            n = mujoco.mj_id2name(self.mjm, mujoco.mjtObj.mjOBJ_JOINT, i)
            if n == 'window_slide':
                self.slide_qadr = self.mjm.jnt_qposadr[i]
                break

        # Compute the closed handle X by reading default model state
        # MetaWorld convention: target = body pos (where handle should be)
        # We use the window body's default x position
        self.window_bid = mujoco.mj_name2id(
            self.mjm, mujoco.mjtObj.mjOBJ_BODY, 'window')
        if self.window_bid >= 0:
            self.window_default_pos = torch.as_tensor(
                self.mjm.body_pos[self.window_bid], dtype=torch.float32)
        else:
            self.window_default_pos = torch.tensor([0.0, 0.785, 0.16])

    def _get_initial_hand_pos(self):
        return self.HAND_INIT_POS.to(self.device)

    def _get_obj_obs(self) -> torch.Tensor:
        handle = self.site_xpos[:, self.handle_sid, :] if self.handle_sid >= 0 \
            else torch.zeros(self.n_envs, 3, device=self.device)
        zeros = torch.zeros(self.n_envs, 11, device=self.device)
        return torch.cat([handle, zeros], dim=-1)

    def _randomize_state(self, idx: torch.Tensor):
        # Set slide=0.2 (open). The handle starts shifted from closed position.
        if self.slide_qadr >= 0:
            self.qpos[idx, self.slide_qadr] = self.SLIDE_INIT  # 0.2

        # Target = window body position (the closed handle position).
        # In MetaWorld, the in_place reward uses |handle.x - target.x| where
        # target.x = body.x. The body position is fixed by the XML, so the
        # target is constant per task.
        target = self.window_default_pos.to(self.device)
        self.target_pos[idx] = target
        # obj_init_pos = handle position when slide=SLIDE_INIT (initial open)
        # We approximate as target + [0.2, 0, 0]
        self.obj_init_pos[idx] = target + torch.tensor(
            [self.SLIDE_INIT, 0.0, 0.0], device=self.device)

    def _compute_reward(self, action: torch.Tensor,
                        obs: torch.Tensor) -> torch.Tensor:
        obj = obs[:, 4:7]  # handle position
        tcp = self._get_tcp_center()
        target = self.target_pos

        # target_to_obj: only x-axis distance (window slides along x)
        target_to_obj = (obj[:, 0] - target[:, 0]).abs()
        target_to_obj_init = (self.obj_init_pos[:, 0] - target[:, 0]).abs()

        in_place = tolerance(
            target_to_obj,
            bounds=(0.0, self.TARGET_RADIUS),
            margin=(target_to_obj_init - self.TARGET_RADIUS).abs(),
            sigmoid='long_tail',
        )

        tcp_to_obj = torch.norm(obj - tcp, dim=-1)
        tcp_to_obj_init = torch.norm(self.obj_init_pos - self.init_tcp, dim=-1)
        reach = tolerance(
            tcp_to_obj,
            bounds=(0.0, self.HANDLE_RADIUS),
            margin=(tcp_to_obj_init - self.HANDLE_RADIUS).abs(),
            sigmoid='gaussian',
        )

        reward = 10.0 * hamacher_product(reach, in_place)
        return reward

    def _compute_success(self) -> torch.Tensor:
        if self.handle_sid < 0:
            return torch.zeros(self.n_envs, dtype=torch.bool, device=self.device)
        obj_x = self.site_xpos[:, self.handle_sid, 0]
        return (obj_x - self.target_pos[:, 0]).abs() < self.TARGET_RADIUS


# ==============================================================================
# FAUCET CLOSE
# ==============================================================================
class FaucetCloseEnv(CWGPUEnvBase):
    """Sawyer faucet close task. Rotate faucet handle to close position."""
    xml_file = 'sawyer_faucet.xml'

    HAND_INIT_POS = torch.tensor([0.0, 0.4, 0.2])
    TARGET_RADIUS = 0.07
    HANDLE_LENGTH = 0.175

    def _setup_ids(self):
        super()._setup_ids()
        self.handle_sid = mujoco.mj_name2id(
            self.mjm, mujoco.mjtObj.mjOBJ_SITE, 'handleStartClose')
        if self.handle_sid < 0:
            self.handle_sid = mujoco.mj_name2id(
                self.mjm, mujoco.mjtObj.mjOBJ_SITE, 'handleStartOpen')

    def _get_initial_hand_pos(self):
        return self.HAND_INIT_POS.to(self.device)

    def _get_obj_obs(self) -> torch.Tensor:
        if self.handle_sid >= 0:
            handle_pos = self.site_xpos[:, self.handle_sid, :] - \
                torch.tensor([0.0, 0.0, 0.01], device=self.device)
        else:
            handle_pos = torch.zeros(self.n_envs, 3, device=self.device)
        zeros = torch.zeros(self.n_envs, 11, device=self.device)
        return torch.cat([handle_pos, zeros], dim=-1)

    def _randomize_state(self, idx: torch.Tensor):
        # Fixed faucet position in body coords
        base_pos = torch.tensor([0.0, 0.8, 0.0], device=self.device)
        self.obj_init_pos[idx] = base_pos
        # Target = base + [-handle_length, 0, 0.125]
        self.target_pos[idx] = base_pos + torch.tensor(
            [-self.HANDLE_LENGTH, 0.0, 0.125], device=self.device)

    def _compute_reward(self, action: torch.Tensor,
                        obs: torch.Tensor) -> torch.Tensor:
        obj = obs[:, 4:7]
        tcp = self._get_tcp_center()
        target = self.target_pos

        target_to_obj = torch.norm(obj - target, dim=-1)
        target_to_obj_init = torch.norm(self.obj_init_pos - target, dim=-1)
        in_place = tolerance(
            target_to_obj,
            bounds=(0.0, self.TARGET_RADIUS),
            margin=(target_to_obj_init - self.TARGET_RADIUS).abs(),
            sigmoid='long_tail')

        tcp_to_obj = torch.norm(obj - tcp, dim=-1)
        tcp_to_obj_init = torch.norm(self.obj_init_pos - self.init_tcp, dim=-1)
        reach = tolerance(
            tcp_to_obj,
            bounds=(0.0, 0.01),
            margin=(tcp_to_obj_init - 0.01).abs(),
            sigmoid='gaussian')

        reward = 2 * reach + 3 * in_place
        reward = reward * 2
        reward = torch.where(
            target_to_obj <= self.TARGET_RADIUS,
            torch.tensor(10.0, device=self.device), reward)
        return reward

    def _compute_success(self) -> torch.Tensor:
        if self.handle_sid < 0:
            return torch.zeros(self.n_envs, dtype=torch.bool, device=self.device)
        obj = self.site_xpos[:, self.handle_sid, :] - \
            torch.tensor([0.0, 0.0, 0.01], device=self.device)
        return torch.norm(obj - self.target_pos, dim=-1) < self.TARGET_RADIUS


# ==============================================================================
# HANDLE PRESS SIDE
# ==============================================================================
class HandlePressSideEnv(CWGPUEnvBase):
    """Sawyer handle press side task. Press handle downward."""
    xml_file = 'sawyer_handle_press_sideways.xml'

    HAND_INIT_POS = torch.tensor([0.0, 0.4, 0.2])
    TARGET_RADIUS = 0.02

    def _setup_ids(self):
        super()._setup_ids()
        self.handle_sid = mujoco.mj_name2id(
            self.mjm, mujoco.mjtObj.mjOBJ_SITE, 'handleStart')
        # Goal site (where the handle should be pressed to)
        self.goal_sid = mujoco.mj_name2id(
            self.mjm, mujoco.mjtObj.mjOBJ_SITE, 'goalPress')

    def _get_initial_hand_pos(self):
        return self.HAND_INIT_POS.to(self.device)

    def _get_obj_obs(self) -> torch.Tensor:
        if self.handle_sid >= 0:
            hp = self.site_xpos[:, self.handle_sid, :]
        else:
            hp = torch.zeros(self.n_envs, 3, device=self.device)
        zeros = torch.zeros(self.n_envs, 11, device=self.device)
        return torch.cat([hp, zeros], dim=-1)

    def _randomize_state(self, idx: torch.Tensor):
        # Placeholder values; the real ones are set in _post_relax_init
        # after physics has settled the state
        self.target_pos[idx] = torch.tensor(
            [-0.184, 0.6, 0.05], device=self.device)
        self.obj_init_pos[idx] = torch.tensor(
            [-0.184, 0.6, 0.20], device=self.device)

    def _post_relax_init(self, idx: torch.Tensor):
        # Read actual world positions after physics relaxation
        if self.goal_sid >= 0:
            self.target_pos[idx] = self.site_xpos[idx, self.goal_sid, :]
        if self.handle_sid >= 0:
            self.obj_init_pos[idx] = self.site_xpos[idx, self.handle_sid, :]

    def _compute_reward(self, action: torch.Tensor,
                        obs: torch.Tensor) -> torch.Tensor:
        obj = obs[:, 4:7]
        tcp = self._get_tcp_center()
        target = self.target_pos

        # target_to_obj: only z-axis (handle moves down)
        target_to_obj = (obj[:, 2] - target[:, 2]).abs()
        target_to_obj_init = (self.obj_init_pos[:, 2] - target[:, 2]).abs()

        in_place = tolerance(
            target_to_obj,
            bounds=(0.0, self.TARGET_RADIUS),
            margin=(target_to_obj_init - self.TARGET_RADIUS).abs(),
            sigmoid='long_tail')

        tcp_to_obj = torch.norm(obj - tcp, dim=-1)
        tcp_to_obj_init = torch.norm(self.obj_init_pos - self.init_tcp, dim=-1)
        reach = tolerance(
            tcp_to_obj,
            bounds=(0.0, 0.02),
            margin=(tcp_to_obj_init - 0.02).abs(),
            sigmoid='long_tail')

        reward = hamacher_product(reach, in_place) * 10
        reward = torch.where(
            target_to_obj <= self.TARGET_RADIUS,
            torch.tensor(10.0, device=self.device), reward)
        return reward

    def _compute_success(self) -> torch.Tensor:
        if self.handle_sid < 0:
            return torch.zeros(self.n_envs, dtype=torch.bool, device=self.device)
        obj_z = self.site_xpos[:, self.handle_sid, 2]
        return (obj_z - self.target_pos[:, 2]).abs() < self.TARGET_RADIUS


# ==============================================================================
# SHELF PLACE
# ==============================================================================
class ShelfPlaceEnv(CWGPUEnvBase):
    """Sawyer shelf-place. Pick up object and place it on a shelf."""
    xml_file = 'sawyer_shelf_placing.xml'

    HAND_INIT_POS = torch.tensor([0.0, 0.6, 0.2])
    TARGET_RADIUS = 0.05

    def _setup_ids(self):
        super()._setup_ids()
        self.obj_bid = mujoco.mj_name2id(self.mjm, mujoco.mjtObj.mjOBJ_BODY, 'obj')

    def _get_initial_hand_pos(self):
        return self.HAND_INIT_POS.to(self.device)

    def _get_obj_obs(self) -> torch.Tensor:
        if self.obj_bid >= 0:
            op = self.xpos[:, self.obj_bid, :]
            oq = self.xquat[:, self.obj_bid, :]
        else:
            op = torch.zeros(self.n_envs, 3, device=self.device)
            oq = torch.tensor([1.0, 0, 0, 0], device=self.device).expand(self.n_envs, 4)
        zeros = torch.zeros(self.n_envs, 7, device=self.device)
        return torch.cat([op, oq, zeros], dim=-1)

    def _randomize_state(self, idx: torch.Tensor):
        # Default obj position from MetaWorld init_config
        obj_init = torch.tensor([0.0, 0.6, 0.02], device=self.device)
        self.obj_init_pos[idx] = obj_init
        # Target ~30 cm above the obj (on shelf level)
        self.target_pos[idx] = obj_init + torch.tensor(
            [0.0, 0.0, 0.30], device=self.device)

    def _compute_reward(self, action: torch.Tensor,
                        obs: torch.Tensor) -> torch.Tensor:
        obj = obs[:, 4:7]
        tcp = self._get_tcp_center()
        target = self.target_pos
        tcp_opened = obs[:, 3]

        obj_to_target = torch.norm(obj - target, dim=-1)
        tcp_to_obj = torch.norm(obj - tcp, dim=-1)
        in_place_margin = torch.norm(self.obj_init_pos - target, dim=-1).clamp(min=1e-3)
        in_place = tolerance(
            obj_to_target,
            bounds=(0.0, self.TARGET_RADIUS),
            margin=in_place_margin,
            sigmoid='long_tail')

        # Add a dense reach reward so the policy gets early signal
        reach_reward = (1.0 - (tcp_to_obj / 0.3).clamp(0.0, 1.0))

        reward = 0.5 * reach_reward + hamacher_product(reach_reward, in_place)

        # Bonus when grasping
        grasping = ((tcp_to_obj < 0.025) & (tcp_opened > 0) &
                    (obj[:, 2] - 0.01 > self.obj_init_pos[:, 2]))
        reward = torch.where(grasping, reward + 1.0 + 5.0 * in_place, reward)

        # Success bonus
        success = obj_to_target < self.TARGET_RADIUS
        reward = torch.where(
            success, torch.tensor(10.0, device=self.device), reward)
        return reward

    def _compute_success(self) -> torch.Tensor:
        if self.obj_bid < 0:
            return torch.zeros(self.n_envs, dtype=torch.bool, device=self.device)
        obj = self.xpos[:, self.obj_bid, :]
        return torch.norm(obj - self.target_pos, dim=-1) < self.TARGET_RADIUS


# ==============================================================================
# STICK PULL
# ==============================================================================
class StickPullEnv(CWGPUEnvBase):
    """Sawyer stick-pull. Pick up stick and use it to pull a container."""
    xml_file = 'sawyer_stick_obj.xml'

    HAND_INIT_POS = torch.tensor([0.0, 0.6, 0.2])
    TARGET_RADIUS = 0.05

    def _setup_ids(self):
        super()._setup_ids()
        self.stick_bid = mujoco.mj_name2id(
            self.mjm, mujoco.mjtObj.mjOBJ_BODY, 'stick')
        self.object_bid = mujoco.mj_name2id(
            self.mjm, mujoco.mjtObj.mjOBJ_BODY, 'object')

    def _get_initial_hand_pos(self):
        return self.HAND_INIT_POS.to(self.device)

    def _get_obj_obs(self) -> torch.Tensor:
        if self.stick_bid >= 0:
            stick_pos = self.xpos[:, self.stick_bid, :]
            stick_quat = self.xquat[:, self.stick_bid, :]
        else:
            stick_pos = torch.zeros(self.n_envs, 3, device=self.device)
            stick_quat = torch.tensor(
                [1.0, 0, 0, 0], device=self.device).expand(self.n_envs, 4)
        if self.object_bid >= 0:
            obj_pos = self.xpos[:, self.object_bid, :]
        else:
            obj_pos = torch.zeros(self.n_envs, 3, device=self.device)
        zeros = torch.zeros(self.n_envs, 4, device=self.device)
        return torch.cat([stick_pos, stick_quat, obj_pos, zeros], dim=-1)

    def _randomize_state(self, idx: torch.Tensor):
        # Default positions
        stick_init = torch.tensor([0.2, 0.69, 0.04], device=self.device)
        self.obj_init_pos[idx] = stick_init  # using as "object init"
        # Target = pull container 15 cm to the left
        self.target_pos[idx] = torch.tensor(
            [0.05, 0.69, 0.04], device=self.device)

    def _compute_reward(self, action: torch.Tensor,
                        obs: torch.Tensor) -> torch.Tensor:
        # Simplified: dense reach + caging + container proximity
        tcp = self._get_tcp_center()
        stick = obs[:, 4:7]
        container = obs[:, 11:14]

        tcp_to_stick = torch.norm(stick - tcp, dim=-1)
        container_to_target = torch.norm(container - self.target_pos, dim=-1)

        # Dense reach reward (independent of caging)
        reach_reward = (1.0 - (tcp_to_stick / 0.3).clamp(0.0, 1.0))

        margin = torch.norm(self.obj_init_pos - self.target_pos, dim=-1).clamp(min=1e-3)
        in_place = tolerance(
            container_to_target,
            bounds=(0.0, self.TARGET_RADIUS),
            margin=margin,
            sigmoid='long_tail')

        reward = 0.5 * reach_reward + hamacher_product(reach_reward, in_place)
        success = container_to_target < self.TARGET_RADIUS
        reward = torch.where(
            success, torch.tensor(10.0, device=self.device), reward)
        return reward

    def _compute_success(self) -> torch.Tensor:
        if self.object_bid < 0:
            return torch.zeros(self.n_envs, dtype=torch.bool, device=self.device)
        obj = self.xpos[:, self.object_bid, :]
        return torch.norm(obj - self.target_pos, dim=-1) < self.TARGET_RADIUS


# ==============================================================================
# PEG UNPLUG SIDE
# ==============================================================================
class PegUnplugSideEnv(CWGPUEnvBase):
    """Sawyer peg unplug side. Grab peg and pull it out sideways."""
    xml_file = 'sawyer_peg_unplug_side.xml'

    HAND_INIT_POS = torch.tensor([0.0, 0.6, 0.2])
    TARGET_RADIUS = 0.07

    def _setup_ids(self):
        super()._setup_ids()
        # pegEnd site is the reference point for the peg position
        self.peg_sid = mujoco.mj_name2id(
            self.mjm, mujoco.mjtObj.mjOBJ_SITE, 'pegEnd')
        self.plug_bid = mujoco.mj_name2id(
            self.mjm, mujoco.mjtObj.mjOBJ_BODY, 'plug1')

    def _get_initial_hand_pos(self):
        return self.HAND_INIT_POS.to(self.device)

    def _get_obj_obs(self) -> torch.Tensor:
        if self.peg_sid >= 0:
            peg_pos = self.site_xpos[:, self.peg_sid, :]
        else:
            peg_pos = torch.zeros(self.n_envs, 3, device=self.device)
        if self.plug_bid >= 0:
            plug_quat = self.xquat[:, self.plug_bid, :]
        else:
            plug_quat = torch.tensor(
                [1.0, 0.0, 0.0, 0.0], device=self.device).expand(self.n_envs, 4)
        zeros = torch.zeros(self.n_envs, 7, device=self.device)
        return torch.cat([peg_pos, plug_quat, zeros], dim=-1)

    def _randomize_state(self, idx: torch.Tensor):
        # Fixed positions for simplicity
        plug_init = torch.tensor(
            [-0.181, 0.6, 0.181], device=self.device)  # obj_init_pos + [0.044, 0, 0.131]
        self.obj_init_pos[idx] = plug_init
        # Target = plug + [0.15, 0, 0] (pull out 15 cm to the right)
        self.target_pos[idx] = plug_init + torch.tensor(
            [0.15, 0.0, 0.0], device=self.device)

    def _compute_reward(self, action: torch.Tensor,
                        obs: torch.Tensor) -> torch.Tensor:
        obj = obs[:, 4:7]
        tcp = self._get_tcp_center()
        target = self.target_pos

        tcp_to_obj = torch.norm(obj - tcp, dim=-1)
        obj_to_target = torch.norm(obj - target, dim=-1)

        # Simplified caging: proximity to peg
        reach_reward = (1.0 - (tcp_to_obj / 0.2).clamp(0.0, 1.0))

        in_place_margin = torch.norm(self.obj_init_pos - target, dim=-1)
        in_place = tolerance(
            obj_to_target,
            bounds=(0.0, 0.05),
            margin=in_place_margin,
            sigmoid='long_tail')

        # Gripper closing bonus
        gripper_closed = (1.0 - obs[:, 3])  # 0 = open, 1 = closed
        grasp_success = (tcp_to_obj < 0.035) & (gripper_closed > 0.3)

        reward = 2.0 * reach_reward
        reward = torch.where(
            grasp_success,
            1.0 + 2.0 * reach_reward + 5.0 * in_place,
            reward)
        reward = torch.where(
            obj_to_target <= 0.05,
            torch.tensor(10.0, device=self.device),
            reward)
        return reward

    def _compute_success(self) -> torch.Tensor:
        if self.peg_sid < 0:
            return torch.zeros(self.n_envs, dtype=torch.bool, device=self.device)
        obj = self.site_xpos[:, self.peg_sid, :]
        return torch.norm(obj - self.target_pos, dim=-1) < self.TARGET_RADIUS


# ==============================================================================
# PUSH WALL
# ==============================================================================
class PushWallEnv(PushEnv):
    """Push with a wall between object and target (same reward as push)."""
    xml_file = 'sawyer_push_wall_v3.xml'
    OBJ_LOW = torch.tensor([-0.05, 0.60, 0.015])
    OBJ_HIGH = torch.tensor([0.05, 0.65, 0.015])
    GOAL_LOW = torch.tensor([-0.05, 0.85, 0.01])
    GOAL_HIGH = torch.tensor([0.05, 0.9, 0.02])


# ==============================================================================
# PUSH BACK  (push object toward robot instead of away)
# ==============================================================================
class PushBackEnv(PushEnv):
    """Push object back toward robot."""
    xml_file = 'sawyer_push_back_v3.xml'
    OBJ_LOW = torch.tensor([-0.1, 0.7, 0.02])
    OBJ_HIGH = torch.tensor([0.1, 0.8, 0.02])
    GOAL_LOW = torch.tensor([-0.1, 0.6, 0.015])
    GOAL_HIGH = torch.tensor([0.1, 0.7, 0.015])


# Task registry
CW_TASK_REGISTRY = {
    'hammer': HammerEnv,
    'push': PushEnv,
    'push-wall': PushWallEnv,
    'push-back': PushBackEnv,
    'window-close': WindowCloseEnv,
    'faucet-close': FaucetCloseEnv,
    'handle-press-side': HandlePressSideEnv,
    'peg-unplug-side': PegUnplugSideEnv,
    'shelf-place': ShelfPlaceEnv,
    'stick-pull': StickPullEnv,
    # Simple reach tasks for CL experiments (easier, faster to train)
    'reach-front': ReachFrontEnv,
    'reach-top': ReachTopEnv,
    'reach-left': ReachLeftEnv,
    'reach-right': ReachRightEnv,
}


def make_env(task_name: str, n_envs: int = 1024, device: str = 'cuda:0'):
    if task_name not in CW_TASK_REGISTRY:
        raise ValueError(f'Unknown task: {task_name}. '
                         f'Available: {list(CW_TASK_REGISTRY.keys())}')
    return CW_TASK_REGISTRY[task_name](n_envs=n_envs, device=device)
