# HOI (Human-Object Interaction) Codebase Optimization Tracker

## 🎯 Project Context
This project focuses on RL-based robotics/computer graphics for hand-object manipulation. The goal is to generate full-body motions for articulated objects driven by text prompts using the SMPL-X model.
- **Current Pipeline:** MANO URDF in Isaac Gym -> PPO optimization for manipulation (hooking open doors/drawers) -> Data collection.
- **Future Pipeline:** Use collected parametric data to drive diffusion models for SMPL-X full-body motion synthesis.
- **Core Requirement:** The data collected in simulation MUST be physically consistent (accounting for reaction forces and inertia) so it can transfer smoothly to the full-body kinematic chain.

## 🤖 Guidelines for AI Agent (Claude)
1. **Work Iteratively:** Resolve one task completely, run tests, and confirm stability with the user before moving to the next.
2. **Preserve Physics Authenticity:** Do not use kinematic workarounds (like root teleportation) that break downstream physical consistency.
3. **Log & Communicate:** When modifying reward functions or loss terms, print the before/after logic so the user can verify the math.
4. **Mark Progress:** Update the `[ ]` to `[x]` in this file as tasks are completed.

---

## 📋 Task Backlog (Prioritized)

### Phase 1: Critical Physics & Data Integrity (P0)
- [x] **Task 1.1: Fix Kinematic-Dynamic Mismatch (`object_gym.py`)**
  - **Issue:** Teleporting the MANO root via `set_actor_root_state_tensor_indexed` gives the palm infinite stiffness, breaking PhysX contact forces and ruining SMPL-X transferability.
  - **Action:** Replace teleportation with a force/impedance controller. Apply desired wrist poses as PD targets on a 6-DOF floating joint with finite stiffness.
- [x] **Task 1.2: Ground-Truth Contact Detection (`fast_contact_calc.py` & RL Envs)**
  - **Issue:** FK-based point-cloud distance threshold (15mm) disagrees with actual PhysX solver.
  - **Action:** For the RL simulation loop, switch to Isaac Gym's `acquire_net_contact_force_tensor` or `get_env_rigid_contacts`. Reserve FK-based contact *strictly* for the grasp optimization phase.

### Phase 2: RL Algorithm & Training Stability (P1)
- [x] **Task 2.1: Remove Off-Policy Bias from Teacher Forcing (`single_door_ppo.py`)**
  - **Issue:** Action blending (`(1 - tf) * sampled + tf * teacher`) corrupts GAE advantage estimates.
  - **Action:** Remove direct action blending. Implement teacher forcing via reward shaping, OR use a DAgger-style approach (separate replay buffer mixing).
- [x] **Task 2.2: Implement Observation Normalization (`single_door_ppo.py` / `single_door_rl_task.py`)**
  - **Issue:** Disparate observation scales (0.01m vs 1.0m/s vs categorical flags) hurt neural network initialization and convergence.
  - **Action:** Implement a `RunningMeanStd` class (Welford's algorithm) to normalize observations dynamically during rollouts.
  - **Status:** Already implemented — `RunningMeanStd` with Welford's algorithm, integrated in rollout loop, stats saved in checkpoints.
- [x]] **Task 2.3: Simplify Reward Function (`single_door_rl_task.py`)**
  - **Issue:** Over-engineered, gated rewards (e.g., `pinch_gate`, `progress_gate`) create non-smooth optimization landscapes.
  - **Action:** Rewrite `compute_single_door_reward`. Remove gates. Use a continuous formula: `R = w_progress * delta_progress + w_contact * contact_score + w_action * action_penalty + success_bonus`.

### Phase 3: Grasp Optimization & Environment Tweaks (P2)
- [x] **Task 3.1: Upgrade Rotation Parameterization (`optimize_hoi.py`)**
  - **Issue:** Optimizing `palm_rot` as a raw 4D quaternion causes erratic Adam updates.
  - **Action:** Implement 6D rotation representation (Zhou et al. CVPR 2019). Convert to matrix for forward passes.
- [x] **Task 3.2: Improve Penetration Check (`optimize_hoi.py`)**
  - **Issue:** The single-plane penetration check fails for complex/recessed handles.
  - **Action:** Wire the `compute_batch_signed_distance` (mesh-aware nearest-surface-point) from `FastContactCalculator` into the optimization loop.
- [x] **Task 3.3: Network and Batch Sizing Adjustments**
  - **Action:** Increase hidden dimensions (e.g., 3-layer MLP with 512 units), increase parallel environments (8-16+), and increase batch sizes to stabilize PPO.
- [ ] **Task 3.4: Dynamic Wrist Rotation Lock (`single_door_residual_env.py`)**
  - **Issue:** `wrist_rotation_lock=0.85` fights the task for revolute doors.
  - **Action:** Make the lock phase-dependent or axis-dependent (e.g., lock roll but allow yaw/pitch).
