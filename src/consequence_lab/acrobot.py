"""Public observation helpers for the Gymnasium Acrobot preflight."""

from __future__ import annotations

import numpy as np


def acrobot_state_from_observation(observation: np.ndarray) -> np.ndarray:
    """Invert Acrobot-v1's six-value observation into its four state variables.

    This consumes only the public observation. Angles are recovered with
    atan2(sin(theta), cos(theta)); angular velocities pass through unchanged.
    """
    obs = np.asarray(observation, dtype=np.float64)
    if obs.shape != (6,) or not np.all(np.isfinite(obs)):
        raise ValueError("Acrobot observation must contain six finite values")
    norm1 = float(np.hypot(obs[0], obs[1]))
    norm2 = float(np.hypot(obs[2], obs[3]))
    if not np.isclose(norm1, 1.0, atol=1e-4) or not np.isclose(norm2, 1.0, atol=1e-4):
        raise ValueError("angle cosine/sine pairs must be unit length")
    return np.asarray(
        [np.arctan2(obs[1], obs[0]), np.arctan2(obs[3], obs[2]), obs[4], obs[5]],
        dtype=np.float64,
    )


def acrobot_mechanical_energy(state: np.ndarray) -> float:
    """Total mechanical energy under the pinned Gymnasium Acrobot convention.

    ``theta1`` is measured from the hanging-down vertical and ``theta2`` is
    relative to link 1.  The default Gymnasium link parameters are used.  The
    returned potential-energy zero is arbitrary; the upright configuration is
    the maximum-potential configuration and is the energy-shaping target.
    """
    theta1, theta2, velocity1, velocity2 = np.asarray(state, dtype=np.float64)
    m1 = m2 = 1.0
    length1 = 1.0
    com1 = com2 = 0.5
    inertia1 = inertia2 = 1.0
    gravity = 9.8

    d1 = (
        m1 * com1**2
        + m2 * (length1**2 + com2**2 + 2.0 * length1 * com2 * np.cos(theta2))
        + inertia1
        + inertia2
    )
    d2 = m2 * (com2**2 + length1 * com2 * np.cos(theta2)) + inertia2
    m22 = m2 * com2**2 + inertia2
    kinetic = 0.5 * (d1 * velocity1**2 + 2.0 * d2 * velocity1 * velocity2 + m22 * velocity2**2)
    potential = -((m1 * com1 + m2 * length1) * gravity * np.cos(theta1))
    potential -= m2 * com2 * gravity * np.cos(theta1 + theta2)
    return float(kinetic + potential)


def acrobot_energy_shaping_action(
    observation: np.ndarray,
    gains: tuple[float, float, float],
) -> int:
    """Quantized collocated-PFL energy-shaping controller for Acrobot-v1.

    This implements the MIT Underactuated Robotics form
    ``q2dd_des = -k1*q2 - k2*q2dot + k3*q1dot*(E - E_upright)``.
    The desired relative-joint acceleration is converted to actuator torque
    by inverting the exact pinned Gymnasium ``book`` dynamics, then quantized
    to its three legal torques. Gains are deliberately explicit and must be
    selected on development seeds; this function performs no hidden tuning.
    """
    state = acrobot_state_from_observation(observation)
    theta1, theta2, velocity1, velocity2 = state
    k1, k2, k3 = map(float, gains)
    if not np.isfinite([k1, k2, k3]).all() or min(k1, k2, k3) < 0.0:
        raise ValueError("energy-shaping gains must be finite and nonnegative")

    m1 = m2 = 1.0
    length1 = 1.0
    com1 = com2 = 0.5
    inertia1 = inertia2 = 1.0
    gravity = 9.8
    d1 = (
        m1 * com1**2
        + m2 * (length1**2 + com2**2 + 2.0 * length1 * com2 * np.cos(theta2))
        + inertia1
        + inertia2
    )
    d2 = m2 * (com2**2 + length1 * com2 * np.cos(theta2)) + inertia2
    phi2 = m2 * com2 * gravity * np.sin(theta1 + theta2)
    phi1 = (
        -m2 * length1 * com2 * velocity2**2 * np.sin(theta2)
        - 2.0 * m2 * length1 * com2 * velocity2 * velocity1 * np.sin(theta2)
        + (m1 * com1 + m2 * length1) * gravity * np.sin(theta1)
        + phi2
    )
    denominator = m2 * com2**2 + inertia2 - d2**2 / d1

    upright_energy = (m1 * com1 + m2 * length1 + m2 * com2) * gravity
    energy_error = acrobot_mechanical_energy(state) - upright_energy
    desired_acceleration2 = -k1 * theta2 - k2 * velocity2 + k3 * velocity1 * energy_error

    # Invert Gymnasium's pinned ``book`` branch of _dsdt, including its
    # -m2*l1*lc2*dtheta1^2*sin(theta2) Coriolis term.
    coupling = m2 * length1 * com2 * velocity1**2 * np.sin(theta2)
    desired_torque = (
        denominator * desired_acceleration2
        - d2 / d1 * phi1
        + coupling
        + phi2
    )
    legal_torques = np.asarray([-1.0, 0.0, 1.0])
    return int(np.argmin(np.abs(legal_torques - desired_torque)))


def acrobot_energy_pumping_action(observation: np.ndarray) -> int:
    """Parameter-free total-energy pumping sanity baseline.

    Applies torque in the direction that changes total energy toward the
    upright energy, using elbow angular velocity as actuator power direction.
    This is intentionally kept distinct from the gain-tuned PFL baseline.
    """
    state = acrobot_state_from_observation(observation)
    upright_energy = (0.5 + 1.0 + 0.5) * 9.8
    energy_error = acrobot_mechanical_energy(state) - upright_energy
    power_sign = energy_error * state[3]
    if abs(power_sign) < 1e-12:
        return 1
    torque = -1.0 if power_sign > 0.0 else 1.0
    return 0 if torque < 0.0 else 2


def acrobot_exact_step_batch(states: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """Vectorized Gymnasium 1.3.0 ``book`` dynamics step for planner branches.

    This mirrors AcrobotEnv's RK4 integration, angle wrapping, and velocity
    clipping for legal discrete action IDs. It is an optimization of the
    pinned exact simulator, not a learned model; callers must parity-check it
    against Gymnasium before using it for planning.
    """
    batch = np.asarray(states, dtype=np.float64)
    action_ids = np.asarray(actions)
    if batch.ndim != 2 or batch.shape[1] != 4 or not np.isfinite(batch).all():
        raise ValueError("states must be a finite (batch, 4) array")
    if action_ids.shape != (batch.shape[0],) or not np.isin(action_ids, [0, 1, 2]).all():
        raise ValueError("actions must contain one legal action ID per state")
    torque = np.asarray([-1.0, 0.0, 1.0], dtype=np.float64)[action_ids.astype(np.int64)]

    m1 = m2 = 1.0
    length1 = 1.0
    com1 = com2 = 0.5
    inertia1 = inertia2 = 1.0
    gravity = 9.8
    dt = 0.2

    def derivative(y):
        theta1, theta2, velocity1, velocity2 = y.T
        d1 = (
            m1 * com1**2
            + m2 * (length1**2 + com2**2 + 2.0 * length1 * com2 * np.cos(theta2))
            + inertia1
            + inertia2
        )
        d2 = m2 * (com2**2 + length1 * com2 * np.cos(theta2)) + inertia2
        phi2 = m2 * com2 * gravity * np.sin(theta1 + theta2)
        phi1 = (
            -m2 * length1 * com2 * velocity2**2 * np.sin(theta2)
            - 2.0 * m2 * length1 * com2 * velocity2 * velocity1 * np.sin(theta2)
            + (m1 * com1 + m2 * length1) * gravity * np.sin(theta1)
            + phi2
        )
        ddtheta2 = (
            torque
            + d2 / d1 * phi1
            - m2 * length1 * com2 * velocity1**2 * np.sin(theta2)
            - phi2
        ) / (m2 * com2**2 + inertia2 - d2**2 / d1)
        ddtheta1 = -(d2 * ddtheta2 + phi1) / d1
        return np.column_stack((velocity1, velocity2, ddtheta1, ddtheta2))

    k1 = derivative(batch)
    k2 = derivative(batch + 0.5 * dt * k1)
    k3 = derivative(batch + 0.5 * dt * k2)
    k4 = derivative(batch + dt * k3)
    next_state = batch + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    for angle_index in (0, 1):
        while np.any(next_state[:, angle_index] > np.pi):
            mask = next_state[:, angle_index] > np.pi
            next_state[mask, angle_index] -= 2.0 * np.pi
        while np.any(next_state[:, angle_index] < -np.pi):
            mask = next_state[:, angle_index] < -np.pi
            next_state[mask, angle_index] += 2.0 * np.pi
    next_state[:, 2] = np.clip(next_state[:, 2], -4.0 * np.pi, 4.0 * np.pi)
    next_state[:, 3] = np.clip(next_state[:, 3], -9.0 * np.pi, 9.0 * np.pi)
    return next_state
