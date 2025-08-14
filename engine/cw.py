import numpy as np

# Continuous-time Clohessy–Wiltshire (Hill) equations
# State x = [x, y, z, vx, vy, vz] in meters and m/s. x,y,z are R (radial), T (along-track), N (cross-track).

def cw_derivatives(n, state, accel=np.zeros(3)):
    x, y, z, vx, vy, vz = state
    ax, ay, az = accel
    dx = vx
    dy = vy
    dz = vz
    dvx = 3 * n**2 * x + 2 * n * vy + ax
    dvy = -2 * n * vx + ay
    dvz = -n**2 * z + az
    return np.array([dx, dy, dz, dvx, dvy, dvz])


def simulate_relative_motion(
    n, x0, steps, dt, detect_R_km, KOZ_R_km, threat_policy, blue_policy, process_noise_accel_std=0.0
):
    # History dict collects time series
    hist = {
        "t": [],
        "x": [],  # position [m]
        "v": [],  # velocity [m/s]
        "range_m": [],
        "closing_speed_mps": [],
        "detected": [],
        "inside_KOZ": [],
        "blue_dv": [],
        "threat_dv": [],
        "blue_events": [],
        "threat_events": [],
    }

    state = x0.astype(float).copy()
    detect_R_m = detect_R_km * 1000.0
    KOZ_R_m = KOZ_R_km * 1000.0

    blue_policy.reset()
    threat_policy.reset()

    detected = False

    for k in range(steps + 1):
        t = k * dt
        r = state[:3]
        v = state[3:]
        rng = float(np.linalg.norm(r))
        closing = -float(np.dot(r, v) / (rng + 1e-9))  # positive if closing

        if (not detected) and (rng <= detect_R_m):
            detected = True
            blue_policy.on_detect(t, state)

        # Query policies for accelerations (m/s^2)
        a_threat = threat_policy.command(t, state, n, rng, closing)
        a_blue, blue_event = blue_policy.command(t, state, n, rng, closing, KOZ_R_m)

        if blue_event:
            hist["blue_events"].append({"t": t, "event": blue_event})

        # Net acceleration = threat accel (applied to deputy) + blue accel effect (equal/opposite) — in Hill frame we approximate blue's maneuver as negative accel on deputy to change relative motion.
        # For this MVP, we model blue accel as directly applied to the relative state (superposition in linear CW).
        a_net = a_threat - a_blue

        # Add optional process noise
        if process_noise_accel_std > 0:
            a_net = a_net + np.random.normal(0.0, process_noise_accel_std, size=3)

        # Integrate (semi-implicit Euler)
        deriv = cw_derivatives(n, state, accel=a_net)
        new_v = v + deriv[3:] * dt
        new_r = r + new_v * dt  # use updated velocity for stability
        state = np.hstack([new_r, new_v])

        # Bookkeeping
        hist["t"].append(t)
        hist["x"].append(state[:3].copy())
        hist["v"].append(state[3:].copy())
        hist["range_m"].append(rng)
        hist["closing_speed_mps"].append(closing)
        hist["detected"].append(detected)
        hist["inside_KOZ"].append(rng < KOZ_R_m)

        # DV tallies (convert accel→DV over dt)
        hist["blue_dv"].append(blue_policy.dv_rate_last * dt)
        hist["threat_dv"].append(threat_policy.dv_rate_last * dt)

    # Convert lists to arrays
    for k in ["t", "range_m", "closing_speed_mps", "blue_dv", "threat_dv"]:
        hist[k] = np.asarray(hist[k], dtype=float)
    hist["x"] = np.asarray(hist["x"], dtype=float)
    hist["v"] = np.asarray(hist["v"], dtype=float)

    return hist
