import numpy as np

class ThreatApproach:
    """Threat tries to achieve a desired closing speed by burning along the line-of-sight.
    dv_rate_limit is the max |a| converted to DV per second [m/s], so accel magnitude = dv_rate_limit / 1s.
    """
    def __init__(self, desired_v_close=0.1, dv_rate_limit=0.05):
        self.desired = float(desired_v_close)
        self.dv_rate_limit = float(dv_rate_limit)
        self.dv_rate_last = 0.0
        self.total_dv = 0.0

    def reset(self):
        self.dv_rate_last = 0.0
        self.total_dv = 0.0

    def command(self, t, state, n, rng, closing):
        r = state[:3]
        if rng < 1.0:
            self.dv_rate_last = 0.0
            return np.zeros(3)
        los_hat = r / rng
        # If closing slower than desired, push along -LOS to increase closing; if too fast, back off.
        err = self.desired - closing
        accel_mag = np.clip(err, -self.dv_rate_limit, self.dv_rate_limit)
        a_cmd = -accel_mag * los_hat
        self.dv_rate_last = float(abs(accel_mag))
        self.total_dv += abs(accel_mag)
        return a_cmd


class KeepOutPolicy:
    """Blue executes a single perpendicular dodge when range < KOZ and closing>0."""
    def __init__(self, dodge_dv=0.1):
        self.dodge_dv = float(dodge_dv)
        self.did_dodge = False
        self.dv_rate_last = 0.0

    def reset(self):
        self.did_dodge = False
        self.dv_rate_last = 0.0

    def on_detect(self, t, state):
        # Placeholder hook; could adjust posture later
        pass

    def command(self, t, state, n, rng, closing, KOZ_R_m):
        self.dv_rate_last = 0.0
        if (not self.did_dodge) and (rng < KOZ_R_m) and (closing > 0.0):
            # Perpendicular dodge: choose unit vector perpendicular to LOS using velocity to define a plane
            r = state[:3]
            v = state[3:]
            los = r / (np.linalg.norm(r) + 1e-9)
            # Try left/right using vy sign for determinism
            basis = np.cross(los, np.array([0.0, 0.0, 1.0]))
            if np.linalg.norm(basis) < 1e-6:
                basis = np.cross(los, np.array([0.0, 1.0, 0.0]))
            dodge_dir = basis / (np.linalg.norm(basis) + 1e-9)
            # Apply as 1-second equivalent acceleration (m/s²)
            accel = self.dodge_dv * dodge_dir
            self.did_dodge = True
            self.dv_rate_last = self.dodge_dv
            return accel, "DODGE"
        return np.zeros(3), None


class LLMHeuristicPolicy(KeepOutPolicy):
    """Demo policy that 'pretends' to ask an LLM to choose HOLD/LEFT/RIGHT based on simple heuristics.
    For MVP we avoid external calls and just make a branching choice deterministically from state.
    """
    def command(self, t, state, n, rng, closing, KOZ_R_m):
        # Use the same trigger as KeepOutPolicy, but pick 'left' or 'right' based on sign of y (along-track)
        self.dv_rate_last = 0.0
        if (not self.did_dodge) and (rng < KOZ_R_m) and (closing > 0.0):
            r = state[:3]
            los = r / (np.linalg.norm(r) + 1e-9)
            # 'LLM' picks side: if y>0 go LEFT, else RIGHT
            side = -1.0 if r[1] > 0 else 1.0
            basis = np.cross(los, np.array([0.0, 0.0, 1.0]))
            if np.linalg.norm(basis) < 1e-6:
                basis = np.cross(los, np.array([0.0, 1.0, 0.0]))
            dodge_dir = side * basis / (np.linalg.norm(basis) + 1e-9)
            accel = self.dodge_dv * dodge_dir
            self.did_dodge = True
            self.dv_rate_last = self.dodge_dv
            event = "AI_DODGE_LEFT" if side < 0 else "AI_DODGE_RIGHT"
            return accel, event
        return np.zeros(3), None
