import numpy as np

def score_run(hist, detect_R_km, KOZ_R_km):
    t = hist["t"]
    rng = hist["range_m"]
    detected = np.asarray(hist["detected"], dtype=bool)
    inside = np.asarray(hist["inside_KOZ"], dtype=bool)
    blue_dv = hist["blue_dv"]
    threat_dv = hist["threat_dv"]

    detection_time = float(t[detected][0]) if detected.any() else float("inf")
    closest = float(np.min(rng))
    time_inside = float(np.sum(inside) * (t[1] - t[0])) if len(t) > 1 else 0.0
    blue_total = float(np.sum(blue_dv))
    threat_total = float(np.sum(threat_dv))

    if np.isfinite(detection_time) and time_inside == 0.0:
        outcome = "Blue maintained KOZ with timely detection"
    elif time_inside > 0.0 and blue_total > 0.0:
        outcome = "KOZ violated briefly; Blue recovered with dodge"
    elif time_inside > 0.0:
        outcome = "KOZ violated (no dodge)"
    else:
        outcome = "No encounter"

    return {
        "detection_time_s": detection_time if np.isfinite(detection_time) else None,
        "closest_approach_m": closest,
        "time_inside_keepout_s": time_inside,
        "blue_total_dv_mps": blue_total,
        "threat_total_dv_mps": threat_total,
        "outcome": outcome,
    }
