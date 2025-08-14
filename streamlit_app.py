import streamlit as st
import numpy as np
from engine.cw import simulate_relative_motion
from engine.policies import ThreatApproach, KeepOutPolicy, LLMHeuristicPolicy
from engine.scoring import score_run
from engine.visualize import plot_range_time, plot_traj_3d

st.set_page_config(page_title="Orbital Range – SIL MVP", layout="wide")

st.title("Orbital Range – Software‑in‑the‑Loop MVP")
st.caption("Red vs. Blue proximity ops in Hill frame (Clohessy–Wiltshire)")

with st.sidebar:
    st.header("Scenario Setup")
    alt_km = st.number_input("Chief (Blue) circular orbit altitude [km]", 300, 1200, 700, 10)
    r_earth_km = 6378.137
    mu_earth_km3_s2 = 398600.4418
    a_km = r_earth_km + alt_km
    n = np.sqrt(mu_earth_km3_s2 / a_km**3)  # mean motion [rad/s]

    st.subheader("Initial Relative State (Threat wrt Blue)")
    x0 = st.number_input("x0 (radial) [m]", -5000, 5000, 3000)
    y0 = st.number_input("y0 (along‑track) [m]", -50000, 50000, -8000)
    z0 = st.number_input("z0 (cross‑track) [m]", -5000, 5000, 500)
    vx0 = st.number_input("vx0 [m/s]", -5, 5, 0)
    vy0 = st.number_input("vy0 [m/s]", -5, 5, -0.02)
    vz0 = st.number_input("vz0 [m/s]", -5, 5, 0)

    st.subheader("Run Settings")
    sim_minutes = st.slider("Simulation duration [min]", 1, 60, 15)
    dt = st.select_slider("Timestep [s]", options=[0.5, 1.0, 2.0, 5.0], value=1.0)
    steps = int(sim_minutes * 60 / dt)

    st.subheader("Detection & Keep‑Out")
    detect_R_km = st.slider("Detection radius [km]", 0, 20, 2)
    KOZ_R_km = st.slider("Keep‑Out radius [km]", 0, 10, 1)

    st.subheader("Policies")
    desired_close_mps = st.slider("Threat desired closing speed [m/s]", 0, 50, 10) / 100.0  # 0.0–0.5
    threat_burn_mps = st.slider("Threat DV per second [m/s]", 0, 50, 5) / 100.0            # 0.0–0.5
    blue_dodge_mps = st.slider("Blue dodge DV [m/s]", 0, 100, 10) / 100.0                 # 0.0–1.0

    ai_policy_on = st.toggle("Use AI heuristic policy for Blue (demo)", value=False)

    st.subheader("Noise & Monte Carlo (optional)")
    process_noise = st.slider("Process noise accel 1σ [m/s²]", 0, 10, 0) / 1000.0

# Build policies
threat_policy = ThreatApproach(desired_v_close=desired_close_mps, dv_rate_limit=threat_burn_mps)
blue_policy = LLMHeuristicPolicy(dodge_dv=blue_dodge_mps) if ai_policy_on else KeepOutPolicy(dodge_dv=blue_dodge_mps)

# Run button
if st.button("Run Simulation", type="primary"):
    x0_vec = np.array([x0, y0, z0, vx0, vy0, vz0], dtype=float)

    hist = simulate_relative_motion(
        n=n,
        x0=x0_vec,
        steps=steps,
        dt=dt,
        detect_R_km=detect_R_km,
        KOZ_R_km=KOZ_R_km,
        threat_policy=threat_policy,
        blue_policy=blue_policy,
        process_noise_accel_std=process_noise,
    )

    # Score
    scores = score_run(hist, detect_R_km=detect_R_km, KOZ_R_km=KOZ_R_km)

    # Layout
    col1, col2 = st.columns([1, 1])
    with col1:
        st.plotly_chart(plot_range_time(hist), use_container_width=True)
        st.json(scores)
    with col2:
        st.plotly_chart(plot_traj_3d(hist), use_container_width=True)

    # Pretty summary
    st.subheader("Outcome")
    st.write(f"**{scores['outcome']}**")
    st.write(
        f"Detection time: {scores['detection_time_s']:.1f} s  ·  "
        f"Closest approach: {scores['closest_approach_m']:.1f} m  ·  "
        f"Time inside KOZ: {scores['time_inside_keepout_s']:.1f} s  ·  "
        f"Blue ΔV: {scores['blue_total_dv_mps']:.3f} m/s  ·  Threat ΔV: {scores['threat_total_dv_mps']:.3f} m/s"
    )
else:
    st.info("Configure parameters on the left, then click **Run Simulation**.")
