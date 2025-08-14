# Orbital Range — SIL MVP

A minimal, no‑CUI, single‑file deployable MVP for a software‑in‑the‑loop proximity ops demo using Hill‑frame (Clohessy–Wiltshire) dynamics. Blue is the chief at the origin; Threat is the deputy in relative coordinates.

## Quickstart
1. **Clone** this repo.
2. `pip install -r requirements.txt`
3. `streamlit run streamlit_app.py`

Or **deploy on Streamlit Community Cloud**:
- Push to GitHub and create an app pointing to `streamlit_app.py` as the entrypoint.

## What it does
- Simulates Threat approach toward Blue with a simple control law for closing speed.
- Blue executes a single perpendicular dodge when the keep‑out zone (KOZ) is violated.
- Produces a scorecard (detection time, closest approach, time inside KOZ, DV used) and two plots.

## Configure
Use the left sidebar to set orbit altitude (derives mean motion), initial relative state, run time, detection radius, KOZ radius, and policy DV limits.

## Extend
- Add more policies in `engine/policies.py` (ensure they expose `reset()`, and `command(...)`).
- Swap dynamics to a higher‑fidelity model later (two‑body or SGP4 truth) while keeping the Hill‑frame controller.
- Add batch runs (Monte Carlo) and export scorecards as JSON/CSV.
