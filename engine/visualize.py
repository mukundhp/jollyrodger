import numpy as np
import plotly.graph_objects as go


def plot_range_time(hist):
    t = hist["t"]
    rng = hist["range_m"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=rng/1000.0, mode="lines", name="Range"))
    fig.update_layout(
        title="Range vs Time",
        xaxis_title="Time [s]",
        yaxis_title="Range [km]",
        height=400,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def plot_traj_3d(hist):
    X = hist["x"]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=X[:,0], y=X[:,1], z=X[:,2],
            mode="lines",
            line=dict(width=4),
            name="Threat rel traj",
        )
    )
    # Keep-out sphere
    # Draw a faint sphere for KOZ at time 0 using first logged KOZ from hist if present
    # (For simplicity, infer from 'inside_KOZ' array and initial range)
    # We'll just draw a unit reference sphere scaled to min KOZ seen in run if any KOZ event happened
    inside = hist.get("inside_KOZ", None)
    if inside is not None and any(inside):
        # Find first index where inside is True and estimate KOZ radius from that position's range
        idx = np.argmax(inside)
        R = np.linalg.norm(X[idx])
        u = np.linspace(0, 2*np.pi, 24)
        v = np.linspace(0, np.pi, 24)
        xs = R * np.outer(np.cos(u), np.sin(v))
        ys = R * np.outer(np.sin(u), np.sin(v))
        zs = R * np.outer(np.ones_like(u), np.cos(v))
        fig.add_trace(go.Surface(x=xs, y=ys, z=zs, showscale=False, opacity=0.2, name="KOZ"))

    fig.update_layout(
        title="Threat Trajectory in Hill Frame (Blue at origin)",
        scene=dict(
            xaxis_title="Radial x [m]",
            yaxis_title="Along-track y [m]",
            zaxis_title="Cross-track z [m]",
            aspectmode="data",
        ),
        height=600,
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig
