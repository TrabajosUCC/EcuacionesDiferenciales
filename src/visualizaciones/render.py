"""Conversión de especificaciones de figuras a objetos Plotly."""
from __future__ import annotations


def a_plotly(spec: dict):
    import plotly.graph_objects as go

    fig = go.Figure()
    for s in spec["series"]:
        fig.add_trace(go.Scatter(
            x=s["x"], y=s["y"], name=s["nombre"], mode=s.get("modo", "lines"),
            line={"dash": s["dash"]} if s.get("dash") else None,
        ))
    fig.update_layout(
        title=spec["titulo"], xaxis_title=spec["xl"], yaxis_title=spec["yl"],
        template="plotly_white", hovermode="x unified" if not spec.get("igual") else "closest",
        legend={"orientation": "h", "y": -0.2}, margin={"l": 50, "r": 20, "t": 50, "b": 50},
    )
    if spec.get("ylog"):
        fig.update_yaxes(type="log")
    if spec.get("igual"):
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig
