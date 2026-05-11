"""Reusable chart components for the dashboard."""

import plotly.graph_objects as go
import plotly.express as px
import numpy as np


def create_gauge_chart(value: float, title: str = "Fraud Score") -> go.Figure:
    """Create a gauge chart for fraud probability.

    Args:
        value: Probability value (0-1).
        title: Chart title.

    Returns:
        Plotly Figure.
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        title={"text": title},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#e74c3c" if value > 0.5 else "#2ecc71"},
            "steps": [
                {"range": [0, 30], "color": "#d5f5e3"},
                {"range": [30, 70], "color": "#fdebd0"},
                {"range": [70, 100], "color": "#fadbd8"},
            ],
        },
    ))
    fig.update_layout(height=250)
    return fig


def create_metrics_comparison(models: list, metrics: dict) -> go.Figure:
    """Create grouped bar chart for model comparison.

    Args:
        models: List of model names.
        metrics: Dict of {metric_name: [values]}.

    Returns:
        Plotly Figure.
    """
    fig = go.Figure()
    colors = px.colors.qualitative.Set2

    for i, (name, values) in enumerate(metrics.items()):
        fig.add_trace(go.Bar(
            name=name,
            x=models,
            y=values,
            marker_color=colors[i % len(colors)]
        ))

    fig.update_layout(
        barmode="group",
        yaxis_range=[0, 1.05],
        xaxis_tickangle=-45,
        height=500
    )
    return fig
