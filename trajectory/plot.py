from data.types import Array
import os
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def save_2d_plot_as_png(xs: Array, ys: Array, file_name: str):
    if len(xs) != len(ys):
        raise ValueError("xs and ys lengths must match")
    if not file_name or file_name.strip() == "":
        raise ValueError("file_name required")

    if not file_name.lower().endswith(".png"):
        file_name += ".png"

    dir_path = os.path.dirname(file_name) or "./output"
    os.makedirs(dir_path, exist_ok=True)

    path = file_name if os.path.isabs(
        file_name) else os.path.join(dir_path, file_name)

    plt.figure(figsize=(7, 5.6))
    plt.plot(xs, ys, linewidth=2)
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()


def save_3d_plot_as_html(xs: Array, ys: Array, zs: Array, file_name: str):
    if not (len(xs) == len(ys) == len(zs)):
        raise ValueError("xs, ys, zs lengths must match")
    if not file_name or file_name.strip() == "":
        raise ValueError("file_name required")

    if not file_name.lower().endswith(".html"):
        file_name += ".html"

    dir_path = os.path.dirname(file_name) or "./output"
    os.makedirs(dir_path, exist_ok=True)

    path = file_name if os.path.isabs(
        file_name) else os.path.join(dir_path, file_name)

    fig = go.Figure(
        data=[go.Scatter3d(  # type: ignore
            x=xs,
            y=ys,
            z=zs,
            mode="lines",
            line=dict(width=4)
        )]  # type: ignore
    )
    fig.update_layout(width=700, height=560)
    fig.write_html(path, include_plotlyjs="cdn")  # type: ignore
