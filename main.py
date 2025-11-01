from trajectory import config, acceleration, solver, plot
from data.types import Float
import numpy as np


def main():
    t0 = Float(0.0)
    dt = Float(0.01)
    steps = 1000
    ts = np.arange(t0, t0 + dt * steps, dt, dtype=Float)

    conf = config.Config(
        wind_accel_x=acceleration.linear(Float(2.0)),
        wind_accel_y=acceleration.linear(Float(2.0)),
        wind_accel_z=acceleration.linear(Float(2.0)),
        brake_mag=Float(3.0),
        dt=dt,
        steps=steps,
        t0=t0,
        x0=Float(5.0),
        y0=Float(0.0),
        z0=Float(0.0)
    )

    (xs, ys, zs), (bxs, bys, bzs) = solver.solve(conf)
    bs = np.sqrt(np.pow(bxs, 2) + np.pow(bys, 2), np.pow(bzs, 2))

    plot.save_3d_plot_as_html(xs, ys, zs, "trajectory")
    plot.save_2d_plot_as_png(ts, xs, "t_vs_x")
    plot.save_2d_plot_as_png(ts, ys, "t_vs_y")
    plot.save_2d_plot_as_png(ts, zs, "t_vs_z")
    plot.save_2d_plot_as_png(ts, bxs, "t_vs_bx")
    plot.save_2d_plot_as_png(ts, bys, "t_vs_by")
    plot.save_2d_plot_as_png(ts, bzs, "t_vs_bz")
    plot.save_2d_plot_as_png(ts, bs, "t_vs_b")


if __name__ == '__main__':
    main()
