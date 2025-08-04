import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def plot_frame(ax, origin, rotation_matrix, scale=0.02):
    x_dir = rotation_matrix[:, 0] * scale
    y_dir = rotation_matrix[:, 1] * scale
    z_dir = rotation_matrix[:, 2] * scale

    ax.quiver(origin[0], origin[1], origin[2], x_dir[0], x_dir[1], x_dir[2], color="r", linewidth=4)
    ax.quiver(origin[0], origin[1], origin[2], y_dir[0], y_dir[1], y_dir[2], color="g", linewidth=4)
    ax.quiver(origin[0], origin[1], origin[2], z_dir[0], z_dir[1], z_dir[2], color="b", linewidth=4)


def plot_interactive_2d(callback_data):

    positions_over_time = np.array(callback_data["position"])

    plt.ion()
    x, y = [0], [0]

    figure = plt.figure(figsize=(10, 8))
    ax = figure.add_subplot(111)

    (line1,) = ax.plot(x, y, c="b", linewidth=3)
    ax.scatter(x, y, c="b")

    ax.set_xlim(-0.1, 0.6)
    ax.set_ylim(-0.1, 0.6)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    plt.grid()

    for time in tqdm(range(1, len(callback_data["time"]))):
        line1.set_xdata(positions_over_time[time][0])
        line1.set_ydata(positions_over_time[time][1])
        figure.canvas.draw()
        figure.canvas.flush_events()

    # keep the plot open
    plt.show(block=True)


def plot_interactive_3d(positions_over_time, time):

    positions_over_time = np.array(positions_over_time)

    plt.ion()
    x, y, z = [0], [0], [0]

    figure = plt.figure(figsize=(10, 8))
    ax = figure.add_subplot(projection="3d")

    (line1,) = ax.plot(x, y, c="b", linewidth=3)
    ax.scatter(x, y, z, c="b")

    ax.set_xlim(-0.1, 0.6)
    ax.set_ylim(-0.1, 0.6)
    ax.set_zlim(-0.2, 0.2)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.grid()

    for time in tqdm(range(1, len(time))):
        line1.set_xdata(positions_over_time[time][0])
        line1.set_ydata(positions_over_time[time][1])
        line1.set_3d_properties(positions_over_time[time][2])
        figure.canvas.draw()
        figure.canvas.flush_events()

    # keep the plot open
    plt.show(block=True)


def plot_result_3d(dict_out):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    action = dict_out["action"][0]
    initial_shape = dict_out["init_shape"]
    initial_directors = dict_out["init_directors"]
    result_shape = dict_out["final_shape"]
    result_directors = dict_out["final_directors"]

    ax.plot(initial_shape[0], initial_shape[1], initial_shape[2], label="initial shape")
    ax.plot(result_shape[0], result_shape[1], result_shape[2], label="final shape")

    # plot the initial frame at the action node
    if action is not None:
        x_world = initial_shape[..., action.idx]
        Q = initial_directors[..., action.idx]
        plot_frame(ax, x_world, Q.T)

        # plot the final frame at the action node
        x_world = result_shape[..., action.idx]
        Q = result_directors[..., action.idx]
        plot_frame(ax, x_world, Q.T)

    ax.legend()

    plt.axis("equal")
    plt.tight_layout()
    plt.show()
