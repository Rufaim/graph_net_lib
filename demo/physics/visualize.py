import numpy as np

import matplotlib.pyplot as pyplot
from matplotlib.animation import FuncAnimation


def visualize_run(rollout_true,rollout_predicted=None,title=None,video_filename=None):
    rollout_true_xs = rollout_true[..., 0]
    rollout_true_ys = rollout_true[..., 1]

    if rollout_predicted is not None:
        rollout_predicted_xs = rollout_predicted[..., 0]
        rollout_predicted_ys = rollout_predicted[..., 1]

    num_frames = rollout_true.shape[0]
    fig, ax = pyplot.subplots()

    max_lim_y = np.max(rollout_true_ys)+0.1
    min_lim_y = np.min(rollout_true_ys)-0.1

    max_lim_x = np.max(rollout_true_xs)+0.1
    min_lim_x = np.min(rollout_true_xs)-0.1

    chain_points_true = ax.plot([], [], 'k-')[0]
    drawable = [chain_points_true]

    if title is not None:
        ax.set_title(title)

    if rollout_predicted is not None:
        chain_points_predicted = ax.plot([], [], 'r-')[0]
        drawable.append(chain_points_predicted)

    def init():
        ax.set_xlim(min_lim_x, max_lim_x)
        ax.set_ylim(min_lim_y, max_lim_y)
        return drawable

    def update(frame):
        chain_points_true.set_data(rollout_true_xs[frame], rollout_true_ys[frame])
        if rollout_predicted is not None:
            chain_points_predicted.set_data(rollout_predicted_xs[frame],rollout_predicted_ys[frame])
        return drawable

    ani = FuncAnimation(fig, update, frames=num_frames, init_func=init, interval=2,repeat=True)

    if video_filename is None:
        pyplot.show()
    else:
        print("Saving videofile")
        ani.save(video_filename,dpi=100,progress_callback=lambda i, n: print(f'\rSaving frame {i} of {n}',end=""))
        print()
    pyplot.close(fig)
