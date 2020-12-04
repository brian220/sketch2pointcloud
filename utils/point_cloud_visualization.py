import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm

# Don't delete this line, even if PyCharm says it's an unused import.
# It is required for projection='3d' in add_subplot()
from mpl_toolkits.mplot3d import Axes3D

def get_point_cloud_image(generate_point_cloud, save_dir, n_itr, img_type):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    generate_point_cloud = generate_point_cloud.squeeze()
    fig = plot_3d_point_cloud(
            generate_point_cloud[:, 0],
            generate_point_cloud[:, 1],
            generate_point_cloud[:, 2],
            in_u_sphere=True,
            show=False,
            title=f"{n_itr} {img_type}"
        )
    
    save_path = os.path.join(save_dir, 'pcs-%06d.png' % n_itr)
    fig.savefig(save_path)
    plt.close(fig)


def plot_3d_point_cloud(x, y, z, show=False, show_axis=True, in_u_sphere=False,
                        marker='.', s=8, alpha=.8, figsize=(5, 5), elev=10,
                        azim=240, axis=None, title=None, *args, **kwargs):
    # plt.switch_backend('tkagg')
    if axis is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = axis
        fig = axis

    if title is not None:
        plt.title(title)

    sc = ax.scatter(x, y, z, marker=marker, s=s, alpha=alpha, *args, **kwargs)
    ax.view_init(elev=elev, azim=azim)

    if in_u_sphere:
        ax.set_xlim3d(-0.5, 0.5)
        ax.set_ylim3d(-0.5, 0.5)
        ax.set_zlim3d(-0.5, 0.5)
    else:
        # Multiply with 0.7 to squeeze free-space.
        miv = 0.7 * np.min([np.min(x), np.min(y), np.min(z)])
        mav = 0.7 * np.max([np.max(x), np.max(y), np.max(z)])
        ax.set_xlim(miv, mav)
        ax.set_ylim(miv, mav)
        ax.set_zlim(miv, mav)
        plt.tight_layout()

    if not show_axis:
        plt.axis('off')

    if 'c' in kwargs:
        plt.colorbar(sc)

    if show:
        plt.show()

    return fig