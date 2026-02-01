import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class FrozenLakeMatplotlibRenderer:
    """
    Matplotlib renderer for FrozenLakeEnv.
    Pure visualization. No RL logic here.
    """

    def __init__(self, env, bg_image_path=None, pause=0.25, title="FrozenLake"):
        self.env = env
        self.pause = float(pause)

        plt.ion()
        self.fig, self.ax = plt.subplots()
        try:
            self.fig.canvas.manager.set_window_title(title)
        except Exception:
            pass

        self.bg = None
        if bg_image_path is not None:
            self.bg = mpimg.imread(bg_image_path)

        self.holes_scatter = None
        self.goal_scatter = None
        self.agent_scatter = None

        self._setup_static()

    def _setup_static(self):
        n = self.env.n
        self.ax.clear()

        if self.bg is not None:
            self.ax.imshow(self.bg, extent=[0, n, n, 0])
        else:
            self.ax.set_facecolor((0.85, 0.92, 1.0))

        # Grid
        for k in range(n + 1):
            self.ax.plot([0, n], [k, k], linewidth=1)
            self.ax.plot([k, k], [0, n], linewidth=1)

        self.holes_scatter = self.ax.scatter([], [], marker="X", s=250)
        self.goal_scatter  = self.ax.scatter([], [], marker="o", s=250)
        self.agent_scatter = self.ax.scatter([], [], marker="s", s=250)

        self.ax.set_xlim(0, n)
        self.ax.set_ylim(n, 0)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_aspect("equal")
        self.fig.tight_layout()

    def _cell_center(self, r, c):
        return c + 0.5, r + 0.5

    def draw(self, action=None, reward=None, done=None):
        # holes
        if len(self.env.holes) > 0:
            pts = [self._cell_center(r, c) for (r, c) in self.env.holes]
            self.holes_scatter.set_offsets(np.array(pts))
        else:
            self.holes_scatter.set_offsets(np.empty((0, 2)))

        # goal
        gr, gc = self.env.goal
        self.goal_scatter.set_offsets(
            np.array([self._cell_center(gr, gc)])
        )

        # agent
        if self.env.s is not None:
            ar, ac = self.env.s
            self.agent_scatter.set_offsets(
                np.array([self._cell_center(ar, ac)])
            )

        title = []
        if action is not None:
            title.append(f"a={action}")
        if reward is not None:
            title.append(f"r={reward}")
        if done is not None:
            title.append(f"done={done}")
        if self.env.s is not None:
            title.append(f"pos={self.env.s}")
        self.ax.set_title(" | ".join(title))

        self.fig.canvas.draw_idle()
        plt.pause(self.pause)
