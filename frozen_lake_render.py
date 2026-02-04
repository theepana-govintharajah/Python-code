import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


class FrozenLakeMatplotlibRenderer:
    """
    Matplotlib renderer for FrozenLakeEnv.
    Pure visualization. No RL logic here.

    Supports sprite images for:
      - goal (gift.png)
      - holes (hole.jpg)
      - agent (robot.png)  <-- you must provide this, otherwise it falls back to a square marker.
    """

    def __init__(
        self,
        env,
        bg_image_path=None,
        pause=0.25,
        title="FrozenLake",
        goal_image_path="bg_images/gift.png",
        hole_image_path="bg_images/hole.jpg",
        agent_image_path="bg_images/robot.png",
        goal_zoom=0.15,
        hole_zoom=0.15,
        agent_zoom=0.15,
    ):
        self.env = env
        self.pause = float(pause)

        # --- interactive plotting ---
        plt.ion()
        self.fig, self.ax = plt.subplots()
        try:
            self.fig.canvas.manager.set_window_title(title)
        except Exception:
            pass

        # --- background ---
        self.bg = self._safe_imread(bg_image_path) if bg_image_path else None

        # --- sprites ---
        self.goal_img = self._safe_imread(goal_image_path)
        self.hole_img = self._safe_imread(hole_image_path)
        self.agent_img = self._safe_imread(agent_image_path)

        self.goal_zoom = float(goal_zoom)
        self.hole_zoom = float(hole_zoom)
        self.agent_zoom = float(agent_zoom)

        # --- marker fallbacks (only used if images fail to load) ---
        self.holes_scatter = None
        self.goal_scatter = None
        self.agent_scatter = None

        # --- image artists ---
        self.goal_artist = None
        self.agent_artist = None
        self.hole_artists = []  # list of AnnotationBbox, one per hole

        self._setup_static()

    # ------------------------
    # internal helpers
    # ------------------------
    def _safe_imread(self, path):
        """Try reading an image; return None if missing/invalid."""
        try:
            if path is None:
                return None
            if not os.path.exists(path):
                print(f"[Renderer] Image not found: {path} (will fallback to markers if needed)")
                return None
            return mpimg.imread(path)
        except Exception as e:
            print(f"[Renderer] Failed to load image {path}: {e} (will fallback to markers if needed)")
            return None

    def _cell_center(self, r, c):
        """Convert grid cell (row, col) to plot coordinates (x, y) at cell center."""
        return c + 0.5, r + 0.5

    def _make_artist(self, img, xy, zoom, zorder=5):
        """Create an AnnotationBbox for an image at xy."""
        oi = OffsetImage(img, zoom=zoom)
        ab = AnnotationBbox(
            oi,
            xy,
            frameon=False,
            box_alignment=(0.5, 0.5),
            zorder=zorder
        )
        self.ax.add_artist(ab)
        return ab

    # ------------------------
    # static plot setup
    # ------------------------
    def _setup_static(self):
        n = self.env.n
        self.ax.clear()

        # Background
        if self.bg is not None:
            # extent maps image to grid coordinates
            self.ax.imshow(self.bg, extent=[0, n, n, 0])
        else:
            self.ax.set_facecolor((0.85, 0.92, 1.0))

        # Grid lines
        for k in range(n + 1):
            self.ax.plot([0, n], [k, k], linewidth=1)
            self.ax.plot([k, k], [0, n], linewidth=1)

        # If hole image exists, create hole artists now (holes typically static)
        self.hole_artists = []
        if self.hole_img is not None and len(self.env.holes) > 0:
            for (r, c) in sorted(self.env.holes):
                xy = self._cell_center(r, c)
                ab = self._make_artist(self.hole_img, xy, self.hole_zoom, zorder=3)
                self.hole_artists.append(ab)
        else:
            # fallback: holes as X markers
            self.holes_scatter = self.ax.scatter([], [], marker="X", s=250, zorder=3)

        # Goal: image if available else circle marker
        if self.goal_img is None:
            self.goal_scatter = self.ax.scatter([], [], marker="o", s=250, zorder=4)
            self.goal_artist = None
        else:
            self.goal_scatter = None
            self.goal_artist = None  # created on first draw()

        # Agent: image if available else square marker
        if self.agent_img is None:
            self.agent_scatter = self.ax.scatter([], [], marker="s", s=250, zorder=6)
            self.agent_artist = None
        else:
            self.agent_scatter = None
            self.agent_artist = None  # created on first draw()

        # Axis cosmetics
        self.ax.set_xlim(0, n)
        self.ax.set_ylim(n, 0)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_aspect("equal")
        self.fig.tight_layout()

    # ------------------------
    # per-step draw/update
    # ------------------------
    def draw(self, action=None, reward=None, done=None):
        # --- holes update (only needed if using fallback scatter OR if env.holes can change) ---
        if self.hole_img is None:
            if len(self.env.holes) > 0:
                pts = [self._cell_center(r, c) for (r, c) in self.env.holes]
                self.holes_scatter.set_offsets(np.array(pts))
            else:
                self.holes_scatter.set_offsets(np.empty((0, 2)))
        else:
            # If holes are fixed, nothing needed. If holes might change dynamically, you’d need to rebuild artists.
            pass

        # --- goal update ---
        gr, gc = self.env.goal
        goal_xy = self._cell_center(gr, gc)

        if self.goal_img is None:
            self.goal_scatter.set_offsets(np.array([goal_xy]))
        else:
            if self.goal_artist is None:
                self.goal_artist = self._make_artist(self.goal_img, goal_xy, self.goal_zoom, zorder=4)
            else:
                self.goal_artist.xy = goal_xy

        # --- agent update ---
        if self.env.s is not None:
            ar, ac = self.env.s
            agent_xy = self._cell_center(ar, ac)

            if self.agent_img is None:
                self.agent_scatter.set_offsets(np.array([agent_xy]))
            else:
                if self.agent_artist is None:
                    self.agent_artist = self._make_artist(self.agent_img, agent_xy, self.agent_zoom, zorder=6)
                else:
                    self.agent_artist.xy = agent_xy

        # --- title/debug line ---
        parts = []
        if action is not None:
            parts.append(f"a={action}")
        if reward is not None:
            parts.append(f"r={reward}")
        if done is not None:
            parts.append(f"done={done}")
        if self.env.s is not None:
            parts.append(f"pos={self.env.s}")
        self.ax.set_title(" | ".join(parts))

        self.fig.canvas.draw_idle()
        plt.pause(self.pause)
