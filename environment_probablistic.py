import random
from collections import deque


class FrozenLakeEnv(object):
    """
    Frozen Lake environment (probabilistic transitions via slipping)

    States: (row, col) on an N x N grid
    Actions: 0=LEFT, 1=DOWN, 2=RIGHT, 3=UP
    Terminal: hole or goal
    Rewards: +1 goal, -1 hole, 0 otherwise

    Transition model (slippery):
      - intended action taken with prob (1 - slip_prob)
      - with prob slip_prob, slip to a perpendicular action (split equally)
    """

    LEFT, DOWN, RIGHT, UP = 0, 1, 2, 3
    ACTIONS = (LEFT, DOWN, RIGHT, UP)
    ACTION_NAMES = {LEFT: "L", DOWN: "D", RIGHT: "R", UP: "U"}

    def __init__(self, n=4, holes=None, start=(0, 0), goal=None, seed=40, slip_prob=0.2):
        self.n = int(n)
        self.start = tuple(start)
        self.goal = tuple(goal) if goal is not None else (self.n - 1, self.n - 1)
        self.rng = random.Random(seed)

        self.slip_prob = float(slip_prob)
        if not (0.0 <= self.slip_prob <= 1.0):
            raise ValueError("slip_prob must be in [0, 1]. Got {}".format(self.slip_prob))

        if holes is None:
            holes = set()
        self.holes = set(map(tuple, holes))

        # basic validity checks
        if self.start in self.holes:
            raise ValueError("Start cannot be a hole.")
        if self.goal in self.holes:
            raise ValueError("Goal cannot be a hole.")
        if not self._in_bounds(self.start) or not self._in_bounds(self.goal):
            raise ValueError("Start/goal out of bounds.")

        self.s = None  # current agent state (r,c)

    # -------- core RL env API --------
    def reset(self):
        self.s = self.start
        return self._state_id(self.s)

    def step(self, a):
        if self.s is None:
            raise RuntimeError("Call reset() before step().")

        a = int(a)
        if a not in self.ACTIONS:
            raise ValueError("Invalid action: {}".format(a))

        # If already terminal, stay terminal
        if self.is_terminal(self.s):
            return self._state_id(self.s), 0.0, True, {}

        # --- KEY CHANGE: sample the ACTUAL executed action (stochastic transition) ---
        executed_a = self._sample_executed_action(a)

        # apply executed action deterministically to get next state
        r, c = self.s
        nr, nc = r, c

        if executed_a == self.LEFT:
            nc -= 1
        elif executed_a == self.RIGHT:
            nc += 1
        elif executed_a == self.UP:
            nr -= 1
        elif executed_a == self.DOWN:
            nr += 1

        # confined within grid bounds
        if not self._in_bounds((nr, nc)):
            nr, nc = r, c  # stay in place if out of bounds

        self.s = (nr, nc)
        done = self.is_terminal(self.s)

        # reward depends only on resulting state (still deterministic given state)
        if self.s == self.goal:
            reward = 1.0
        elif self.s in self.holes:
            reward = -1.0
        else:
            reward = 0.0

        info = {
            "intended_action": a,
            "executed_action": executed_a,
            "slipped": (executed_a != a),
        }

        return self._state_id(self.s), reward, done, info

    # -------- helpers --------
    def num_states(self):
        return self.n * self.n

    def num_actions(self):
        return 4

    def is_terminal(self, s):
        s = tuple(s)
        return (s == self.goal) or (s in self.holes)

    def state_to_pos(self, sid):
        sid = int(sid)
        return (sid // self.n, sid % self.n)

    def pos_to_state(self, pos):
        return self._state_id(tuple(pos))

    def sample_action(self):
        return self.rng.choice(self.ACTIONS)

    def seed(self, seed=40):
        self.rng.seed(seed)

    def render(self):
        grid = [["F"] * self.n for _ in range(self.n)]
        for (r, c) in self.holes:
            grid[r][c] = "H"
        sr, sc = self.start
        gr, gc = self.goal
        grid[sr][sc] = "S"
        grid[gr][gc] = "G"

        if self.s is not None:
            cr, cc = self.s
            if grid[cr][cc] == "F":
                grid[cr][cc] = "A"
            print(f"Agent at: (row={cr}, col={cc}) | state_id={self._state_id(self.s)}")

        print("-" * (2 * self.n + 1))
        for r in range(self.n):
            print("|" + " ".join(grid[r]) + "|")
        print("-" * (2 * self.n + 1))

    def _in_bounds(self, pos):
        r, c = pos
        return 0 <= r < self.n and 0 <= c < self.n

    def _state_id(self, pos):
        r, c = pos
        return r * self.n + c

    # --- NEW: stochastic transition sampler ---
    def _sample_executed_action(self, intended_a):
        """
        With prob (1-slip_prob), execute intended_a.
        With prob slip_prob, execute a perpendicular action (split equally).
        """
        if self.slip_prob == 0.0:
            return intended_a

        p = self.rng.random()
        if p < (1.0 - self.slip_prob):
            return intended_a

        # choose one of the two perpendicular actions uniformly
        perp_actions = self._perpendicular_actions(intended_a)
        return perp_actions[0] if self.rng.random() < 0.5 else perp_actions[1]

    def _perpendicular_actions(self, a):
        """
        Returns the two actions perpendicular to a.
        - If a is LEFT/RIGHT, perpendicular are UP/DOWN
        - If a is UP/DOWN, perpendicular are LEFT/RIGHT
        """
        if a in (self.LEFT, self.RIGHT):
            return (self.UP, self.DOWN)
        else:
            return (self.LEFT, self.RIGHT)


def generate_random_solvable_holes(n, hole_ratio=0.25, start=(0, 0), goal=None, seed=40, max_tries=5000):
    """
    Same as your original: generates a solvable hole configuration (BFS reachability).
    """
    n = int(n)
    start = tuple(start)
    goal = tuple(goal) if goal is not None else (n - 1, n - 1)
    rng = random.Random(seed)

    total = n * n
    k = int(round(hole_ratio * total))

    all_cells = [(r, c) for r in range(n) for c in range(n) if (r, c) not in (start, goal)]

    def reachable(holes_set):
        q = deque([start])
        visited = set([start])
        while q:
            r, c = q.popleft()
            if (r, c) == goal:
                return True

            for dr, dc in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                nr, nc = r + dr, c + dc
                np = (nr, nc)
                if 0 <= nr < n and 0 <= nc < n and np not in holes_set and np not in visited:
                    visited.add(np)
                    q.append(np)
        return False

    for _ in range(max_tries):
        holes = set(rng.sample(all_cells, k))
        if reachable(holes):
            return holes

    raise RuntimeError(
        "Failed to sample solvable holes after {} tries. Try lowering hole_ratio or increasing max_tries.".format(max_tries)
    )
