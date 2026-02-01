import random
from collections import deque

class FrozenLakeEnv(object):
    """
    Frozen Lake environment 
    considered deterministic transitions - since no stochasticity is specified
    States: (row, col) on an N x N grid
    Actions: 0=LEFT, 1=DOWN, 2=RIGHT, 3=UP
    Terminal: hole or goal
    Rewards: +1 goal, -1 hole, 0 otherwise
    """

    LEFT, DOWN, RIGHT, UP = 0, 1, 2, 3
    ACTIONS = (LEFT, DOWN, RIGHT, UP)
    ACTION_NAMES = {LEFT: "L", DOWN: "D", RIGHT: "R", UP: "U"}

    def __init__(self, n=4, holes=None, start=(0, 0), goal=None, seed=40):
        self.n = int(n)
        self.start = tuple(start)
        self.goal = tuple(goal) if goal is not None else (self.n - 1, self.n - 1) # default goal at bottom-right if not specified by user
        self.rng = random.Random(seed)

        if holes is None:
            holes = set()
        self.holes = set(map(tuple, holes))

        # basic validity checks
        if self.start in self.holes:
            raise ValueError("Start cannot be a hole.")
        if self.goal in self.holes:
            raise ValueError("Goal cannot be a hole.")
        if not self._in_bounds(self.start) or not self._in_bounds(self.goal):
            raise ValueError("Start/goal out of bounds.") # Prevents invalid coordinates

        self.s = None  # current agent state (r,c)

    # -------- core RL env API --------
    # reset agent to start state
    def reset(self):
        self.s = self.start
        return self._state_id(self.s)  # return integer state id

    # take action a, return (next_state, reward, done, info)
    def step(self, a):
        if self.s is None:
            raise RuntimeError("Call reset() before step().") # Ensure environment is reset before stepping

        a = int(a)
        if a not in self.ACTIONS:
            raise ValueError("Invalid action: {}".format(a)) # Rejects invalid actions, in this case not in {0,1,2,3}

        # If already terminal, stay terminal
        if self.is_terminal(self.s):
            return self._state_id(self.s), 0.0, True, {}

        # Unpack row and col.
        r, c = self.s

        # Initialize next row and col (nr,nc) to current state as default.
        nr, nc = r, c

        # Updates next coordinates based on action.
        if a == self.LEFT:
            nc -= 1
        elif a == self.RIGHT:
            nc += 1
        elif a == self.UP:
            nr -= 1
        elif a == self.DOWN:
            nr += 1

        # confined within grid bounds
        if not self._in_bounds((nr, nc)):
            nr, nc = r, c # stay in place if out of bounds

        # update state
        self.s = (nr, nc)
        # check terminal
        done = self.is_terminal(self.s)

        # assign reward
        if self.s == self.goal:
            reward = 1.0
        elif self.s in self.holes:
            reward = -1.0
        else:
            reward = 0.0

        return self._state_id(self.s), reward, done, {}

    # -------- helpers --------

    # number of states
    def num_states(self):
        return self.n * self.n

    # number of actions - always 4
    def num_actions(self):
        return 4

    # check if state s is terminal - returns True if s is goal or hole
    def is_terminal(self, s):
        s = tuple(s)
        return (s == self.goal) or (s in self.holes)

    # convert state id (used by RL) to grid position (row, col)
    def state_to_pos(self, sid):
        sid = int(sid)
        return (sid // self.n, sid % self.n)

    # convert grid position (row, col) to state id
    def pos_to_state(self, pos):
        return self._state_id(tuple(pos))

    # sample random action
    def sample_action(self):
        return self.rng.choice(self.ACTIONS)

    # set random seed
    def seed(self, seed=40):
        self.rng.seed(seed)

    # Prints a simple view of the grid.
    def render(self):
        # Simple ASCII render - S=start, G=goal, H=hole, F=frozen/safe, A=agent
        grid = [["F"] * self.n for _ in range(self.n)]
        for (r, c) in self.holes:
            grid[r][c] = "H"
        sr, sc = self.start
        gr, gc = self.goal
        grid[sr][sc] = "S"
        grid[gr][gc] = "G"
        
        if self.s is not None:
            cr, cc = self.s
            # Don't overwrite S/G/H markers; show agent as 'A' on safe tiles only
            if grid[cr][cc] == "F":
                grid[cr][cc] = "A"
            print(f"Agent at: (row={cr}, col={cc}) | state_id={self._state_id(self.s)}")

        # border printing
        print("-" * (2 * self.n + 1))
        for r in range(self.n):
            print("|" + " ".join(grid[r]) + "|")
        print("-" * (2 * self.n + 1))

    # Checks valid coordinate.
    def _in_bounds(self, pos):
        r, c = pos
        return 0 <= r < self.n and 0 <= c < self.n

    # Computes state id from position.
    def _state_id(self, pos):
        r, c = pos
        return r * self.n + c


def generate_random_solvable_holes(n, hole_ratio=0.25, start=(0, 0), goal=None, seed=40, max_tries=5000):
    """
    Generate random holes with given ratio, ensuring the goal remains reachable from start.
    Ratio is holes / (n*n). For 10x10 with 25%, hole_ratio=0.25 -> 25 holes.
    """
    n = int(n)
    start = tuple(start)
    goal = tuple(goal) if goal is not None else (n - 1, n - 1)
    rng = random.Random(seed)

    # number of holes to place
    total = n * n
    k = int(round(hole_ratio * total))

    # Build list of possible hole locations - forbid placing holes on start/goal
    all_cells = [(r, c) for r in range(n) for c in range(n) if (r, c) not in (start, goal)]

    # function to check if goal is reachable from start given a set of holes (using BFS)
    def reachable(holes_set):
        # BFS on non-hole cells
        q = deque([start])
        visited = set([start])
        while q:
            r, c = q.popleft()
            if (r, c) == goal:
                return True # If goal reached, solvable.
            
            for dr, dc in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                nr, nc = r + dr, c + dc
                np = (nr, nc)
                # Check bounds and if not a hole or visited
                if 0 <= nr < n and 0 <= nc < n and np not in holes_set and np not in visited:
                    visited.add(np)
                    q.append(np)
        return False # Goal not reachable.

    for _ in range(max_tries):
        holes = set(rng.sample(all_cells, k))
        if reachable(holes):
            return holes # Return holes if solvable

    raise RuntimeError("Failed to sample solvable holes after {} tries. Try lowering hole_ratio or increasing max_tries.".format(max_tries))



