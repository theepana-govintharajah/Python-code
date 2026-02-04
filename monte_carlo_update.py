import random

def _argmax_random_tie(q_row, rng):
    """
    Argmax with RANDOM tie-breaking.
    Returns an action index among those with max Q-value, chosen uniformly at random.

    This replaces the old "first max wins" behavior (which biases LEFT due to action order).
    """
    best_q = q_row[0]
    best_actions = [0]

    for a in range(1, len(q_row)):
        qa = q_row[a]
        if qa > best_q:
            best_q = qa
            best_actions = [a]
        elif qa == best_q:
            best_actions.append(a)

    return rng.choice(best_actions)


def _epsilon_soft_probs(q_row, epsilon, rng):
    """
    Given Q(s, :) return an epsilon-soft probability distribution over actions.
    """
    nA = len(q_row)

    # random tie-breaking greedy action
    best_a = _argmax_random_tie(q_row, rng)

    # epsilon-soft distribution:
    #   pi(best) = 1 - eps + eps/|A|
    #   pi(other)= eps/|A|
    p = [epsilon / float(nA)] * nA
    p[best_a] += (1.0 - epsilon)
    return p


def _sample_from_probs(rng, probs):
    """
    Sample an index according to probs (list of nonnegative numbers summing to 1).
    """
    x = rng.random()
    cum = 0.0
    for i, p in enumerate(probs):
        cum += p
        if x <= cum:
            return i
    return len(probs) - 1  # numerical fallback


def _epsilon_by_episode(
    ep_idx,
    num_episodes,
    epsilon_start,
    epsilon_min,
    decay_type="exp",
    decay_fraction=0.8,
    decay_rate=5.0,
):
    """
    Episode-dependent epsilon schedule - "start high and decay later".

    ep_idx: 1..num_episodes

    decay_type:
      - "exp": exponential decay from epsilon_start to epsilon_min
      - "linear": linear decay over first decay_fraction of training, then flat at epsilon_min
      - epsilon_min prevents policy from becoming fully greedy too early (and freezing).

    For exp decay:
      eps(t) = eps_min + (eps_start - eps_min) * exp(-decay_rate * t)
      where t in [0, 1].

    For linear decay:
      decays to eps_min by ep = decay_fraction * num_episodes.
    """
    if num_episodes <= 1:
        return float(epsilon_min)

    # normalize progress to [0,1]
    t = float(ep_idx - 1) / float(num_episodes - 1)

    eps_start = float(epsilon_start)
    eps_min = float(epsilon_min)

    if decay_type == "linear":
        # decay only during first part, then clamp
        cutoff = max(1e-12, float(decay_fraction))
        if t >= cutoff:
            return eps_min
        # map t in [0, cutoff] -> u in [0,1]
        u = t / cutoff
        return eps_start + (eps_min - eps_start) * u

    # default: exponential decay
    return eps_min + (eps_start - eps_min) * (2.718281828459045 ** (-float(decay_rate) * t))


def generate_episode_mc(env, Q, epsilon, max_steps, rng):
    """
    Generate one episode following the current epsilon-soft policy derived from Q.

    Returns:
        episode: list of (s, a, r) tuples with s as integer state_id.
                Each tuple is (s_t, a_t, r_{t+1})
    """
    episode = []
    s = env.reset()  # Start at fixed start state (no exploring starts!).

    for _ in range(max_steps):
        action_probs = _epsilon_soft_probs(Q[s], epsilon, rng)
        a = _sample_from_probs(rng, action_probs)

        ns, r, done, _ = env.step(a)
        episode.append((s, a, r))

        s = ns
        if done:
            break

    return episode


def mc_control_first_visit_no_exploring_starts(
    env,
    num_episodes=20000,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_min=0.10,
    epsilon_decay_type="exp",   # "exp" or "linear"
    epsilon_decay_fraction=0.8, # used only for linear decay
    epsilon_decay_rate=5.0,     # used only for exp decay
    max_steps_per_episode=200,
    seed=0,
    verbose_every=2000
):
    """
    First-visit Monte Carlo control without exploring starts (on-policy control).

    CHANGES MADE:
      1) RANDOM tie-breaking when selecting greedy actions (inside epsilon-soft + greedy eval).
      2) EPSILON DECAY: epsilon starts high (epsilon_start) and decays towards epsilon_min.

    """
    rng = random.Random(seed)

    nS = env.num_states()
    nA = env.num_actions()

    Q = [[0.0 for _ in range(nA)] for _ in range(nS)]
    N = [[0 for _ in range(nA)] for _ in range(nS)]

    for ep in range(1, num_episodes + 1):
        # --- epsilon decay per episode ---
        eps_t = _epsilon_by_episode(
            ep_idx=ep,
            num_episodes=num_episodes,
            epsilon_start=epsilon_start,
            epsilon_min=epsilon_min,
            decay_type=epsilon_decay_type,
            decay_fraction=epsilon_decay_fraction,
            decay_rate=epsilon_decay_rate,
        )

        episode = generate_episode_mc(
            env=env,
            Q=Q,
            epsilon=eps_t,
            max_steps=max_steps_per_episode,
            rng=rng
        )

        # Record earliest time each (s,a) appears (first-visit)
        first_idx = {}
        for idx, (s_t, a_t, _) in enumerate(episode):
            if (s_t, a_t) not in first_idx:
                first_idx[(s_t, a_t)] = idx

        # Backward return computation + first-visit updates
        G = 0.0
        for t in range(len(episode) - 1, -1, -1):
            s_t, a_t, r_tp1 = episode[t]
            G = r_tp1 + gamma * G

            if first_idx.get((s_t, a_t), None) == t:
                N[s_t][a_t] += 1
                Q[s_t][a_t] += (G - Q[s_t][a_t]) / float(N[s_t][a_t])

        if verbose_every is not None and ep % int(verbose_every) == 0:
            sr = evaluate_greedy_policy(
                env, Q, episodes=200, max_steps=max_steps_per_episode, seed=seed + ep
            )
            print(
                "[MC FV] episode={} | eps={:.4f} | greedy success_rate={:.3f}".format(ep, eps_t, sr)
            )

    # Final policy table: use epsilon_min (end-of-training behavior)
    policy = []
    for s in range(nS):
        policy.append(_epsilon_soft_probs(Q[s], epsilon_min, rng))

    return Q, policy


def greedy_action(Q_s, rng=None):
    """
    Greedy action with RANDOM tie-breaking.
    If rng is None, uses a local RNG (still random, but not reproducible across calls).
    """
    if rng is None:
        rng = random.Random()
    return _argmax_random_tie(Q_s, rng)


def evaluate_greedy_policy(env, Q, episodes=200, max_steps=200, seed=123):
    """
    Evaluate the greedy policy induced by Q with random tie-breaking.
    """
    rng = random.Random(seed)
    success = 0

    for _ in range(episodes):
        s = env.reset()

        for _ in range(max_steps):
            a = greedy_action(Q[s], rng=rng)
            ns, r, done, _ = env.step(a)
            s = ns
            if done:
                if r == 1.0:
                    success += 1
                break

    return success / float(episodes)


def print_greedy_policy_grid(env, Q, seed=0):
    """
    Print the best action at each grid cell with random tie-breaking.

    """
    rng = random.Random(seed)

    n = env.n
    arrows = {env.LEFT: "L", env.DOWN: "D", env.RIGHT: "R", env.UP: "U"}

    for r in range(n):
        row = []
        for c in range(n):
            pos = (r, c)
            if pos == env.start:
                row.append("S")
            elif pos == env.goal:
                row.append("G")
            elif pos in env.holes:
                row.append("H")
            else:
                s = env.pos_to_state(pos)
                a = greedy_action(Q[s], rng=rng)
                row.append(arrows[a])
        print(" ".join(row))


