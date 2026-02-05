import random


def _argmax_random_tie(q_row, rng):
    """
    Argmax with RANDOM tie-breaking.
    Returns an action index among those with max Q-value, chosen uniformly at random.
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
    Epsilon-soft distribution derived from Q(s,:),
    with RANDOM tie-breaking for the greedy action.
    """
    nA = len(q_row)
    best_a = _argmax_random_tie(q_row, rng)

    p = [epsilon / float(nA)] * nA
    p[best_a] += (1.0 - epsilon)
    return p


def _sample_from_probs(rng, probs):
    """
    Sample an index according to probs (list summing to ~1).
    """
    x = rng.random()
    cum = 0.0
    for i, p in enumerate(probs):
        cum += p
        if x <= cum:
            return i
    return len(probs) - 1  # numeric fallback


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
    Episode-dependent epsilon schedule.
    - "exp": eps = eps_min + (eps_start-eps_min)*exp(-decay_rate*t)
    - "linear": decay to eps_min over first decay_fraction of training, then clamp
    """
    if num_episodes <= 1:
        return float(epsilon_min)

    t = float(ep_idx - 1) / float(num_episodes - 1)
    eps_start = float(epsilon_start)
    eps_min = float(epsilon_min)

    if decay_type == "linear":
        cutoff = max(1e-12, float(decay_fraction))
        if t >= cutoff:
            return eps_min
        u = t / cutoff
        return eps_start + (eps_min - eps_start) * u

    # exp
    return eps_min + (eps_start - eps_min) * (2.718281828459045 ** (-float(decay_rate) * t))


def greedy_action(Q_s, rng=None):
    """
    Greedy action with RANDOM tie-breaking.
    """
    if rng is None:
        rng = random.Random()
    return _argmax_random_tie(Q_s, rng)


def evaluate_greedy_policy(env, Q, episodes=200, max_steps=200, seed=123):
    """
    Evaluate greedy policy induced by Q with random tie-breaking.
    Success = reaching goal (reward +1).
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
    Print best action at each cell (random tie-breaking).
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


def sarsa_control_epsilon_greedy(
    env,
    num_episodes=30000,
    gamma=0.99,
    alpha=0.10,
    epsilon=0.10,
    use_epsilon_decay=False,
    epsilon_start=1.0,
    epsilon_min=0.10,
    epsilon_decay_type="exp",      # "exp" or "linear"
    epsilon_decay_fraction=0.8,    # only for linear
    epsilon_decay_rate=5.0,        # only for exp
    max_steps_per_episode=200,
    seed=0,
    verbose_every=3000,
):
    """
    SARSA (on-policy TD control) with epsilon-greedy behavior policy.

      Initialize S
      Choose A ~ pi(.|S) (epsilon-greedy from Q)
      Repeat:
        Take A, observe R, S'
        Choose A' ~ pi(.|S')
        Q(S,A) <- Q(S,A) + alpha * [ R + gamma*Q(S',A') - Q(S,A) ]
        S <- S', A <- A'
      until terminal

    """
    rng = random.Random(seed)

    nS = env.num_states()
    nA = env.num_actions()

    # Q init
    Q = [[0.0 for _ in range(nA)] for _ in range(nS)]

    for ep in range(1, num_episodes + 1):
        # epsilon schedule
        if use_epsilon_decay:
            eps_t = _epsilon_by_episode(
                ep_idx=ep,
                num_episodes=num_episodes,
                epsilon_start=epsilon_start,
                epsilon_min=epsilon_min,
                decay_type=epsilon_decay_type,
                decay_fraction=epsilon_decay_fraction,
                decay_rate=epsilon_decay_rate,
            )
        else:
            eps_t = float(epsilon)

        s = env.reset()

        # Choose initial action A from S using epsilon-greedy derived from Q
        probs = _epsilon_soft_probs(Q[s], eps_t, rng)
        a = _sample_from_probs(rng, probs)

        for _ in range(max_steps_per_episode):
            ns, r, done, _ = env.step(a)

            if done:
                # terminal: target is just r (since Q(terminal,.) treated as 0)
                td_target = r
                Q[s][a] += alpha * (td_target - Q[s][a])
                break

            # Choose A' from S' using same epsilon-greedy behavior policy
            probs2 = _epsilon_soft_probs(Q[ns], eps_t, rng)
            a2 = _sample_from_probs(rng, probs2)

            # SARSA update
            td_target = r + gamma * Q[ns][a2]
            Q[s][a] += alpha * (td_target - Q[s][a])

            # move forward
            s, a = ns, a2

        if verbose_every is not None and ep % int(verbose_every) == 0:
            sr = evaluate_greedy_policy(env, Q, episodes=200, max_steps=max_steps_per_episode, seed=seed + ep)
            print("[SARSA] episode={} | eps={:.4f} | greedy success_rate={:.3f}".format(ep, eps_t, sr))

    final_eps = float(epsilon_min if use_epsilon_decay else epsilon)
    policy = []
    for s in range(nS):
        policy.append(_epsilon_soft_probs(Q[s], final_eps, rng))

    return Q, policy
