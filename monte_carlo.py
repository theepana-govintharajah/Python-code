import random
from frozen_lake_env import FrozenLakeEnv

def _epsilon_soft_probs(q_row, epsilon):
    """
    Given Q(s, :) as a list/array of length |A|, return an epsilon-soft
    probability distribution over actions.
    Args:
        q_row: list/array of Q(s,a) for all a in state s
        epsilon: exploration rate in [0, 1]
    """
    nA = len(q_row) # number of actions

    # greedy action
    best_a = 0 # initialization of best action
    best_q = q_row[0] # initialization of best Q-value

    for a in range(1, nA):
        if q_row[a] > best_q: # found better action
            best_q = q_row[a]
            best_a = a

    # epsilon-soft distribution:
    #   pi(best) = 1 - eps + eps/|A|
    #   pi(other)= eps/|A|

    p = [epsilon / float(nA)] * nA # start with uniform small prob for all actions
    p[best_a] += (1.0 - epsilon) # boost best action probability
    return p # return probability distribution list


def _sample_from_probs(rng, probs):
    """
    Sample an index according to probs (list of nonnegative numbers that sum to 1).
    """
    x = rng.random()
    cum = 0.0 # cumulative probability 

    for i, p in enumerate(probs): # iterate over actions
        cum += p
        if x <= cum: 
            return i
    # numerical fallback
    return len(probs) - 1


def generate_episode_mc(env, Q, epsilon, max_steps, rng):
    """
    Generate one episode following the current epsilon-soft policy derived from Q.

    Returns:
        episode: list of (s, a, r) tuples with s as integer state_id.
                Each tuple is the transition outcome after taking action a at state s:
                (s_t, a_t, r_{t+1})
    """
    episode = []

    s = env.reset() # Start at fixed start state (no exploring starts!).

    for _ in range(max_steps):
        # choose action from epsilon-soft policy derived from Q at state s
        action_probs = _epsilon_soft_probs(Q[s], epsilon) # get epsilon-soft probabilities for state s
        a = _sample_from_probs(rng, action_probs) # sample action a according to probs

        ns, r, done, _ = env.step(a) # take action a, observe next state ns, reward r, done flag

        episode.append((s, a, r)) # record transition (s_t, a_t, r_{t+1})

        s = ns # move to next state
        if done: #If hit goal or hole, episode ends.
            break

    return episode


def mc_control_first_visit_no_exploring_starts(
    env,
    num_episodes=20000,
    gamma=0.99, # discount factor
    epsilon=0.10,
    max_steps_per_episode=200,
    seed=0,
    verbose_every=2000
):
    
    """
    First-visit Monte Carlo control without exploring starts (on-policy control).

    What this function does:
      1) Initialize Q(s,a) arbitrarily (here: zeros).
      2) Maintain running average of returns for each (s,a).
         Instead of storing full Returns(s,a) lists (memory-heavy),
         we store:
             N[s][a] = number of first-visit samples seen for (s,a)
             Q[s][a] = running mean of observed returns
         Update rule for running mean:
             N <- N + 1
             Q <- Q + (G - Q)/N
      3) For each episode:
         - generate episode using epsilon-soft policy from current Q
         - walk backward computing G
         - for each (s,a) encountered FIRST time in that episode:
              update Q(s,a) using that G
         - policy improvement is implicit: next episode uses epsilon-soft from updated Q

    Returns:
        Q: state-action value table, shape [num_states][num_actions]
        policy: epsilon-soft policy table, shape [num_states][num_actions]
                (computed from final Q)
    """
    rng = random.Random(seed)

    nS = env.num_states()
    nA = env.num_actions()

    # Q(s,a) initialized to zero
    Q = [[0.0 for _ in range(nA)] for _ in range(nS)]

    # N(s,a): number of first-visit returns used so far
    N = [[0 for _ in range(nA)] for _ in range(nS)]

    for ep in range(1, num_episodes + 1):
        episode = generate_episode_mc(
            env=env,
            Q=Q,
            epsilon=epsilon,
            max_steps=max_steps_per_episode,
            rng=rng
        )

        # Earliest time each (s,a) appears in that episode
        first_idx = {}
        # Traverse episode forward: record first occurrence of (s,a)
        for idx, (s_t, a_t, _) in enumerate(episode):
            if (s_t, a_t) not in first_idx:
                first_idx[(s_t, a_t)] = idx

        # Now compute returns backward and update only when t is that earliest index
        G = 0.0
        # Traverse episode backward 
        for t in range(len(episode) - 1, -1, -1):
            s_t, a_t, r_tp1 = episode[t]
            G =  r_tp1 + gamma * G # update return G

            # update ONLY at the first-visit time (earliest occurrence)
            if first_idx.get((s_t, a_t), None) == t:
                N[s_t][a_t] += 1
                # incremental average update (running mean)
                Q[s_t][a_t] += (G - Q[s_t][a_t]) / float(N[s_t][a_t])

        # Progress logging
        if verbose_every is not None and ep % int(verbose_every) == 0:
            # quick diagnostic: greedy success rate estimate (small rollout batch)
            sr = evaluate_greedy_policy(env, Q, episodes=200, max_steps=max_steps_per_episode, seed=seed + ep)
            print("[MC FV] episode={} | greedy success_rate={:.3f}".format(ep, sr))

    # Build final epsilon-soft policy table from Q
    policy = []
    for s in range(nS):
        policy.append(_epsilon_soft_probs(Q[s], epsilon))

    return Q, policy


def greedy_action(Q_s):
    """
    Return argmax_a Q(s,a). Deterministic greedy (ties -> first max).
    """
    best_a = 0
    best_q = Q_s[0]
    for a in range(1, len(Q_s)):
        if Q_s[a] > best_q:
            best_q = Q_s[a]
            best_a = a
    return best_a


def evaluate_greedy_policy(env, Q, episodes=200, max_steps=200, seed=123):
    """
    Evaluate the greedy policy induced by Q:
      a_t = argmax_a Q(s_t,a)

    Returns:
        success_rate: fraction of episodes reaching goal (reward +1 terminal)
    """

    success = 0

    for _ in range(episodes):
        s = env.reset() # start state
        done = False

        for _ in range(max_steps):
            a = greedy_action(Q[s])
            ns, r, done, _ = env.step(a)
            s = ns
            if done:
                if r == 1.0:
                    success += 1
                break
        # if max_steps reached without done, count as failure (no increment)

    return success / float(episodes)


def print_greedy_policy_grid(env, Q):
    """
    Print the best action at each grid cell.
    H holes, G goal, S start.
    For other states, prints L/D/R/U corresponding to argmax_a Q(s,a).
    """
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
                a = greedy_action(Q[s])
                row.append(arrows[a])
        print(" ".join(row))



