import sys
import time
import matplotlib.pyplot as plt

# Deterministic environment
from frozen_lake_env import FrozenLakeEnv as FrozenLakeEnvDet, generate_random_solvable_holes

# Probabilistic environment 
from environment_probablistic import FrozenLakeEnv as FrozenLakeEnvProb

from frozen_lake_render import FrozenLakeMatplotlibRenderer

# ---- Original MC implementation ----
from monte_carlo import (
    mc_control_first_visit_no_exploring_starts as mc_control_base,
    greedy_action as greedy_action_base,
    print_greedy_policy_grid as print_policy_base,
    evaluate_greedy_policy as eval_greedy_base,
)

# ---- Updated MC implementation (random tie-break + epsilon decay) ----
from monte_carlo_update import (
    mc_control_first_visit_no_exploring_starts as mc_control_up,
    greedy_action as greedy_action_up,
    print_greedy_policy_grid as print_policy_up,
    evaluate_greedy_policy as eval_greedy_up,
)


def run_episode_with_render(env, Q, renderer, greedy_action_fn, max_steps=200, pause=0.30):
    """
    Run ONE episode using the greedy policy from Q.
    Updates matplotlib renderer each step.
    """
    s = env.reset()

    print("Initial state_id:", s, "pos:", env.s)
    env.render()        # ASCII
    renderer.draw()     # GUI initial draw

    for t in range(max_steps):
        a = greedy_action_fn(Q[s])
        ns, r, done, info = env.step(a)

        # If probabilistic env, show executed action info too
        extra = ""
        if isinstance(info, dict) and "executed_action" in info:
            extra = f" | executed={env.ACTION_NAMES[info['executed_action']]} slipped={info.get('slipped', False)}"

        print(
            f"[t={t}] a={env.ACTION_NAMES[a]} ({a})  ns={ns}  r={r}  done={done}  pos={env.s}{extra}"
        )

        env.render()
        renderer.draw(action=a, reward=r, done=done)

        s = ns

        if pause is not None and pause > 0:
            time.sleep(pause)

        if done:
            print("Episode finished.")
            break

    print("Close plot window to exit.")
    plt.ioff()
    plt.show()


def run_mc_experiment(
    env_class,
    n,
    holes,
    env_seed,
    mc_control_fn,
    greedy_action_fn,
    print_policy_fn,
    eval_fn,
    mc_kwargs=None,
    # Common experiment settings:
    train_episodes=30000,
    gamma=0.99,
    max_steps_train=100,
    eval_episodes=2000,
    max_steps_eval=100,
    verbose_every=3000,
    render_one_episode=True,
    render_pause=0.30,
    env_kwargs=None,
):
    """
    Runs MC control on either deterministic or probabilistic environment,
    depending on env_class passed in.

    """
    env_kwargs = env_kwargs or {}
    mc_kwargs = mc_kwargs or {}

    print("\n" + "=" * 80)
    print(f"Running MC experiment: grid={n}x{n} | holes={len(holes)} | hole_ratio={len(holes)/(n*n):.2%}")
    print(f"Environment: {env_class.__name__} | extra_args={env_kwargs}")
    print(f"MC implementation: {mc_control_fn.__module__}.{mc_control_fn.__name__} | mc_kwargs={mc_kwargs}")
    print("=" * 80)

    env = env_class(n=n, holes=holes, seed=env_seed, **env_kwargs)

    # --- Train ---
    Q, pi = mc_control_fn(
        env,
        num_episodes=train_episodes,
        gamma=gamma,
        max_steps_per_episode=max_steps_train,
        seed=0,
        verbose_every=verbose_every,
        **mc_kwargs
    )

    # --- Print policy ---
    print("\nFinal greedy policy (grid):")
    print_policy_fn(env, Q)

    # --- Evaluate ---
    sr = eval_fn(env, Q, episodes=eval_episodes, max_steps=max_steps_eval, seed=999)
    print(f"\nFinal greedy success rate over {eval_episodes} episodes: {sr:.3f}")

    # --- Render one episode (optional) ---
    if render_one_episode:
        renderer = FrozenLakeMatplotlibRenderer(
            env,
            bg_image_path=None,
            pause=render_pause,
            title=f"FrozenLake {n}x{n} (Greedy after MC)"
        )
        run_episode_with_render(env, Q, renderer, greedy_action_fn, max_steps=max_steps_eval, pause=render_pause)


def main(mode: str):
    mode = mode.strip()

    # Task 1: 4x4 fixed holes
    holes_4x4 = {(1, 1), (1, 3), (2, 3), (3, 0)}

    # Task 2: 10x10 random solvable holes (use deterministic generator)
    holes_10x10 = generate_random_solvable_holes(
        n=10,
        hole_ratio=0.25,
        seed=123,
        start=(0, 0),
        goal=(9, 9),
        max_tries=5000
    )

    # -------------------- ORIGINAL MC (fixed epsilon) --------------------
    if mode == "detMC4":
        run_mc_experiment(
            env_class=FrozenLakeEnvDet,
            n=4,
            holes=holes_4x4,
            env_seed=40,
            mc_control_fn=mc_control_base,
            greedy_action_fn=greedy_action_base,
            print_policy_fn=print_policy_base,
            eval_fn=eval_greedy_base,
            mc_kwargs={"epsilon": 0.10},
            train_episodes=30000,
            gamma=0.99,
            max_steps_train=100,
            eval_episodes=2000,
            max_steps_eval=100,
            verbose_every=3000,
            render_one_episode=True,
            render_pause=0.30,
            env_kwargs={}
        )

    elif mode == "probMC4":
        run_mc_experiment(
            env_class=FrozenLakeEnvProb,
            n=4,
            holes=holes_4x4,
            env_seed=40,
            mc_control_fn=mc_control_base,
            greedy_action_fn=greedy_action_base,
            print_policy_fn=print_policy_base,
            eval_fn=eval_greedy_base,
            mc_kwargs={"epsilon": 0.10},
            train_episodes=30000,
            gamma=0.99,
            max_steps_train=100,
            eval_episodes=2000,
            max_steps_eval=100,
            verbose_every=3000,
            render_one_episode=True,
            render_pause=0.30,
            env_kwargs={"slip_prob": 0.2}
        )

    elif mode == "probMC4-moreTrain":
        run_mc_experiment(
            env_class=FrozenLakeEnvProb,
            n=4,
            holes=holes_4x4,
            env_seed=40,
            mc_control_fn=mc_control_base,
            greedy_action_fn=greedy_action_base,
            print_policy_fn=print_policy_base,
            eval_fn=eval_greedy_base,
            mc_kwargs={"epsilon": 0.10},
            train_episodes=80000,
            gamma=0.99,
            max_steps_train=100,
            eval_episodes=2000,
            max_steps_eval=100,
            verbose_every=10000,
            render_one_episode=True,
            render_pause=0.30,
            env_kwargs={"slip_prob": 0.2}
        )

    elif mode == "detMC10":
        run_mc_experiment(
            env_class=FrozenLakeEnvDet,
            n=10,
            holes=holes_10x10,
            env_seed=123,
            mc_control_fn=mc_control_base,
            greedy_action_fn=greedy_action_base,
            print_policy_fn=print_policy_base,
            eval_fn=eval_greedy_base,
            mc_kwargs={"epsilon": 0.10},
            train_episodes=80000,
            gamma=0.99,
            max_steps_train=300,
            eval_episodes=2000,
            max_steps_eval=300,
            verbose_every=10000,
            render_one_episode=True,
            render_pause=0.10,
            env_kwargs={}
        )

    elif mode == "probMC10":
        run_mc_experiment(
            env_class=FrozenLakeEnvProb,
            n=10,
            holes=holes_10x10,
            env_seed=123,
            mc_control_fn=mc_control_base,
            greedy_action_fn=greedy_action_base,
            print_policy_fn=print_policy_base,
            eval_fn=eval_greedy_base,
            mc_kwargs={"epsilon": 0.10},
            train_episodes=80000,
            gamma=0.99,
            max_steps_train=300,
            eval_episodes=2000,
            max_steps_eval=300,
            verbose_every=10000,
            render_one_episode=True,
            render_pause=0.10,
            env_kwargs={"slip_prob": 0.2}
        )

    # -------------------- UPDATED MC (random tie-break + epsilon decay) --------------------
    elif mode == "detMCup4":
        run_mc_experiment(
            env_class=FrozenLakeEnvDet,
            n=4,
            holes=holes_4x4,
            env_seed=40,
            mc_control_fn=mc_control_up,
            greedy_action_fn=greedy_action_up,
            print_policy_fn=print_policy_up,
            eval_fn=eval_greedy_up,
            # epsilon decay knobs
            mc_kwargs={
                "epsilon_start": 0.40,
                "epsilon_min": 0.15,
                "epsilon_decay_type": "exp",   # "exp" or "linear"
                "epsilon_decay_rate": 1.5,     # higher = faster decay
                "epsilon_decay_fraction": 0.8  # only used if decay_type == "linear"
            },
            train_episodes=30000,
            gamma=0.99,
            max_steps_train=100,
            eval_episodes=2000,
            max_steps_eval=100,
            verbose_every=3000,
            render_one_episode=True,
            render_pause=0.30,
            env_kwargs={}
        )

    elif mode == "detMCup10":
        run_mc_experiment(
            env_class=FrozenLakeEnvDet,
            n=10,
            holes=holes_10x10,
            env_seed=123,
            mc_control_fn=mc_control_up,
            greedy_action_fn=greedy_action_up,
            print_policy_fn=print_policy_up,
            eval_fn=eval_greedy_up,
            mc_kwargs={
                "epsilon_start": 0.6,
                "epsilon_min": 0.2,
                "epsilon_decay_type": "exp",
                "epsilon_decay_rate": 1.0,
                "epsilon_decay_fraction": 0.8
            },
            train_episodes=80000,
            gamma=0.99,
            max_steps_train=300,
            eval_episodes=2000,
            max_steps_eval=300,
            verbose_every=10000,
            render_one_episode=True,
            render_pause=0.10,
            env_kwargs={}
        )
    elif mode == "probMCup4":
        run_mc_experiment(
            env_class=FrozenLakeEnvProb,
            n=4,
            holes=holes_4x4,
            env_seed=40,
            mc_control_fn=mc_control_up,
            greedy_action_fn=greedy_action_up,
            print_policy_fn=print_policy_up,
            eval_fn=eval_greedy_up,
            mc_kwargs={
                "epsilon_start": 0.40,
                "epsilon_min": 0.15,
                "epsilon_decay_type": "exp",
                "epsilon_decay_rate": 1.5,
                "epsilon_decay_fraction": 0.8
            },
            train_episodes=30000,
            gamma=0.99,
            max_steps_train=100,
            eval_episodes=2000,
            max_steps_eval=100,
            verbose_every=3000,
            render_one_episode=True,
            render_pause=0.30,
            env_kwargs={"slip_prob": 0.2}
        )

    elif mode == "probMCup10":
        run_mc_experiment(
            env_class=FrozenLakeEnvProb,
            n=10,
            holes=holes_10x10,
            env_seed=123,
            mc_control_fn=mc_control_up,
            greedy_action_fn=greedy_action_up,
            print_policy_fn=print_policy_up,
            eval_fn=eval_greedy_up,
            mc_kwargs={
                "epsilon_start": 0.6,
                "epsilon_min": 0.2,
                "epsilon_decay_type": "exp",
                "epsilon_decay_rate": 1.0,
                "epsilon_decay_fraction": 0.8
            },
            train_episodes=80000,
            gamma=0.99,
            max_steps_train=300,
            eval_episodes=2000,
            max_steps_eval=300,
            verbose_every=10000,
            render_one_episode=True,
            render_pause=0.10,
            env_kwargs={"slip_prob": 0.2}
        )

    else:
        print("Invalid mode:", mode)
        print(
            "Valid modes:\n"
            "  detMC4 | detMC10 | probMC4 | probMC10 | probMC4-moreTrain\n"
            "  detMCup4 | detMCup10 | probMCup4 | probMCup10"
        )
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <mode>")
        print(
            "Modes:\n"
            "  detMC4 | detMC10 | probMC4 | probMC10 | probMC4-moreTrain\n"
            "  detMCup4 | detMCup10 | probMCup4 | probMCup10"
        )
        sys.exit(1)

    main(sys.argv[1])
