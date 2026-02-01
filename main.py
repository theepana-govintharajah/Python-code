import time
from frozen_lake_env import FrozenLakeEnv
from frozen_lake_render import FrozenLakeMatplotlibRenderer

if __name__ == "__main__":
    holes = {(1, 1), (1, 3), (2, 3), (3, 0)}
    env = FrozenLakeEnv(n=4, holes=holes, seed=40)
    s = env.reset()

    renderer = FrozenLakeMatplotlibRenderer(env, bg_image_path=None, pause=0.30, title="FrozenLake 4x4")

    # Initial print + draw
    print("Initial state:", s, "pos:", env.s)
    env.render()              # ASCII grid
    renderer.draw()           # GUI window

    for t in range(20):
        a = env.sample_action()
        ns, r, done, _ = env.step(a)

        # Terminal debug (this is what you want)
        print(f"[t={t}] a={FrozenLakeEnv.ACTION_NAMES[a]} ({a})  ns={ns}  r={r}  done={done}  pos={env.s}")
        env.render()

        # GUI debug
        renderer.draw(action=a, reward=r, done=done)

        if done:
            print("Episode finished.")
            break

    print("Close plot window to exit.")
    import matplotlib.pyplot as plt
    plt.ioff()
    plt.show()
