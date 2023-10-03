import os
from pathlib import Path

with open("replay_path.txt", "r") as f:
    replays_path = Path(f.read().strip()) / "ranked-duels"

replays = list(replays_path.glob("*.bin"))
num_replays = len(replays)

def make_gym():
    from rlgym_sim import make
    from rlgym_sim.utils.action_parsers import DiscreteAction

    from making import REWARD, TERMINAL, Obs, ReplaySetter

    return make(
        spawn_opponents=True,
        team_size=1,
        terminal_conditions=TERMINAL,
        reward_fn=REWARD,
        obs_builder=Obs(),
        action_parser=DiscreteAction(),
        state_setter=ReplaySetter(replays, num_replays),
    )

if __name__ == "__main__":
    import torch
    from rlgym_ppo import Learner

    NEW_AI = True
    RUSTICL_GPU_ACCEL = False

    # check if "libpt_ocl.so" exists in the current directory
    if RUSTICL_GPU_ACCEL and os.path.isfile("libpt_ocl.so"):
        torch.ops.load_library("libpt_ocl.so")
        device = "privateuseone:0"
    else:
        device = "auto"

    online = True
    while online:
        print("Initiating parallel model training environments...")
        load_folder = "data/checkpoints/rlgym-ppo-run-1696289578913477681/6009400"
        learner = Learner(make_gym, checkpoint_load_folder=None if NEW_AI else load_folder, n_proc=96, device=device)

        try:
            # Train our agent!
            while online:
                try:
                    print("Training model...")
                    learner.learn()
                except InterruptedError:
                    learner.save(learner.agent.cumulative_timesteps)
                    online = False
        except Exception as e:
            print(e)
