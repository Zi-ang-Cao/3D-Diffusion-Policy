import os
import numpy as np
import zarr
import click
from tqdm import tqdm
from diffusion_policy_3d.env.pushT.sim_pusht.pusht_pc_env import PushTPCEnv
# from diffusion_policy_3d.env.pushT.sim_pusht.pusht_dense_pc_env import PushTDensePCEnv
from termcolor import cprint
import copy

class AttrDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"{key} not found")

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"{key} not found")


@click.command()
@click.option("--in_dir", type=str)
@click.option("--num_demos", type=int)
@click.option("--obs_mode", type=str, default="pc")
@click.option("--ac_mode", type=str, default="rel")
@click.option("--codebase", type=str, default="dp3")
def main(in_dir, num_demos, obs_mode, ac_mode, codebase):
    SAVE_TO_ZARR = True if codebase == "dp3" else False
    # load environment for replay
    args = AttrDict(
        legacy=False,
        block_cog=None,
        damping=None,
        render_size=96,
        max_episode_length=300,
        randomize_rotation=False,
        scale_low=1.0,
        scale_high=1.0,
        scale_aspect_limit=100.0,
        uniform_scaling=True,
        obs_mode=obs_mode,
        ac_mode=ac_mode,
        seed=0,
        num_points=8,
    )
    env = PushTPCEnv(args)
    # env = PushTDensePCEnv(args)
    print(env.name)
    print(env.metadata)
    env.reset()

    data_dir = os.path.join(os.path.dirname(__file__), '../../../../../data')
    save_dir = os.path.join(data_dir, 'pusht_cornerpc_expert.zarr')
    if os.path.exists(save_dir):
        cprint('Data already exists at {}'.format(save_dir), 'red')
        cprint("If you want to overwrite, delete the existing directory first.", "red")
        cprint("Do you want to overwrite? (y/n)", "red")
        user_input = 'y'
        if user_input == 'y':
            cprint('Overwriting {}'.format(save_dir), 'red')
            os.system('rm -rf {}'.format(save_dir))
        else:
            cprint('Exiting', 'red')
            return
    os.makedirs(save_dir, exist_ok=True)

    data = zarr.open(in_dir, mode="r")
    episode_ends = data["meta"]["episode_ends"][...]
    actions = data["data"]["action"][...]
    states = data["data"]["state"][...]
    images = data["data"]["img"][...]


    point_cloud_arrays = []
    state_arrays = []
    action_arrays = []
    episode_ends_arrays = []

    point_cloud_arrays_sub = []
    state_arrays_sub = []
    action_arrays_sub = []
    total_count = 0
    total_count_sub = 0

    episode_index = 0
    episode_t = 0
    num_steps = len(states)
    if num_demos <= len(episode_ends):
        num_steps = episode_ends[num_demos]

    # 200 * 10 frames
    for i in tqdm(range(num_steps), desc="Steps"):
        save_episode = False
        early_exit = False
        if episode_ends[episode_index] == i:
            episode_index += 1
            save_episode = True
            if episode_index == num_demos:
                early_exit = True
        
        if episode_t == 0 and i > 0:
            save_episode = True

        if save_episode:
            point_cloud_arrays.extend(copy.deepcopy(point_cloud_arrays_sub))
            state_arrays.extend(copy.deepcopy(state_arrays_sub))    
            action_arrays.extend(copy.deepcopy(action_arrays_sub))
            total_count += total_count_sub
            episode_ends_arrays.append(copy.deepcopy(total_count)) # the index of the last step of the episode    

            cprint('Episode: {}, Success Times: {}'.format(episode_index, total_count_sub), 'green')

            point_cloud_arrays_sub = []
            state_arrays_sub = []
            action_arrays_sub = []
            total_count_sub = 0
            episode_t = 0

        if early_exit:
            break
            
        env._set_state(states[i])
        env.render_cache = None
        render = env.render()

        if SAVE_TO_ZARR:
            obs_pc = render["pc"]
            obs_state = env._get_obs()  # (1, 9)
            obs_state = obs_state.squeeze(0)  # (9,)
            action = actions[i]

            point_cloud_arrays_sub.append(obs_pc)
            state_arrays_sub.append(obs_state)
            action_arrays_sub.append(action)
        
        total_count_sub += 1
        episode_t += 1


    # save data
    ###############################
    # save data
    ###############################

    # keys=['state', 'action', 'point_cloud']
    """
    --------------------------------------------------
    point_cloud shape: (24892, 8, 3), range: [0.0, 518.9389640463407]
    state shape: (24892, 9), range: [0.0, 510.9578857421875]
    action shape: (24892, 2), range: [12.0, 511.0]
    """

    # create zarr file
    zarr_root = zarr.group(save_dir)
    zarr_data = zarr_root.create_group('data')
    zarr_meta = zarr_root.create_group('meta')
    # save img, state, action arrays into data, and episode ends arrays into meta
    state_arrays = np.stack(state_arrays, axis=0)
    point_cloud_arrays = np.stack(point_cloud_arrays, axis=0)
    action_arrays = np.stack(action_arrays, axis=0)
    episode_ends_arrays = np.array(episode_ends_arrays)

    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
    state_chunk_size = (100, state_arrays.shape[1])
    point_cloud_chunk_size = (100, point_cloud_arrays.shape[1], point_cloud_arrays.shape[2])
    action_chunk_size = (100, action_arrays.shape[1])
    zarr_data.create_dataset('state', data=state_arrays, chunks=state_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('point_cloud', data=point_cloud_arrays, chunks=point_cloud_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('action', data=action_arrays, chunks=action_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_meta.create_dataset('episode_ends', data=episode_ends_arrays, dtype='int64', overwrite=True, compressor=compressor)

    cprint(f'-'*50, 'cyan')
    # print shape
    cprint(f'point_cloud shape: {point_cloud_arrays.shape}, range: [{np.min(point_cloud_arrays)}, {np.max(point_cloud_arrays)}]', 'green')
    cprint(f'state shape: {state_arrays.shape}, range: [{np.min(state_arrays)}, {np.max(state_arrays)}]', 'green')
    cprint(f'action shape: {action_arrays.shape}, range: [{np.min(action_arrays)}, {np.max(action_arrays)}]', 'green')
    cprint(f'Saved zarr file to {save_dir}', 'green')

    # clean up
    del state_arrays, point_cloud_arrays, action_arrays, episode_ends_arrays
    del zarr_root, zarr_data, zarr_meta

if __name__ == "__main__":
    main()
