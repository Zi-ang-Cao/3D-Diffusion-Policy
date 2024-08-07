import wandb
import numpy as np
import torch
import collections
from diffusion_policy_3d.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy_3d.gym_util.video_recording_wrapper import SimpleVideoRecordingWrapper

from diffusion_policy_3d.policy.base_policy import BasePolicy
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.env_runner.base_runner import BaseRunner
import diffusion_policy_3d.common.logger_util as logger_util
from termcolor import cprint

import os
import numpy as np
import zarr
import click
from tqdm import tqdm
from diffusion_policy_3d.env.pushT.sim_pusht.pusht_pc_env import PushTPCEnv
from diffusion_policy_3d.env.pushT.sim_pusht.utils.process_data_for_dp3 import AttrDict
from termcolor import cprint
import copy

import cv2
import skvideo.io
import imageio

def _make_dir(filename):
    folder = os.path.dirname(filename)
    os.makedirs(folder, exist_ok=True)

def save_images_to_video(video_frames, filename, fps=10, video_format="mp4"):
    if len(video_frames) == 0:
        return False

    assert fps == int(fps), fps
    _make_dir(filename)

    skvideo.io.vwrite(
        filename,
        video_frames,
        inputdict={
            "-r": str(int(fps)),
        },
        outputdict={"-f": video_format, "-pix_fmt": "yuv420p"},
    )

    return True


class PushTRunner(BaseRunner):
    def __init__(self,
                 output_dir,
                 seed,
                 eval_episodes=20,
                 max_steps=1000,
                 n_obs_steps=8,
                 n_action_steps=8,
                 fps=10,
                 crf=22,
                 render_size=96,
                 tqdm_interval_sec=5.0,
                 n_envs=None,
                 task_name=None,
                 n_train=None,
                 n_test=None,
                 device="cuda:0",
                 use_point_crop=True,
                 num_points=512
                 ):
        super().__init__(output_dir)
        self.task_name = task_name
        self.seed = seed
        self.render_size = render_size
        self.for_eval = False


        def env_fn(task_name, seed, render_size):
            args = AttrDict(
                legacy=False,
                block_cog=None,
                damping=None,
                render_size=render_size,
                max_episode_length=300,
                randomize_rotation=False,
                scale_low=1.0,
                scale_high=1.0,
                scale_aspect_limit=100.0,
                uniform_scaling=True,
                obs_mode="pc2",
                ac_mode="abs",
                seed=seed,
                num_points=8,
            )
            
            return MultiStepWrapper(
                SimpleVideoRecordingWrapper(
                    PushTPCEnv(args, for_dp3_runner=True)),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps,
                reward_agg_method='last',
            )
        self.eval_episodes = eval_episodes
        self.env = env_fn(self.task_name, self.seed, self.render_size)

        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec

        self.logger_util_test = logger_util.LargestKRecorder(K=3)
        self.logger_util_test10 = logger_util.LargestKRecorder(K=5)

    def run(self, policy: BasePolicy, save_video=False, epoch=-1):
        device = policy.device
        dtype = policy.dtype

        all_traj_rewards = []
        all_success_rates = []
        env = self.env

        if self.for_eval:
            # save_video = True
            save_video = False
            eval_episodes = self.eval_episodes
        else:
            save_video = False
            # For training, we only need to evaluate one episode when epoch < 1600
            if epoch >=1600:
                print(f"Epoch {epoch} is greater than 1600, so setting eval_episodes to self.eval_episodes -- real eval")
                eval_episodes = self.eval_episodes
            else:
                print(f"Epoch {epoch} is less than 1600, so setting eval_episodes to 1 -- saving time")
                eval_episodes = 1
        
        for episode_idx in tqdm(range(eval_episodes), desc=f"Eval in PushT {self.task_name} Pointcloud Env", leave=False, mininterval=self.tqdm_interval_sec):
            
            # update seed
            self.env.update_seed(self.seed * 100 + episode_idx)
            
            # start rollout
            obs = env.reset()
            policy.reset()

            done = False
            traj_reward = 0
            is_success = False
            while not done:
                np_obs_dict = dict(obs)
                obs_dict = dict_apply(np_obs_dict,
                                      lambda x: torch.from_numpy(x).to(
                                          device=device))

                with torch.no_grad():
                    obs_dict_input = {}
                    obs_dict_input['point_cloud'] = obs_dict['point_cloud'].unsqueeze(0)
                    obs_dict_input['agent_pos'] = obs_dict['agent_pos'].unsqueeze(0)
                    # obs_dict_input = dict_keys(['point_cloud', 'agent_pos'])
                    # torch.Size([1, 2, 512, 6]), torch.Size([1, 2, 9])
                    action_dict = policy.predict_action(obs_dict_input)

                np_action_dict = dict_apply(action_dict,
                                            lambda x: x.detach().to('cpu').numpy())
                action = np_action_dict['action'].squeeze(0)

                obs, reward, done, info = env.step(action)

                traj_reward = reward
                done = np.all(done)
                is_success_from_info = int(traj_reward > 0.9)
                is_success = is_success or is_success_from_info

            all_success_rates.append(is_success)
            all_traj_rewards.append(traj_reward)

            cprint(f"Finish the episode_idx={episode_idx} with traj_reward={traj_reward}", 'red')

            if save_video:
                video_frames = env.env.get_video()
                if len(video_frames.shape) == 5:
                    video_frames = video_frames[:, 0]  # select first frame
            
                filename = os.path.join(self.output_dir, f"sim_video_eval_for_{episode_idx}.mp4")

                save_images_to_video(video_frames, filename, fps=10, video_format="mp4")


        max_rewards = collections.defaultdict(list)
        log_data = dict()

        log_data['mean_traj_rewards'] = np.mean(all_traj_rewards)
        cprint(f"mean_traj_rewards: {np.mean(all_traj_rewards)}", 'green')

        log_data['mean_success_rates'] = np.mean(all_success_rates)

        log_data['test_mean_score'] = np.mean(all_success_rates)
        
        cprint(f"test_mean_score: {np.mean(all_success_rates)}", 'green')

        self.logger_util_test.record(np.mean(all_success_rates))
        self.logger_util_test10.record(np.mean(all_success_rates))
        log_data['SR_test_L3'] = self.logger_util_test.average_of_largest_K()
        log_data['SR_test_L5'] = self.logger_util_test10.average_of_largest_K()

        
        
        _ = env.reset()
        videos = None

        return log_data
