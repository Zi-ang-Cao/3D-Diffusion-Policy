import gym
import numpy as np
from termcolor import cprint


class SimpleVideoRecordingWrapper(gym.Wrapper):
    def __init__(self, 
            env, 
            mode='rgb_array',
            steps_per_render=1,
        ):
        """
        When file_path is None, don't record.
        """
        super().__init__(env)
        
        self.mode = mode
        self.steps_per_render = steps_per_render

        self.step_count = 0

        self.env_render_return_dict = False
        if hasattr(env, 'name'):
            self.env_render_return_dict = True if 'pushT' in env.name else False

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        self.frames = list()

        if self.env_render_return_dict:
            render = self.env.render(mode=self.mode)
            frame = render['images'][0] # (H, W, C)
        else:
            frame = self.env.render(mode=self.mode)
        assert frame.dtype == np.uint8
        self.frames.append(frame)
        
        self.step_count = 1
        return obs
    
    def step(self, action):
        result = super().step(action)
        self.step_count += 1

        if self.env_render_return_dict:
            render = self.env.render(mode=self.mode)
            frame = render['images'][0] # (H, W, C)
        else:
            frame = self.env.render(mode=self.mode)
        assert frame.dtype == np.uint8
        self.frames.append(frame)
        
        return result
    
    def get_video(self):
        video = np.stack(self.frames, axis=0) # (T, H, W, C)
        # to store as mp4 in wandb, we need (T, H, W, C) -> (T, C, H, W)
        video = video.transpose(0, 3, 1, 2)
        return video

