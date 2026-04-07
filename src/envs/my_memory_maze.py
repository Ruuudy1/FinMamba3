import gym
import numpy as np

###from tf dreamerv2 code


class MemoryMaze:
    def __init__(self, name, action_repeat=2, obs_key="image", act_key="action", size=(64, 64), seed=0):
        # 9x9, 11x11, 13x13 and 15x15 are available
        self._env_id = name
        self._env = gym.make(self._env_id)
        self._seed = seed
        self._obs_is_dict = hasattr(self._env.observation_space, "spaces")
        self._obs_key = obs_key
        self._act_key = act_key
        self._size = size
        self._gray = False
        self._is_init = True
        self._step = 0
        self._repeat = action_repeat


    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)

    @property
    def observation_space(self):
        if self._obs_is_dict:
            spaces = self._env.observation_space.spaces.copy()
        else:
            spaces = {self._obs_key: self._env.observation_space}
        return gym.spaces.Dict(
            {
                **spaces
            }
        )

    @property
    def action_space(self):
        space = self._env.action_space
        space.discrete = True
        return space

    def step(self, action):
        total = 0.0
        for repeat in range(self._repeat):
            obs, reward, done, info = self._env.step(action)
            self._step += 1
            total += reward
            if done:
                break
        info["is_first"] = False
        info["is_last"] = done
        info["is_terminal"] = info.get("is_terminal", False)
        info["episode_frame_number"] = self._step // self._repeat
        return obs, reward, done, info

    def reset(self):
        obs = self._env.reset()
        self._step = 0
        info = {}
        info["is_first"] = True
        info["is_last"] = False
        info["is_terminal"] = False
        return obs, info
    
if __name__ == '__main__':
    env = MemoryMaze(
        'memory_maze:MemoryMaze-9x9-v0'
    )
    obs, info = env.reset()
    import time
    
    # Start the timer
    start_time = time.time()

    for i in range(10000):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        if done:
            obs = env.reset()
    
    # End the timer
    end_time = time.time()
    env.close()
    # Print the elapsed time
    print(f"Time taken for 10,000 iterations: {end_time - start_time:.2f} seconds")