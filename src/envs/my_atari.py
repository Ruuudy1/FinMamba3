import gymnasium as gym
import numpy as np


class Atari(gym.Env):
    LOCK = None
    metadata = {}

    def __init__(
        self,
        name,
        action_repeat=4,
        size=(64, 64),
        gray=False,
        noops=0,
        lives="discount",
        sticky=False,
        actions="needed",
        length=108000,
        resize="opencv",
        seed=None,
    ):
        assert size[0] == size[1]
        assert lives in ("unused", "discount", "reset"), lives
        assert actions in ("all", "needed"), actions
        assert resize in ("opencv", "pillow"), resize
        if self.LOCK is None:
            import multiprocessing as mp

            mp = mp.get_context("spawn")
            self.LOCK = mp.Lock()
        self._resize = resize
        if self._resize == "opencv":
            import cv2

            self._cv2 = cv2
        if self._resize == "pillow":
            from PIL import Image

            self._image = Image
        # import gym.envs.atari

        self._env_id = name
        self._repeat = action_repeat
        self._size = size
        self._gray = gray
        self._noops = noops
        self._lives = lives
        self._sticky = sticky
        self._length = length
        self._random = np.random.RandomState(seed)
        with self.LOCK:
            self._env = gym.make(
                id=self._env_id,
                obs_type="rgb" if not gray else "grayscale",
                frameskip=1,
                repeat_action_probability=0.25 if sticky else 0.0,
                full_action_space=(actions == "all"),
            )
        assert self._env.unwrapped.get_action_meanings()[0] == "NOOP"
        self._seed = seed
        self._is_init = True
        shape = self._env.observation_space.shape
        self._buffer = [np.zeros(shape, np.uint8) for _ in range(2)]
        self._ale = self._env.unwrapped.ale
        self._last_lives = None
        self._done = True
        self._step = 0
        self.reward_range = [-np.inf, np.inf]

    @property
    def observation_space(self):
        img_shape = self._size + ((1,) if self._gray else (3,))
        return gym.spaces.Dict(
            {
                "image": gym.spaces.Box(0, 255, img_shape, np.uint8),
            }
        )

    @property
    def action_space(self):
        space = self._env.action_space
        space.discrete = True
        return space

    def step(self, action):
        # if action['reset'] or self._done:
        #   with self.LOCK:
        #     self._reset()
        #   self._done = False
        #   self._step = 0
        #   return self._obs(0.0, is_first=True)
        total = 0.0
        dead = False
        # if len(action.shape) >= 1:
        #     action = np.argmax(action)
        for repeat in range(self._repeat):
            _, reward, terminated, truncated, info = self._env.step(action)
            self._step += 1
            total += reward
            if repeat == self._repeat - 2:
                self._screen(self._buffer[1])
            if terminated or truncated:
                break
            if self._lives != "unused":
                current = self._ale.lives()
                if current < self._last_lives:
                    dead = True
                    self._last_lives = current
                    break
        if not self._repeat:
            self._buffer[1][:] = self._buffer[0][:]
        self._screen(self._buffer[0])
        my_truncated = self._length and self._step >= self._length
        self._done = terminated or my_truncated or truncated
        return self._obs(
            total,
            info,
            is_last=self._done or (dead and self._lives == "reset"), # use for record the episode reward
            is_terminal=dead or terminated, # for the critic to estimate the value of a state
        )
    def reset(self):
        # if self._is_init:
        #     _, info = self._env.reset(seed=self._seed)
        #     self._is_init = False
        # else:
        #     _, info = self._env.reset()
        _, info = self._env.reset()
        if self._noops:
            for _ in range(self._random.randint(self._noops)):
                _, _, dead, _ = self._env.step(0)
                if dead:
                    self._env.reset()
        self._last_lives = self._ale.lives()
        self._screen(self._buffer[0])
        self._buffer[1].fill(0)

        self._done = False
        self._step = 0
        obs, reward, is_last, info = self._obs(0.0, info, is_first=True)
        return obs, info

    def _obs(self, reward, info, is_first=False, is_last=False, is_terminal=False):
        np.maximum(self._buffer[0], self._buffer[1], out=self._buffer[0]) # https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/
        image = self._buffer[0]
        if image.shape[:2] != self._size:
            if self._resize == "opencv":
                image = self._cv2.resize(
                    image, self._size, interpolation=self._cv2.INTER_AREA
                )
            if self._resize == "pillow":
                image = self._image.fromarray(image)
                image = image.resize(self._size, self._image.NEAREST)
                image = np.array(image)
        if self._gray:
            weights = [0.299, 0.587, 1 - (0.299 + 0.587)]
            image = np.tensordot(image, weights, (-1, 0)).astype(image.dtype)
            image = image[:, :, None]
        info['is_first'] = is_first
        info['is_terminal'] = is_terminal
        info['episode_frame_number'] //= self._repeat
        return (
            # {"image": image, "is_terminal": is_terminal, "is_first": is_first},
            image,
            reward,
            is_last,
            info,
        )

    def _screen(self, array):
        self._ale.getScreenRGB(array)

    def close(self):
        return self._env.close()
    
if __name__ == '__main__':
    env = Atari(
        'ALE/Pong-v5',
        4,
        (64, 64),
        gray=False,
        noops=0,
        lives="reset",
        sticky=False,
        actions="needed",
        resize="opencv",
    )
    obs, info = env.reset()
    import time
    
    # Start the timer
    start_time = time.time()

    for i in range(10000):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        if done:
            obs, info = env.reset()
    
    # End the timer
    end_time = time.time()
    env.close()
    # Print the elapsed time
    print(f"Time taken for 10,000 iterations: {end_time - start_time:.2f} seconds")

        # Initialize the Gymnasium Atari environment with atari-preprocessing
    gym_env = gym.make("ALE/Pong-v5", render_mode='rgb_array', frameskip=1)
    gym_env = gym.wrappers.AtariPreprocessing(gym_env)

    # Reset the environment
    obs, info = gym_env.reset()

    # Start the timer for the Gymnasium environment
    start_time_gym = time.time()

    # Run 10,000 iterations with random actions
    for i in range(10000):
        action = gym_env.action_space.sample()
        obs, reward, terminated, truncated, info = gym_env.step(action)
        if terminated or truncated:
            obs, info = gym_env.reset()

    # End the timer
    end_time_gym = time.time()
    gym_env.close()

    # Print the elapsed time
    print(f"Time taken for 10,000 iterations: {end_time_gym - start_time_gym:.2f} seconds")