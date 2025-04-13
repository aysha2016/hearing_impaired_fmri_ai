import gym
import numpy as np
import soundfile as sf

class AudioGameEnv(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(16000,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(3)
        self.current_step = 0

    def reset(self):
        self.current_step = 0
        return self._get_obs()

    def step(self, action):
        self.current_step += 1
        reward = 1 if action == (self.current_step % 3) else -1
        done = self.current_step > 10
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        audio = np.random.randn(16000).astype(np.float32) * 0.01
        return audio

env = AudioGameEnv()
obs = env.reset()
for _ in range(5):
    action = env.action_space.sample()
    obs, reward, done, _ = env.step(action)
    print(f"Step: {_}, Reward: {reward}")