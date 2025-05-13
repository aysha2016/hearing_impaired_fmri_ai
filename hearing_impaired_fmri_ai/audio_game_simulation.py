import gymnasium as gym
import numpy as np
import soundfile as sf
import librosa
import os
from typing import Tuple, Dict, Any
from pydub import AudioSegment

class AudioGameEnv(gym.Env):
    """
    Custom environment for auditory learning tasks.
    Simulates various auditory challenges for children with hearing impairments.
    """
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 task_type: str = "pitch_discrimination",
                 difficulty: float = 0.5):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.task_type = task_type
        self.difficulty = np.clip(difficulty, 0.0, 1.0)
        
        # Observation space: 1 second of audio at sample_rate
        self.observation_space = gym.spaces.Box(
            low=-1, 
            high=1, 
            shape=(sample_rate,), 
            dtype=np.float32
        )
        
        # Action space: 0=lower, 1=same, 2=higher for pitch discrimination
        self.action_space = gym.spaces.Discrete(3)
        
        # Task parameters
        self.current_step = 0
        self.max_steps = 20
        self.current_pitch = 440.0  # A4 note
        self.target_pitch = None
        self.reference_audio = None
        
        # Load or generate reference sounds
        self._initialize_sounds()
    
    def _initialize_sounds(self):
        """Initialize or load reference sounds for the task."""
        if self.task_type == "pitch_discrimination":
            # Generate reference tones
            duration = 0.5  # seconds
            t = np.linspace(0, duration, int(self.sample_rate * duration))
            self.reference_audio = np.sin(2 * np.pi * self.current_pitch * t)
            
    def _generate_stimulus(self) -> np.ndarray:
        """Generate the current audio stimulus based on task type."""
        if self.task_type == "pitch_discrimination":
            # Generate a tone with some pitch variation
            pitch_variation = (np.random.rand() - 0.5) * 100 * self.difficulty
            current_pitch = self.current_pitch + pitch_variation
            duration = 0.5
            t = np.linspace(0, duration, int(self.sample_rate * duration))
            audio = np.sin(2 * np.pi * current_pitch * t)
            return audio
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")
    
    def _calculate_reward(self, action: int) -> float:
        """Calculate reward based on the action and current state."""
        if self.task_type == "pitch_discrimination":
            # For pitch discrimination, reward correct identification
            current_pitch = self.current_pitch
            if action == 0 and current_pitch < self.target_pitch:
                return 1.0
            elif action == 1 and abs(current_pitch - self.target_pitch) < 10:
                return 1.0
            elif action == 2 and current_pitch > self.target_pitch:
                return 1.0
            return -0.5
        return 0.0
    
    def reset(self, seed: int = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        self.current_step = 0
        self.target_pitch = self.current_pitch
        self.current_pitch = 440.0 + (np.random.rand() - 0.5) * 100
        return self._get_obs(), {}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment."""
        self.current_step += 1
        
        # Generate new stimulus
        observation = self._generate_stimulus()
        
        # Calculate reward
        reward = self._calculate_reward(action)
        
        # Check if episode is done
        done = self.current_step >= self.max_steps
        truncated = False
        
        # Additional info
        info = {
            'current_pitch': self.current_pitch,
            'target_pitch': self.target_pitch,
            'step': self.current_step
        }
        
        return observation, reward, done, truncated, info
    
    def _get_obs(self) -> np.ndarray:
        """Get current observation."""
        return self._generate_stimulus()
    
    def render(self):
        """Render the current state (optional)."""
        pass

    def close(self):
        """Clean up resources."""
        pass

env = AudioGameEnv()
obs = env.reset()
for _ in range(5):
    action = env.action_space.sample()
    obs, reward, done, _ = env.step(action)
    print(f"Step: {_}, Reward: {reward}")