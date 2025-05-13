# Hearing Impaired fMRI-RL Research Project
# 
# This notebook implements a research framework combining:
# - DreamerV3 Reinforcement Learning
# - fMRI Data Processing
# - Auditory Learning Environment
# - Behavior-Brain Mapping
# 
# ## Setup and Installation

# %%
# Mount Google Drive (optional)
from google.colab import drive
drive.mount('/content/drive')

# Create project directory
!mkdir -p /content/hearing_impaired_fmri_ai
%cd /content/hearing_impaired_fmri_ai

# %%
# Install required packages
!pip install torch gymnasium dreamerv3 nilearn nibabel librosa soundfile pydub pandas matplotlib seaborn tqdm scikit-learn

# %%
# Import necessary libraries
import os
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import soundfile as sf
import librosa
from pydub import AudioSegment
from nilearn import datasets, input_data
import nibabel as nib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm
import logging
from datetime import datetime
import json
from IPython.display import Audio, display

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Check GPU availability
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# %% [markdown]
# ## 1. Audio Game Environment

# %%
class AudioGameEnv(gym.Env):
    """Custom environment for auditory learning tasks."""
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 task_type: str = "pitch_discrimination",
                 difficulty: float = 0.5):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.task_type = task_type
        self.difficulty = np.clip(difficulty, 0.0, 1.0)
        
        # Observation space: 1 second of audio
        self.observation_space = gym.spaces.Box(
            low=-1, 
            high=1, 
            shape=(sample_rate,), 
            dtype=np.float32
        )
        
        # Action space: 0=lower, 1=same, 2=higher
        self.action_space = gym.spaces.Discrete(3)
        
        # Task parameters
        self.current_step = 0
        self.max_steps = 20
        self.current_pitch = 440.0  # A4 note
        self.target_pitch = None
        self.reference_audio = None
        
        self._initialize_sounds()
    
    def _initialize_sounds(self):
        """Initialize reference sounds."""
        if self.task_type == "pitch_discrimination":
            duration = 0.5
            t = np.linspace(0, duration, int(self.sample_rate * duration))
            self.reference_audio = np.sin(2 * np.pi * self.current_pitch * t)
    
    def _generate_stimulus(self) -> np.ndarray:
        """Generate audio stimulus."""
        if self.task_type == "pitch_discrimination":
            pitch_variation = (np.random.rand() - 0.5) * 100 * self.difficulty
            current_pitch = self.current_pitch + pitch_variation
            duration = 0.5
            t = np.linspace(0, duration, int(self.sample_rate * duration))
            audio = np.sin(2 * np.pi * current_pitch * t)
            return audio
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")
    
    def reset(self, seed: int = None) -> tuple:
        """Reset environment."""
        super().reset(seed=seed)
        self.current_step = 0
        self.target_pitch = self.current_pitch
        self.current_pitch = 440.0 + (np.random.rand() - 0.5) * 100
        return self._get_obs(), {}
    
    def step(self, action: int) -> tuple:
        """Execute one step."""
        self.current_step += 1
        observation = self._generate_stimulus()
        reward = self._calculate_reward(action)
        done = self.current_step >= self.max_steps
        truncated = False
        info = {
            'current_pitch': self.current_pitch,
            'target_pitch': self.target_pitch,
            'step': self.current_step
        }
        return observation, reward, done, truncated, info
    
    def _calculate_reward(self, action: int) -> float:
        """Calculate reward."""
        if self.task_type == "pitch_discrimination":
            current_pitch = self.current_pitch
            if action == 0 and current_pitch < self.target_pitch:
                return 1.0
            elif action == 1 and abs(current_pitch - self.target_pitch) < 10:
                return 1.0
            elif action == 2 and current_pitch > self.target_pitch:
                return 1.0
            return -0.5
        return 0.0
    
    def _get_obs(self) -> np.ndarray:
        return self._generate_stimulus()
    
    def play_audio(self, audio: np.ndarray):
        """Play audio in notebook."""
        display(Audio(audio, rate=self.sample_rate))

# Test the environment
env = AudioGameEnv()
obs, _ = env.reset()
print(f"Observation shape: {obs.shape}")
print(f"Action space: {env.action_space}")

# Play a sample audio
print("\nPlaying sample audio:")
env.play_audio(obs)

# %% [markdown]
# ## 2. DreamerV3 Agent Training

# %%
from dreamerv3 import embodied
from dreamerv3 import agent

def create_agent_config():
    """Create DreamerV3 configuration."""
    config = embodied.Config(dreamerv3.configs['defaults'])
    
    # Custom configuration for Colab
    config = config.update({
        'logdir': './logs/audio_game',
        'run.train_ratio': 32,  # Reduced for Colab
        'run.eval_ratio': 5,
        'run.steps': 1e5,  # Reduced for demonstration
        'batch_size': 32,
        'rssm.hidden': 200,
        'rssm.deter': 200,
        'rssm.stoch': 30,
        'rssm.discrete': 32,
    })
    
    return embodied.Flags(config).parse()

def train_agent(env, config, num_episodes=100):
    """Train DreamerV3 agent."""
    # Initialize agent
    agnt = agent.Agent(env.observation_space, env.action_space, config)
    
    # Training metrics
    rewards = []
    
    # Training loop
    for episode in tqdm(range(num_episodes)):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            # Get action
            action = agnt.policy(obs)
            
            # Step environment
            next_obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            
            # Store transition
            agnt.observe(obs, action, reward, done)
            obs = next_obs
            
            # Training step
            if episode % config.run.train_ratio == 0:
                agnt.train()
        
        rewards.append(episode_reward)
        
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1}, Reward: {episode_reward:.2f}")
    
    return agnt, rewards

# Train agent
config = create_agent_config()
trained_agent, training_rewards = train_agent(env, config)

# Plot training progress
plt.figure(figsize=(10, 5))
plt.plot(training_rewards)
plt.title('Training Progress')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.show()

# Save trained agent
torch.save(trained_agent.state_dict(), 'trained_agent.pt')

# %% [markdown]
# ## 3. fMRI Data Processing

# %%
class FMRIProcessor:
    """Process fMRI data for integration with RL agent."""
    
    def __init__(self, smoothing_fwhm: float = 5.0):
        self.smoothing_fwhm = smoothing_fwhm
        self.masker = None
        self.scaler = StandardScaler()
    
    def fetch_data(self, n_subjects: int = 1):
        """Fetch fMRI data from OpenNeuro."""
        print("Fetching fMRI data...")
        dataset = datasets.fetch_development_fmri(n_subjects=n_subjects)
        return dataset
    
    def preprocess_data(self, fmri_filename: str) -> np.ndarray:
        """Preprocess fMRI data."""
        # Initialize masker
        self.masker = input_data.NiftiMasker(
            smoothing_fwhm=self.smoothing_fwhm,
            standardize=True,
            detrend=True,
            high_pass=0.01,
            low_pass=0.1,
            t_r=2.0
        )
        
        # Process data
        fmri_signals = self.masker.fit_transform(fmri_filename)
        return self.scaler.fit_transform(fmri_signals)
    
    def extract_roi_signals(self, fmri_signals: np.ndarray) -> np.ndarray:
        """Extract signals from regions of interest."""
        # Use default auditory cortex mask
        roi_mask = datasets.fetch_neurovault_motor_task()['images'][0]
        
        roi_masker = input_data.NiftiMasker(
            mask_img=roi_mask,
            smoothing_fwhm=self.smoothing_fwhm,
            standardize=True
        )
        
        return roi_masker.fit_transform(fmri_signals)

# Process fMRI data
processor = FMRIProcessor()
dataset = processor.fetch_data(n_subjects=1)

# Process first subject's data
fmri_signals = processor.preprocess_data(dataset.func[0])
roi_signals = processor.extract_roi_signals(fmri_signals)

print(f"Processed fMRI signals shape: {fmri_signals.shape}")
print(f"ROI signals shape: {roi_signals.shape}")

# Visualize some signals
plt.figure(figsize=(15, 5))
plt.plot(fmri_signals[:100, :5])
plt.title('First 100 timepoints of 5 voxels')
plt.xlabel('Time')
plt.ylabel('Signal')
plt.show()

# Save processed data
np.save('processed_fmri.npy', fmri_signals)
np.save('roi_signals.npy', roi_signals)

# %% [markdown]
# ## 4. Behavior-Brain Mapping

# %%
class BehaviorBrainMapper(nn.Module):
    """Neural network to map agent behavior to brain activity."""
    
    def __init__(self, behavior_dim: int, brain_dim: int):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(behavior_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, brain_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

def collect_agent_behavior(agent, env, num_episodes: int = 50) -> np.ndarray:
    """Collect behavior data from trained agent."""
    behaviors = []
    
    for _ in tqdm(range(num_episodes)):
        obs, _ = env.reset()
        done = False
        episode_behavior = []
        
        while not done:
            action = agent.policy(obs)
            next_obs, reward, done, truncated, info = env.step(action)
            
            # Store behavior features
            behavior = np.concatenate([
                obs,  # Current observation
                [action],  # Action taken
                [reward],  # Reward received
                [info['current_pitch']],  # Current pitch
                [info['target_pitch']]  # Target pitch
            ])
            
            episode_behavior.append(behavior)
            obs = next_obs
        
        behaviors.extend(episode_behavior)
    
    return np.array(behaviors)

def train_behavior_brain_mapping(behavior_data: np.ndarray,
                               brain_data: np.ndarray,
                               epochs: int = 50,
                               batch_size: int = 32) -> Tuple[nn.Module, list]:
    """Train mapping between behavior and brain activity."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Prepare data
    X_train, X_test, y_train, y_test = train_test_split(
        behavior_data, brain_data, test_size=0.2, random_state=42
    )
    
    # Convert to tensors
    X_train = torch.FloatTensor(X_train).to(device)
    y_train = torch.FloatTensor(y_train).to(device)
    X_test = torch.FloatTensor(X_test).to(device)
    y_test = torch.FloatTensor(y_test).to(device)
    
    # Initialize model
    model = BehaviorBrainMapper(
        behavior_dim=behavior_data.shape[1],
        brain_dim=brain_data.shape[1]
    ).to(device)
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    train_losses = []
    test_losses = []
    
    for epoch in tqdm(range(epochs)):
        model.train()
        total_loss = 0
        
        # Mini-batch training
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Validation
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, y_test)
            
            train_losses.append(total_loss / len(X_train))
            test_losses.append(test_loss.item())
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, "
                  f"Train Loss: {train_losses[-1]:.4f}, "
                  f"Test Loss: {test_losses[-1]:.4f}")
    
    return model, (train_losses, test_losses)

# Collect agent behavior
behavior_data = collect_agent_behavior(trained_agent, env)

# Load processed fMRI data
brain_data = np.load('roi_signals.npy')

# Train mapping model
mapping_model, (train_losses, test_losses) = train_behavior_brain_mapping(
    behavior_data, brain_data
)

# Plot training progress
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.title('Behavior-Brain Mapping Training Progress')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Save mapping model
torch.save(mapping_model.state_dict(), 'behavior_brain_mapping.pt')

# %% [markdown]
# ## 5. Save Results to Google Drive (Optional)

# %%
# Create timestamp for unique folder name
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = f"/content/drive/MyDrive/hearing_impaired_fmri_ai/results_{timestamp}"
os.makedirs(results_dir, exist_ok=True)

# Save all results
!cp trained_agent.pt {results_dir}/
!cp behavior_brain_mapping.pt {results_dir}/
!cp processed_fmri.npy {results_dir}/
!cp roi_signals.npy {results_dir}/

# Save training plots
plt.figure(figsize=(10, 5))
plt.plot(training_rewards)
plt.title('Agent Training Progress')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.savefig(f'{results_dir}/agent_training.png')

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.title('Behavior-Brain Mapping Training Progress')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f'{results_dir}/mapping_training.png')

print(f"Results saved to: {results_dir}")