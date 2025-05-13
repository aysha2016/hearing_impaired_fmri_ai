# Hearing Impaired fMRI-RL Research Project

## Overview
This research project explores auditory learning environments for children with hearing impairments by integrating reinforcement learning (RL), neuroimaging, and sound-based tasks. The project uses a DreamerV3 RL agent to interact with simulated auditory tasks while mapping cognitive responses using real brain activity data (fMRI) from OpenNeuro.

## Key Components

### 🎮 DreamerV3 Agent
- Implements adaptive learning behaviors in response to audio stimuli
- Uses world model-based reinforcement learning
- Trained on custom auditory learning tasks

### 🧠 fMRI Integration
- Processes and analyzes brain activity data from OpenNeuro
- Maps cognitive responses to agent behavior
- Focuses on auditory cortex activation patterns

### 🎧 Audio Environment
- Simulates auditory learning challenges
- Implements pitch discrimination tasks
- Provides customizable difficulty levels

### 🧪 Applications
- Neuroadaptive interfaces
- Auditory training systems
- Inclusive AI design for hearing-impaired individuals

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- Git

### Setup
1. Clone the repository:
```bash
git clone https://github.com/yourusername/hearing_impaired_fmri_ai.git
cd hearing_impaired_fmri_ai
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -e .
```

## Usage

### 1. Data Preparation
```bash
python src/fmri/data_loader.py --dataset development_fmri --n_subjects 1
```

### 2. Environment Testing
```bash
python -m src.environment.audio_game --task pitch_discrimination --difficulty 0.5
```

### 3. Agent Training
```bash
python -m src.agent.training --config configs/dreamerv3_default.yaml
```

### 4. Behavior-Brain Mapping
```bash
python -m src.mapping.behavior_brain --model_path results/models/agent.pt
```

### Jupyter Notebooks
For interactive exploration and development:
1. `notebooks/01_data_exploration.ipynb`: Explore fMRI data
2. `notebooks/02_environment_testing.ipynb`: Test audio environment
3. `notebooks/03_agent_training.ipynb`: Train and evaluate agent
4. `notebooks/04_behavior_brain_mapping.ipynb`: Map behavior to brain activity

## Project Structure