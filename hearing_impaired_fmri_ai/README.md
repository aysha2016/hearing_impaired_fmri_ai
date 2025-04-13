# AI for Hearing-Impaired Cognitive Development

This research simulates auditory learning environments for children with hearing impairments. A DreamerV3 reinforcement learning agent interacts with sound-based tasks. Brain activity (fMRI) from OpenNeuro is used to map cognitive responses to in-game behavior.

## Highlights
- Simulated auditory cognitive games for children
- DreamerV3 learns task-based audio behavior
- Latent representation alignment with fMRI (OpenNeuro dataset)
- Predict and visualize brain region activity

## Usage
```bash
pip install -r requirements.txt

python audio_game_simulation.py       # Simulate environment
python train_agent.py                 # Train DreamerV3 RL agent
python extract_latents.py             # Extract internal states
python load_openneuro_fmri.py         # Load and preprocess fMRI
python brain_mapping.py               # Align latent <-> brain
python behavior_to_brain.py           # Predict brain activity
python visualize_results.py           # Plot and visualize
```