import os
import torch
import numpy as np
from dreamerv3 import embodied
from dreamerv3 import agent
from audio_game_simulation import AudioGameEnv
from tqdm import tqdm
import logging
import json
from datetime import datetime

def setup_logging(log_dir: str):
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def create_agent_config():
    """Create and configure the DreamerV3 agent."""
    config = embodied.Config(dreamerv3.configs['defaults'])
    
    # Custom configuration
    config = config.update({
        'logdir': './logs/audio_game',
        'run.train_ratio': 64,
        'run.eval_ratio': 10,
        'run.steps': 1e6,
        'batch_size': 50,
        'rssm.hidden': 200,
        'rssm.deter': 200,
        'rssm.stoch': 30,
        'rssm.discrete': 32,
        'rssm.act': 'elu',
        'rssm.norm': 'none',
        'rssm.std_act': 'softplus',
        'rssm.min_std': 0.1,
    })
    
    return embodied.Flags(config).parse()

def train_agent(env: AudioGameEnv, config: embodied.Config, logger: logging.Logger):
    """Train the DreamerV3 agent on the audio environment."""
    # Initialize agent
    agnt = agent.Agent(env.observation_space, env.action_space, config)
    
    # Training loop
    total_steps = int(config.run.steps)
    eval_interval = int(config.run.eval_ratio)
    train_interval = int(config.run.train_ratio)
    
    logger.info(f"Starting training for {total_steps} steps")
    
    for step in tqdm(range(total_steps)):
        # Collect experience
        obs = env.reset()[0]
        done = False
        episode_reward = 0
        
        while not done:
            # Get action from agent
            action = agnt.policy(obs)
            
            # Step environment
            next_obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            
            # Store transition
            agnt.observe(obs, action, reward, done)
            obs = next_obs
            
            # Training step
            if step % train_interval == 0:
                agnt.train()
        
        # Evaluation
        if step % eval_interval == 0:
            eval_reward = evaluate_agent(env, agnt, logger)
            logger.info(f"Step {step}: Eval reward = {eval_reward:.2f}")
            
        # Logging
        if step % 1000 == 0:
            logger.info(f"Step {step}: Episode reward = {episode_reward:.2f}")
            
    # Save final model
    save_path = os.path.join(config.logdir, 'final_model.pt')
    torch.save(agnt.state_dict(), save_path)
    logger.info(f"Model saved to {save_path}")

def evaluate_agent(env: AudioGameEnv, agent: agent.Agent, logger: logging.Logger, 
                  num_episodes: int = 5) -> float:
    """Evaluate the agent's performance."""
    total_reward = 0
    
    for _ in range(num_episodes):
        obs = env.reset()[0]
        done = False
        episode_reward = 0
        
        while not done:
            action = agent.policy(obs, eval_mode=True)
            obs, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            
        total_reward += episode_reward
    
    return total_reward / num_episodes

def main():
    # Setup
    config = create_agent_config()
    logger = setup_logging(config.logdir)
    
    # Create environment
    env = AudioGameEnv(
        task_type="pitch_discrimination",
        difficulty=0.5
    )
    
    # Save configuration
    config_path = os.path.join(config.logdir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Train agent
    try:
        train_agent(env, config, logger)
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise
    finally:
        env.close()

if __name__ == "__main__":
    main()