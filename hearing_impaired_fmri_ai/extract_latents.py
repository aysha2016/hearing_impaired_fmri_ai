import torch
import numpy as np
from audio_game_simulation import AudioGameEnv

model = torch.load('./log/audio_game/checkpoint.pt')
env = AudioGameEnv()

latents = []
obs = env.reset()
done = False
while not done:
    latent = model.encoder(obs)
    latents.append(latent.detach().cpu().numpy())
    obs, _, done, _ = env.step(model.policy(latent))

np.save('audio_latents.npy', np.array(latents))