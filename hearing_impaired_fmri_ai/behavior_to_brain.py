import numpy as np
from sklearn.linear_model import Ridge

latents = np.load('audio_latents.npy')
fmri = np.load('child_fmri.npy')

model = Ridge()
model.fit(latents, fmri)
predicted = model.predict(latents)
np.save('predicted_fmri.npy', predicted)