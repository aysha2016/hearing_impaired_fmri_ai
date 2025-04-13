import numpy as np
from sklearn.cross_decomposition import CCA

latents = np.load('audio_latents.npy')
fmri = np.load('child_fmri.npy')

min_len = min(len(latents), len(fmri))
latents, fmri = latents[:min_len], fmri[:min_len]

cca = CCA(n_components=10)
X_c, Y_c = cca.fit_transform(latents, fmri)

np.save('cca_latent.npy', X_c)
np.save('cca_fmri.npy', Y_c)
print("CCA mapping completed.")