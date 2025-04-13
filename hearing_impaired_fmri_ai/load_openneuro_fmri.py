from nilearn import datasets, input_data
import numpy as np

dataset = datasets.fetch_development_fmri(n_subjects=1)
fmri_filename = dataset.func[0]

masker = input_data.NiftiMasker(smoothing_fwhm=5, standardize=True)
fmri_signals = masker.fit_transform(fmri_filename)

np.save('child_fmri.npy', fmri_signals)