import numpy as np
import matplotlib.pyplot as plt

actual = np.load('child_fmri.npy')
predicted = np.load('predicted_fmri.npy')

plt.plot(actual[0], label='Actual')
plt.plot(predicted[0], label='Predicted')
plt.title('fMRI Prediction - Hearing Impaired Child')
plt.legend()
plt.show()