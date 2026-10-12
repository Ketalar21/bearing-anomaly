import os
import numpy as np
from scipy.stats import kurtosis

def compute_features(data_dir, column):
    files = sorted(os.listdir(data_dir))
    rows = []
    for f in files:
        x = np.loadtxt(os.path.join(data_dir, f))[:, column]
        rms = np.sqrt(np.mean(x**2))
        k = kurtosis(x, fisher=False)
        p2p = x.max() - x.min()
        rows.append([rms, k, p2p])
    return files, np.array(rows)