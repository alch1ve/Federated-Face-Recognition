import os
import numpy as np
from sklearn.model_selection import train_test_split

def load_dataset_from_npz(npz_path, test_size=0.2):
    data = np.load(npz_path)
    x = data['arr_0']
    y = data['arr_1']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)
    return x_train, x_test, y_train, y_test

# Edit this part base on how you load your dataset
