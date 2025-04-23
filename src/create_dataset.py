import numpy as np
import pandas as pd
import os
from data_generation import generate_signal

def create_dataset(n_samples = 1000, duration = 1.0, sampling_rate = 1000,
                   base_freq_range = (1,10), amplitude_range = (0.5, 2.0), noise_range = (0.0, 0.3), save_path = None):
    """ 
    Create a dataset of signals with labels for classification.
    
    Parameters:
    - n_samples: number of samples per signal type
    - Other parameters are passed to generate_signal function
    
    Returns:
    - X: signal data (n_samples * n_signal_types, duration * sampling_rate)
    - y: labels(n_samples * n_signal_types,)
    """

    signal_types = ['sine', 'square', 'sawtooth', 'triangle', 'am', 'fm']
    n_features = int(duration * sampling_rate)

    X = np.zeros((n_samples * len(signal_types), n_features))
    y = np.zeros(n_samples *len(signal_types), dtype=int)

    for i, signal_type in enumerate(signal_types):
        for j in range(n_samples):
            # randomly select parameters
            freq = np.random.uniform(*base_freq_range) 
            amplitude = np.random.uniform(*amplitude_range)
            noise_level = np.random.uniform(*noise_range)

            # generate signal
            _, signal = generate_signal(
                signal_type=signal_type,
                duration=duration,
                sampling_rate=sampling_rate,
                freq=freq,
                amplitude=amplitude,
                noise_level=noise_level
                )
            
            # store signal and label
            idx = i * n_samples + j
            X[idx] = signal
            y[idx] = i

    if save_path:
        # create dataframe with features and label
        df = pd.DataFrame(X)
        df['label'] = y
        df['label_name'] = df['label'].map(dict(enumerate(signal_types)))

        # save to csv
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)

        print(f"Saving dataset to: {save_path}")

    return X, y

print("Current working directory:", os.getcwd())

if __name__ == '__main__':
    X , y = create_dataset(n_samples=500, save_path="data/raw/signal_dataset.csv")
    print(f"Dataset created with shape: {X.shape}, {y.shape}")