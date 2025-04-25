import numpy as np
from scipy import signal
from scipy.fft import fft

def extract_time_domain_features(signals):
    """Extract time domain features from signals."""
    features = []

    for s in signals:
        # stat features
        mean = np.mean(s)  
        std = np.std(s)      
        rms = np.sqrt(np.mean(s**2))    
        peak = np.max(np.abs(s))        
        crest_factor = peak / rms if rms > 0 else 0     

        # zero crossings
        zero_crossings = np.sum(np.diff(np.signbit(s)))

        # energy
        energy = np.sum(s**2)

        features.append([mean, std, rms, peak, crest_factor, zero_crossings, energy])
    
    return np.array(features)

def extract_frequency_domain_features(signals, sampling_rate = 1000):
    """Extract frequency domain features from signals."""
    features = []

    for s in signals:
        # computing fft
        spectrum = np.abs(fft(s))
        spectrum = spectrum[:len(s)//2] # taking only the positive frequencies
        freqs = np.fft.fftfreq(len(s), 1/sampling_rate)[:len(s)//2]

        # dominant frequency
        if len(spectrum) > 0:
            dominant_freq = freqs[np.argamx(spectrum)]
        else:
            dominant_freq = 0

        # spectral centroid
        if np.sum(spectrum) > 0:
            spectral_centroid = np.sum(freqs * spectrum)/np.sum(spectrum)
        else:
            spectral_centroid = 0

        # spectral bandwidth
        if np.sum(spectrum) > 0:
            spectral_bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * spectrum)  / np.sum(spectrum))
        else:
            spectral_bandwidth = 0

        # spectral energy
        spectral_energy = np.sum(spectrum ** 2)

        features.append([dominant_freq, spectral_centroid, spectral_bandwidth, spectral_energy])

    return np.array(features)

def extract_features(signals, sampling_rate = 1000):
    """ Extract both time and frequency domain features."""
    time_features = extract_time_domain_features(signals)
    freq_features = extract_frequency_domain_features(signals, sampling_rate)

    # combine features
    return np.hstack((time_features, freq_features))

