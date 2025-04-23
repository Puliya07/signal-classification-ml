import numpy as np
import matplotlib.pyplot as plt

def generate_signal(signal_type, duration=1.0, sampling_rate=1000, freq=5, amplitude=1.0, noise_level=0.0):
    """Generate different types of signals with optional noise.
    
    Parameters:
    - signal_type: 'sine', 'square', 'sawtooth', 'triangle', 'am', 'fm'
    - duration: signal duration in seconds
    - sampling_rate: number of samples per second
    - freq: base frequency in Hz
    - amplitude: signal amplitude
    - noise_level: standard deviation of Gaussian noise

    Returns:
    - t: time array
    - signal: generated signal with noise
    """

    t = np.linspace(0, duration, int(duration * sampling_rate), endpoint=False)

    if signal_type == 'sine':
        signal = amplitude * np.sin(2 * np.pi * freq * t)
    elif signal_type == 'square':
        signal = amplitude * np.sign(np.sin(2 * np.pi * freq * t))
    elif signal_type == 'sawtooth':
        signal = amplitude * ((2 * np.mod(freq * t, 1))-1)
    elif signal_type == 'triangle':
        signal = amplitude * (2 * np.abs(2 * np.mod(freq * t, 1) - 1) - 1)
    elif signal_type == 'am': 
        # amplitude modulation
        carrier_freq = freq * 10 # carrier frequency
        modulation_index = 0.5
        signal = amplitude * (1 + modulation_index * np.sin(2 * np.pi * freq * t)) * np.sin(2 * np.pi * carrier_freq * t)
    elif signal_type == 'fm':
        # frequency modulation
        carrier_freq = freq * 10
        modulation_index = 5
        signal = amplitude * np.sin(2 * np.pi * carrier_freq * t + modulation_index * np.sin(2 * np.pi * freq * t))
    else:
        raise ValueError("Unknown signal type")
    
    # adding noise if specified
    if noise_level > 0:
        noise = np.random.normal(scale = noise_level, size= len(t))
        signal = signal + noise

    return t, signal

signal_types = ['sine', 'square', 'sawtooth', 'triangle', 'am', 'fm']
plt.figure(figsize = (15,10))

for i, signal_type in enumerate(signal_types):
    t, signal = generate_signal(signal_type, duration=1.0, sampling_rate=1000, freq=5, amplitude=1.0, noise_level = 0.1)

    plt.subplot(len(signal_types), 1, i+1)
    plt.plot(t, signal)
    plt.title(f'{signal_type.upper()} Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)

plt.tight_layout()
plt.show()

    

