# Signal Classification Project

This project implements an ML system for identifying different types of signals based on their waveform characteristics.

## Project Structure

```bash
signal-classification-ml/
├── data/ 
│ ├── raw/ #Raw generated signal data
│ └── processed/ #Processed features
├── models/ # Saved trained models
├── notebooks/ # Jupyter notebooks for exploration
├── src/ # Source code
│ ├── data_generation.py # Signal generation
│ ├── create_dataset.py # Dataset creation
│ ├── feature_extraction.py # Feature extraction
│ ├── model.py # Model training and evaluation
│ ├── train.py # Main training script
│ └── predict.py # Prediction script
├── tests/ # Unit tests
├── .gitignore
├── README.md
└── requirements.txt
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Puliya07/signal-classification-ml.git
cd signal-classification-ml
```

2. Create a virtual environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Generate Dataset

```bash 
cd src
python create_dataset.py
```

### Train Models
```bash
cd src
python train.py --models rf svm nn --best_model random_forest
```

### Make Predictions
```bash
cd src
python predict.py --signal_type sine --frequency 7 --noise_level 0.2
```

## Signal Types

The project can generate and classify the following signal types:

1. **Sine Wave**: Basic sinusoidal signal
2. **Square Wave**: Signal that alternates between two fixed values
3. **Sawtooth Wave**: Signal that ramps up linearly and drops sharply
4. **Triangle Wave**: Signal that ramps up and down linearly
5. **AM (Amplitude Modulation)**: Carrier signal with amplitude varied by a modulating signal
6. **FM (Frequency Modulation)**: Carrier signal with frequency varied by a modulating signal

## License

MIT






