import numpy as np
import joblib
import os 
import argparse
from data_generation import generate_signal
from feature_extraction import extract_features

def load_model(model_path, scaler_path):
    """Load trained model and scaler."""
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

def predict_signal_type(signal_data, model, scaler, class_names):
    """Predict signal type from raw signal data."""
    #Extract features
    features = extract_features(signal_data.reshape(1, -1))

    # Scale features
    scaled_features = scaler.transform(features)
    
    # Make prediction
    prediction = model.predict(scaled_features)[0]
    probabilities = model.predict_proba(scaled_features)[0]
    
    return {
        'predicted_class': class_names[prediction],
        'predicted_class_id': int(prediction),
        'probabilities': {class_names[i]: float(prob) for i, prob in enumerate(probabilities)}
    }

def main(args):
    # Load model and scaler
    model, scaler = load_model(args.model_path, args.scaler_path)
    
    # Define class names
    class_names = ['sine', 'square', 'sawtooth', 'triangle', 'am', 'fm']
    
    # Generate a test signal
    _, signal = generate_signal(
        signal_type=args.signal_type,
        duration=args.duration,
        sampling_rate=args.sampling_rate,
        freq=args.frequency,
        amplitude=args.amplitude,
        noise_level=args.noise_level
    )
    
    # Make prediction
    result = predict_signal_type(signal, model, scaler, class_names)
    
    # Print results
    print("\nPrediction Results:")
    print(f"True signal type: {args.signal_type}")
    print(f"Predicted signal type: {result['predicted_class']}")
    print("\nClass probabilities:")
    for class_name, prob in result['probabilities'].items():
        print(f"  {class_name}: {prob:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict signal type")
    parser.add_argument("--model_path", type=str, default="models/random_forest_model.pkl",
                        help="Path to trained model")
    parser.add_argument("--scaler_path", type=str, default="models/scaler.pkl",
                        help="Path to feature scaler")
    parser.add_argument("--signal_type", type=str, default="sine",
                        choices=['sine', 'square', 'sawtooth', 'triangle', 'am', 'fm'],
                        help="Type of signal to generate for testing")
    parser.add_argument("--duration", type=float, default=1.0,
                        help="Duration of signal in seconds")
    parser.add_argument("--sampling_rate", type=int, default=1000,
                        help="Sampling rate in Hz")
    parser.add_argument("--frequency", type=float, default=5.0,
                        help="Signal frequency in Hz")
    parser.add_argument("--amplitude", type=float, default=1.0,
                        help="Signal amplitude")
    parser.add_argument("--noise_level", type=float, default=0.1,
                        help="Noise level (standard deviation)")
    
    args = parser.parse_args()
    main(args)
