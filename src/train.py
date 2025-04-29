import pandas as pd
import numpy as np
import os
import argparse
from create_dataset import create_dataset
from feature_extraction import extract_features
from model import prepare_data, train_random_forest, train_svm, train_neural_network, evaluate_model, save_model

def main(args):
    print("Signal Classification Project")
    print("=============================")

    # step 1: Generate dataset
    if args.regenerate_data or not os.path.exists(args.data_path):
        print("Generating new dataset...")
        X, y = create_dataset(
            n_samples=args.n_samples,
            duration=args.duration,
            sampling_rate=args.sampling_rate,
            save_path=args.data_path
        )
        print(f"Dataset creted with {len(y)} samples")
    else:
        print(f"Loading dataset from {args.data_path}")
        df = pd.read_csv(args.data_path)
        y = df['label'].values
        X = df.drop(['label', 'label_name'], axis=1).values
        print(f"Dataset loaded with {len(y)} samples")

    # step 2: Feature extraction
    print("\nExtracting features...")
    features = extract_features(X, sampling_rate=args.sampling_rate)
    print(f"Extracted {features.shape[1]} features from each signal")

    # step 3: Prepare data
    print("\nPreparing data for training...")
    X_train, X_test, y_train, y_test, scaler = prepare_data(
        features, y, test_size=args.test_size
    )

    # Get class names
    if os.path.exists(args.data_path):
        class_names = pd.read_csv(args.data_path)['label_name'].unique()
    else: 
        class_names = ['sine', 'square', 'sawtooth', 'triangle', 'am', 'fm']

    # step 4: Train and evaluate models
    models = {}

    if 'rf' in args.models:
        print("\nTraining Random Forest...")
        rf_model = train_random_forest(X_train, y_train)
        print("\nEvaluating Random Forest...")
        evaluate_model(rf_model, X_test, y_test, class_names)
        models['random_forest'] = rf_model

    if 'svm' in args.models:
        print("\nTraining SVM...")
        svm_model = train_svm(X_train, y_train)
        print("\nEvaluating SVM...")
        evaluate_model(svm_model, X_test, y_test, class_names)
        models['svm'] = svm_model

    if 'nn' in args.models:
        print("\nTraining Neural Network...")
        nn_model = train_neural_network(X_train, y_train)
        print("\nEvaluating Neural Network...")
        evaluate_model(nn_model, X_test, y_test, class_names)
        models['neural_network'] = nn_model

    # save best model
    best_model_name = args.best_model if args.best_model in models else list(models.keys())[0]
    best_model = models[best_model_name]

    model_path = os.path.join(args.model_dir, f"{best_model_name}_model.pkl")
    scaler_path = os.path.join(args.model_dir, "scaler.pkl")

    save_model(best_model, scaler, model_path, scaler_path)
    print(f"\nBest model ({best_model_name}) saved successfully!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train signal classification models")
    parser.add_argument("--data_path", type=str, default='data/raw/signal_dataset.csv', help="Path to save/load the dataset")
    parser.add_argument("--model_dir", type=str, default='models', help="Directory to save trained models")
    parser.add_argument("--regenerate_data", action="store_true", help="Force regeneration of dataset")
    parser.add_argument("--n_samples", type=int, default=500, help="Number of samples per signal type")
    parser.add_argument("--duration", type=float, default=1.0, help="Duration of each signal in seconds")
    parser.add_argument("--sampling_rate", type=int, default=1000, help="Sampling rate in Hz")
    parser.add_argument("--test_size", type=float, default=0.2, help="Proportion of data to use for testing")
    parser.add_argument("--models", nargs="+", default=["rf", "svm", "nn"], help="Models to train: rf (Random Forest), svm, nn (Neural Network)")
    parser.add_argument("--best_model", type=str, default="random_forest", help="Model to save as the best model")

    args = parser.parse_args()
    main(args)