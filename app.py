# Import modules from the src directory
from src.classification import train_classification_models
from src.clustering import train_clustering_models
from src.data_loader import load_data, preprocess_data
from src.visualization import visualize_data


def main():
    data_file = "C:/Users/This PC/Desktop/Machine Learning/ML-Repo/diabetes.csv"

    # Load and preprocess data
    df = load_data(data_file)

    X_scaled, y = preprocess_data(df)

    # Visualize data
    # visualize_data(df)

    # Train and evaluate classification models
    train_classification_models(X_scaled, y)

    # Train and evaluate clustering models
    train_clustering_models(X_scaled)


if __name__ == "__main__":
    main()
