from data.fetch_data import load_and_preprocess_data
from models.svm_model import train_svm, evaluate_model, hyperparameter_tuning
from utils.visualization import visualize_predictions

def main():
    # Load and preprocess data
    x_train, y_train, x_test, y_test = load_and_preprocess_data()

    # Train SVM model
    model = train_svm(x_train, y_train)

    # Evaluate the model
    accuracy = evaluate_model(model, x_test, y_test)
    print(f"Test set accuracy: {accuracy}")

    # Hyperparameter tuning (optional)
    best_params = hyperparameter_tuning(x_train, y_train)
    print(f"Best parameters: {best_params}")

    # Visualize some predictions
    visualize_predictions(model, x_test, y_test)

if __name__ == "__main__":
    main()
