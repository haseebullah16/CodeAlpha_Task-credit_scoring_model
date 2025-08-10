from utils.data_loader import load_data
from utils.trainer import train_model
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

def main():
    print("Running main.py from: C:\\Users\\Haseeb\\Desktop\\credit_scoring_model")

    # Load the dataset
    X, y = load_data("data/credit_data.csv")
    print(f"Loaded X shape: {X.shape} y shape: {y.shape}")

    # Train model and get predictions
    model, X_test, y_test, y_pred = train_model(X, y)

    # Show confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Confusion Matrix")
    plt.show()

    # Show classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
