# classification.py

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.Classification.LogisticRegression import LogisticRegression
from src.Classification.KNN import KNN
from src.Classification.SVM import SVM
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def train_classification_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Example: Train and evaluate classification models
    models = {
        "Logistic Regression": LogisticRegression(lr=0.01),
        "Euclidean K-Nearest Neighbors": KNN(k=13, distance_metric='euclidean'),
        "Manhattan K-Nearest Neighbors": KNN(k=13, distance_metric='manhattan'),
        "Support Vector Machine": SVM(),
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier()
    }

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        classification_rep = classification_report(y_test, y_pred, zero_division=1)
        confusion_mat = confusion_matrix(y_test, y_pred)

        print(f"Model: {model_name}")
        print(f"Accuracy: {accuracy}")
        print("Classification Report:")
        print(classification_rep)
        print("Confusion Matrix:")
        print(confusion_mat)
        print("-" * 40)
