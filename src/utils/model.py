import os

from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
# from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix

def train_model(X_train, y_train, random_state = 0, estimators = 100) -> RandomForestClassifier:
    # Train the Model
    model = RandomForestClassifier(n_estimators=estimators, random_state=random_state)
    model.fit(X_train, y_train)

    return model

def validate_model(model, X_test, y_test) -> list:
    # Validating the Model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Get Confusion Matrix
    confusion = confusion_matrix(y_test, y_pred)
    # TP = confusion[1, 1]
    # FP = confusion[0, 1]
    # TN = confusion[0, 0]
    # FN = confusion[1, 0]

    return [accuracy, confusion]

def save_model_to_file(model: RandomForestClassifier, file_directory: str, file_name: str) -> str:
    file_path = file_directory
    if os.path.abspath(os.curdir) not in file_path:
        file_path = os.path.join(os.path.abspath(os.curdir), file_path)
    
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    
    file_path = os.path.join(file_path, file_name)
    dump(model, file_path)

    return file_path

def read_model_from_file(file_path: str) -> RandomForestClassifier:
    model = load(file_path) 
    
    return model