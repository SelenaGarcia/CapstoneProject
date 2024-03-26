import os

from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
# from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def train_model(X_train, y_train, random_state = 0, estimators = 100, epochs = 10, batch_size = 32, validation_split = 0.2) -> RandomForestClassifier:
    # Train a Random Forest Classifier Model
    model = RandomForestClassifier(n_estimators=estimators, random_state=random_state)
    model.fit(X_train, y_train)

    # # Train a Sequential Neural Network Model
    # model = Sequential([
    #             Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    #             Dense(32, activation='relu'),
    #             Dense(2, activation='softmax')  # Output layer with 2 neurons for IsEmergency and Priority
    #         ])
    
    # # Compile Model
    # model.compile(
    #             optimizer='adam',
    #             loss='sparse_categorical_crossentropy',  # Use 'sparse_categorical_crossentropy' for integer labels
    #             metrics=['accuracy']
    #         )
    
    # # Train the Model
    # model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)

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