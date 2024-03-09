import os

from utils.dataset import download_dataset, convert_dataset
from utils.eda import prepare_data, split_train_test_data, plot_data
from utils.path import get_absolute_path
from utils.model import train_model, validate_model, save_model_to_file, read_model_from_file
from utils.web import WebApp

if __name__ == '__main__':
    # Original RAW Dataset URL
    DATASET_URL = 'https://github.com/yaleemmlc/admissionprediction/raw/master/Results/5v_cleandf.RData'

    # Path for Dataset Storage
    DATASET_PATH = 'data'
    DATASET_FILE = 'Results.rdata'

    # Path for Model Storage
    MODEL_PATH = 'model'
    MODEL_FILE = 'Classifier.joblib'

    # Random State to Split Data
    RANDOM_STATE = 0

    # Test and Train Size to Split Data
    TEST_SIZE = 0.2
    TRAIN_SIZE = 0.8

    # Downloads the Original Dataset (RData File)
    DATASET_RDATA_PATH = download_dataset(DATASET_URL, DATASET_PATH, DATASET_FILE)

    # Converts the RData File to a New CSV File
    DATASET_CSV_PATH = convert_dataset(DATASET_RDATA_PATH, DATASET_RDATA_PATH.replace('.rdata', '.csv'))

    # Prepare Dataset to Train Model
    X, y = prepare_data(DATASET_CSV_PATH)

    # Split Train and Test Data from Clean Dataframe
    X_train, X_test, y_train, y_test = split_train_test_data(X, y, RANDOM_STATE, TEST_SIZE, TRAIN_SIZE)

    # Defines Path for Model Storage
    model_path = os.path.join(get_absolute_path(MODEL_PATH), MODEL_FILE)

    # Defines model Variable with None as Placeholder
    model = None

    # Trains and Saves Trained Model to File if Doesn't Exists
    if not os.path.isfile(model_path):
        # Train Model
        model = train_model(X_train, y_train, RANDOM_STATE)

        # Save Trained Model to File
        model_path = save_model_to_file(model, MODEL_PATH, MODEL_FILE)
    
    # Reads Trained Model from File
    if os.path.isfile(model_path) and not model:
        model = read_model_from_file(model_path)
    
    # Gets Model Metrics
    accuracy, confusion = validate_model(model, X_test, y_test)

    # Plot Predictions
    plot_data(X, y)

    #Runs Web APP
    app = WebApp(model=model)
    app.launch()