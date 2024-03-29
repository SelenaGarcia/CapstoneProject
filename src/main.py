import os
import matplotlib.pyplot as plt
import numpy as np

from utils.dataset import download_dataset, convert_dataset
from utils.eda import calculate_and_prune_dataset, prepare_data, split_train_test_data, plot_data, scale_data
from utils.path import get_absolute_path
from utils.model import train_model, validate_model, save_model_to_file, read_model_from_file
from utils.web import WebApp

if __name__ == '__main__':
    # Original RAW Dataset URL
    DATASET_URL = 'https://github.com/yaleemmlc/admissionprediction/raw/master/Results/5v_cleandf.RData'

    # Path for Dataset Storage
    DATASET_PATH = 'data'
    DATASET_FILE = 'Results.rdata'
    PRIORITY_FILE = 'Priority.xlsx'

    # Path for Model Storage
    MODEL_PATH = 'model'
    MODEL_FILE = 'Classifier.joblib'

    # Random State to Split Data
    RANDOM_STATE = 0

    # Random Forest Estimators
    ESTIMATORS = 100

    # Test and Train Size to Split Data
    TEST_SIZE = 0.2
    TRAIN_SIZE = 0.8

    # Downloads the Original Dataset (RData File)
    print('Downloading Dataset')
    DATASET_RDATA_PATH = download_dataset(DATASET_URL, DATASET_PATH, DATASET_FILE)

    # Converts the RData File to a New CSV File
    print('Converting Dataset to CSV')
    DATASET_CSV_PATH = convert_dataset(DATASET_RDATA_PATH, DATASET_RDATA_PATH.replace('.rdata', '.csv'))
    
    # Prepare Dataset to Train Model
    print('Preparing Dataset')
    PRIORITY_FILE_PATH = os.path.join(get_absolute_path(DATASET_PATH), PRIORITY_FILE)
    X, y = calculate_and_prune_dataset(PRIORITY_FILE_PATH, DATASET_CSV_PATH)
    # X, y = prepare_data(DATASET_CSV_PATH)

    # Saves X and Y Data to CSV Files
    X.to_csv(DATASET_RDATA_PATH.replace(DATASET_FILE, 'X.csv'), index=True)
    y.to_csv(DATASET_RDATA_PATH.replace(DATASET_FILE, 'y.csv'), index=True)

    # Scale Data
    X = scale_data(X)

    # Split Train and Test Data from Clean Dataframe
    print('Spliting Dataset')
    X_train, X_test, y_train, y_test = split_train_test_data(X, y, RANDOM_STATE, TEST_SIZE, TRAIN_SIZE)

    # Plot Histogram for Each Feature
    # columns = len(X.columns)
    # ncolumns = 5
    # nrows = round(columns / ncolumns) + 1
    # f = plt.figure()
    # for index in range(columns):
    #     column = X.columns[index]
    #     f.set_figheight(15)
    #     f.set_figwidth(15)
    #     plt.subplot(nrows, ncolumns, index + 1, title=column)
    #     X[column].hist(bins=len(X[column].unique()))
    
    # plt.tight_layout()
    # plt.show()

    # # Show Correlations between X and Y
    # correlations = X_train.corrwith(y_train) #X.corrwith(y)
    # top_correlations = np.abs(correlations).sort_values(ascending=False).iloc[0:10].index
    # # print(correlations)
    # # print(top_correlations)
    # print(correlations[top_correlations])

    # Defines Path for Model Storage
    model_path = os.path.join(get_absolute_path(MODEL_PATH), MODEL_FILE)

    # Defines model Variable with None as Placeholder
    model = None

    # Reads Trained Model from File
    if os.path.isfile(model_path):
        print('Reading Model from File')
        model = read_model_from_file(model_path)
    
    # Trains and Saves Trained Model to File if Doesn't Exists
    else:
        # Train Model
        print('Training Model')
        model = train_model(X_train, y_train, RANDOM_STATE, ESTIMATORS)

        # Save Trained Model to File
        print('Saving Model to File')
        model_path = save_model_to_file(model, MODEL_PATH, MODEL_FILE)
    
    # Gets Model Metrics
    # print('Validating Model')
    # accuracy, confusion = validate_model(model, X_test, y_test)

    # Plot Predictions
    # plot_data(X, y)

    #Runs Web APP
    print('Serving Model with Gradio')
    app = WebApp(model=model, priority_file_path=PRIORITY_FILE_PATH, host='0.0.0.0', port=6789)
    app.launch()