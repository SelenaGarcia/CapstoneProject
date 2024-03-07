import matplotlib.pyplot as plt
# import pandas as pd
import polars as pl
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def prepare_data(dataset_path: str) -> tuple:
    # Loads Dataframe from CSV File
    df = pl.read_csv(dataset_path)

    # Removes the First Column (Index)
    df = df.select(pl.col(df.columns[1:]))

    # Remove Columns Where all Elements are Equal
    columns_to_remove = df.columns[df.n_unique() == 1]
    df_witout_nunique = df.drop(columns=columns_to_remove)
    # df_witout_nunique.head()

    # Gets Significant Columns
    significant_columns = ['cc_chestpain', 'cc_breathingdifficulty', 'cc_syncope', 'cc_unresponsive', 'cc_seizure-newonset', 'cc_seizure-priorhxof', 'cc_seizures', 'cc_bleeding/bruising', 'cc_hyperglycemia', 'cc_hypertension', 'cc_hypotension', 'cc_strokealert', 'cc_overdose-accidental', 'cc_overdose-intentional', 'cc_suicidal']

    # If any of the significant columns are on true, automatically has the flag of emergency
    df_witout_nunique = df_witout_nunique.to_pandas()
    df_witout_nunique['emergency_flag_column'] = df_witout_nunique[significant_columns].any(axis=1)

    # Creates a Copy of DF
    df_converted = df_witout_nunique.copy()

    # Inicializar el codificador de etiquetas
    label_encoder = LabelEncoder()

    # Identify Categoric Variables
    categorical_columns = []
    for column in df_converted.columns:
        # Verify if Column is Object or Category
        if df_converted[column].dtype == 'object' or df_converted[column].dtype == 'category' :  
            categorical_columns.append(column)

    # Convert Categoric Variables Using LabelEncoder
    for column in categorical_columns:
        df_converted[column] = label_encoder.fit_transform(df_converted[column])

    # Replace NA Values with -999
    df_filled = df_converted.apply(lambda col: col.fillna(-999))
    # newColumnsWithNan = df_filled.isna().any().pipe(lambda x: x.index[x])

    # Remove "emergency_flag_column" from X Values and Assign it to Y
    X = df_filled.drop(columns=['emergency_flag_column'])
    y = df_filled['emergency_flag_column']

    return (X, y)

def split_train_test_data(X, y, random_state = 0, test_size = 0.2, train_size = 0.8) -> list:
    # Split Data into X_Train, X_Test, Y_Train, Y_Test using Test Size and Train Size
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, train_size=train_size, random_state=random_state)
    return train_test_split(X, y, test_size=test_size, train_size=train_size, random_state=random_state)

def plot_data(X, y) -> None:
    # Encode Categoric Variables
    # df_encoded = pd.get_dummies(df_witout_nunique)

    # Handling Null values
    # imputer = SimpleImputer(strategy='most_frequent')
    # df_without_null = imputer.fit_transform(df_witout_nunique)

    # Create DataFrame With Encoded Labels and Characteristics
    df_plot = X.copy()
    df_plot['emergency_flag'] = y

    # Plot Class Distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(x='emergency_flag', data=df_plot)
    plt.title('Distribución de Clases')
    plt.xlabel('Clase')
    plt.ylabel('Conteo')
    plt.show()

if __name__ == '__main__':
    print(f'Running Class: {__name__}')