import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from utils.main import download_dataset, convert_dataset

if __name__ == '__main__':
    print(f'Running Class: {__name__}')
    
    DATASET_URL = 'https://github.com/yaleemmlc/admissionprediction/raw/master/Results/5v_cleandf.RData'
    FILE_PATH = '../data'
    FILE_NAME = 'Results.rdata'

    # Downloads the Original Dataset (RData File)
    DATASET_RDATA_PATH = download_dataset(DATASET_URL, FILE_PATH, FILE_NAME)

    # Converts the RData File to a New CSV File
    DATASET_CSV_PATH = convert_dataset(DATASET_RDATA_PATH, DATASET_RDATA_PATH.replace('.rdata', '.csv'))

    # Loads Dataframe from CSV File
    df = pl.read_csv(DATASET_CSV_PATH)

    # Removes the First Column (Index)
    df = df.select(pl.col(df.columns[1:]))

    # Elimina las columnas donde todos sus elementos sean iguales
    columns_to_remove = df.columns[df.n_unique() == 1]
    df_witout_nunique = df.drop(columns=columns_to_remove)
    df_witout_nunique.head()

    # Gets Significant Columns
    significantColumns = ['cc_chestpain', 'cc_breathingdifficulty', 'cc_syncope', 'cc_unresponsive', 'cc_seizure-newonset', 'cc_seizure-priorhxof', 'cc_seizures', 'cc_bleeding/bruising', 'cc_hyperglycemia', 'cc_hypertension', 'cc_hypotension', 'cc_strokealert', 'cc_overdose-accidental', 'cc_overdose-intentional', 'cc_suicidal']

    # If any of the significant columns are on true, automatically has the flag of emergency
    df_witout_nunique = df_witout_nunique.to_pandas()
    df_witout_nunique['emergency_flag_column'] = df_witout_nunique[significantColumns].any(axis=1)

    # Creates a Copy of DF
    df_converted = df_witout_nunique.copy()

    # Inicializar el codificador de etiquetas
    label_encoder = LabelEncoder()

    # Identificar las variables categóricas
    categorical_columns = []
    for column in df_converted.columns:
        # Verificar si la columna es de tipo 'object' o 'category'
        if df_converted[column].dtype == 'object' or df_converted[column].dtype == 'category' :  
            categorical_columns.append(column)

    # Convertir las variables categóricas usando LabelEncoder
    for column in categorical_columns:
        df_converted[column] = label_encoder.fit_transform(df_converted[column])

    # Reemplazo los valores nulos con -999. Se podría intentar en una proxima iteracion con algo como col.mode()[0] o valores significativos reales
    df_filled = df_converted.apply(lambda col: col.fillna(-999))
    newColumnsWithNan = df_filled.isna().any().pipe(lambda x: x.index[x])

    # División de datos
    random_state = 42

    X = df_filled.drop(columns=['emergency_flag_column'])
    y = df_filled['emergency_flag_column']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    # Entrenamiento del modelo
    model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    model.fit(X_train, y_train)

    # Validación del modelo
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    confusion = confusion_matrix(y_test, y_pred)
    TP = confusion[1, 1]
    FP = confusion[0, 1]
    TN = confusion[0, 0]
    FN = confusion[1, 0]

    # Codificación de variables categóricas
    df_encoded = pd.get_dummies(df_witout_nunique)

    # Manejo de valores nulos
    imputer = SimpleImputer(strategy='most_frequent')  
    df_without_null = imputer.fit_transform(df_witout_nunique)

    # Crear un DataFrame con las características y las etiquetas codificadas
    df_plot = X.copy()
    df_plot['emergency_flag'] = y

    # Graficar la distribución de las clases
    plt.figure(figsize=(8, 6))
    sns.countplot(x='emergency_flag', data=df_plot)
    plt.title('Distribución de Clases')
    plt.xlabel('Clase')
    plt.ylabel('Conteo')
    plt.show()