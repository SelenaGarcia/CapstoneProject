import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def calculateAndPruneDataset(priorizationFile: str, datasetFile: str):
    # Read datasets from files
    priorization = pd.read_excel(priorizationFile)
    dataset = pd.read_csv(datasetFile)

    # filter non used columns
    columns = priorization['Name'].tolist()
    datasetFiltered = dataset[columns]

    # Replace all numbers in the columns with True or False
    datasetBoolean = datasetFiltered[columns] != 0

    # Prepare for destructive changes
    datasetTemp = datasetBoolean.copy()

    for column in datasetTemp.columns:
        if column in priorization['Name'].values:
            priority_value = priorization.loc[priorization['Name']
                                              == column, 'Priority'].iloc[0]
            # Replace values in datasetTemp with priority values if not equal to 0
            datasetTemp[column] = datasetTemp[column].apply(
                lambda x: priority_value if x != False else 0)

    # Create a new column with the sum of columns present in priorization['Name']
    datasetTemp['result'] = datasetTemp[priorization['Name']].sum(axis=1)

    X = datasetBoolean
    y = datasetTemp['result']

    return (X, y)
