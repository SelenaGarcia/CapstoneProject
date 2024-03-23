import os
import pyreadr
import urllib.request

from src.utils.path import get_absolute_path

def download_dataset(file_url: str, file_directory: str, file_name: str) -> str:
    directory = get_absolute_path(file_directory)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    file = os.path.join(directory, file_name)
    if not os.path.isfile(file):
        urllib.request.urlretrieve(file_url, file)
    
    return file

def convert_dataset(original_path: str, destination_path: str) -> str:
    if not os.path.isfile(destination_path):
        result = pyreadr.read_r(original_path)
        df = result['df']

        df.to_csv(destination_path, index=True)
    
    return destination_path