import os

def get_absolute_path(path: str) -> str:
    absolute_path = path
    if os.path.abspath(os.curdir) not in absolute_path:
        absolute_path = os.path.join(os.path.abspath(os.curdir), absolute_path)
    
    return absolute_path

if __name__ == '__main__':
    print(f'Running Class: {__name__}')