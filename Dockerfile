FROM python:3.8.18-slim-bookworm

WORKDIR /app

COPY src .
COPY requirements.txt .

RUN python -m pip install -U pip

RUN pip install -r requirements.txt

CMD ["python", "./src/main.py"]