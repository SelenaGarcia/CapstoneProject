FROM python:3.8.18-slim-bookworm

WORKDIR /app

COPY . .

RUN python -m pip install -U pip

RUN pip install -r requirements.txt

# CMD ["python", "./src/main.py"]
ENTRYPOINT ["python", "./src/main.py"]