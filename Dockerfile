FROM python:3.8.18-alpine3.19

RUN apk add --no-cache --update bash clang curl gcc gfortran libffi-dev lld musl-dev openssl-dev python3 python3-dev

RUN python -m pip install -U pip

ENV PYTHONUNBUFFERED 1

WORKDIR /app

RUN curl https://sh.rustup.rs -sSf | bash -s -- -y

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY src .

CMD ["python", "./src/main.py"]