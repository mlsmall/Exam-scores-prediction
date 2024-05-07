FROM python:3.12-slim-bookworm
WORKDIR /app
COPY . /app

run apt update -y && apt install awscli -y

RUN pip install -r requirements.txt
CMD ["python3", "app.py"]

