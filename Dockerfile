FROM python:3.12-slim-bookworm
WORKDIR /app
COPY . /app

RUN apt update

RUN pip install -r requirements.txt
CMD ["python3", "app.py"]