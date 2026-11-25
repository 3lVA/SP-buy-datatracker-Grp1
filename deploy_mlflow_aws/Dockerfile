FROM python:3.12-slim

RUN pip install --no-cache-dir mlflow==3.18.0 "boto3>=1.42,<2"

WORKDIR /mlflow
COPY persistent_server.py /mlflow/persistent_server.py

EXPOSE 5000
CMD ["python", "-u", "/mlflow/persistent_server.py"]
