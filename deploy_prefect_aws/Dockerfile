FROM python:3.12-slim

RUN pip install --no-cache-dir prefect==3.9.5 "boto3>=1.42,<2"
COPY persistent_server.py /opt/prefect/persistent_server.py

EXPOSE 4200

CMD ["python", "-u", "/opt/prefect/persistent_server.py"]
