# start with a clean image, install mlflow and its dependencies
FROM python:3.12-slim 

RUN pip install --no-cache-dir mlflow==3.18.0 "boto3>=1.42,<2"

# set up the working directory and create a directory for artifacts
WORKDIR /mlflow
COPY persistent_server.py /mlflow/persistent_server.py

# declare the port that the container will listen on
EXPOSE 5000

# set the command to run mlflow
CMD ["python", "-u", "/mlflow/persistent_server.py"]

