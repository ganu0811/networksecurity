FROM python:3.10-slim-buster
WORKDIR /app   
#Set the working directory to /app in the dockerfile
COPY . /app
# Copy the current directory contents into the container at /app
RUN apt update -y && apt install awscli -y

RUN apt-get update && pip install -r requirements.txt
CMD ["python3", "app.py"]