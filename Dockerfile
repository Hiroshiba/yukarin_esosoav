FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-devel

SHELL ["/bin/bash", "-c"]
WORKDIR /app

# install requirements
COPY requirements.txt /app/
RUN pip install -r <(cat requirements.txt | grep -v torch) -r requirements_dev.txt
