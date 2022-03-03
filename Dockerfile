# install necessary tensorflow CUDA environment
FROM tensorflow/tensorflow:latest-gpu

RUN apt-get update && apt-get install -y git
RUN mkdir /init
# on the host, run: pip freeze > requirements.txt
COPY ./requirements.txt /init/requirements.txt
RUN pip3 -q install pip --upgrade
RUN pip install -r /init/requirements.txt

# mount ./data
VOLUME ./data