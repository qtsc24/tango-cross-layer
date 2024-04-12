FROM python:3.8
COPY . /
RUN apt-get update && apt-get install -y libadios-dev libsm6 libxext6 libxrender-dev
RUN pip install matplotlib scipy numpy scikit-learn zfpy pytz opencv-python==4.2.0.32
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/