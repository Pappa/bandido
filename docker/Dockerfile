FROM tensorflow/tensorflow:2.15.0-gpu-jupyter

# RUN apt-get update && \
#     apt-get -y upgrade

# RUN apt-get install -y libcairo2-dev pkg-config python3-dev

RUN pip install --upgrade pip

RUN pip install tf_keras==2.15.0
RUN pip install tf_agents==0.19.0
RUN pip install tensorflow[and-cuda]~=2.15.0
RUN pip install pandas==2.3.0
RUN pip install scikit-learn==1.7.0
RUN pip install seaborn==0.13.2
RUN pip install ccxt==4.4.95
RUN pip install keras-tuner==1.4.7

# RUN pip install numpy==2.0.2
# RUN pip install pycairo

# ENV PYTHONDONTWRITEBYTECODE=1 \
#     PYTHONUNBUFFERED=1 \
#     PYTHONIOENCODING=UTF-8 \
#     LANG=C.UTF-8 \
#     LC_ALL=C.UTF-8

# RUN mkdir -p /app/bandido
# COPY src /app/bandido/src
# COPY pyproject.toml /app/bandido
# COPY README.md /app/bandido
# WORKDIR /app/bandido
# RUN pip install .


RUN mkdir -p /tf/notebooks

CMD ["bash", "-c", "source /etc/bash.bashrc && jupyter lab --notebook-dir=/tf --ip 0.0.0.0 --no-browser --allow-root"]
