FROM tensorflow/tensorflow:2.18.0

# RUN apt-get update && \
#     apt-get -y upgrade

# RUN apt-get install -y libcairo2-dev pkg-config python3-dev

RUN pip install pip --upgrade

RUN pip install pipdeptree
RUN pipdeptree

# RUN pip uninstall -y jupyterlab jupyter notebook jupyterlab_widgets jupyter_server
RUN pip install jupyter_server==2.14.2

# RUN pip install setuptools --upgrade

# RUN pip install \
#     tf_agents==0.19.0 \
#     tf_keras==2.18.0 \
#     numpy==2.0.2
#     pycairo

# ENV PYTHONDONTWRITEBYTECODE=1 \
#     PYTHONUNBUFFERED=1 \
#     PYTHONIOENCODING=UTF-8 \
#     LANG=C.UTF-8 \
#     LC_ALL=C.UTF-8

RUN mkdir -p /app/bandido
COPY src /app/bandido/src
COPY pyproject.toml /app/bandido
COPY README.md /app/bandido
WORKDIR /app/bandido
RUN pip install .

RUN pipdeptree

# CMD ["bash", "-c", "source /etc/bash.bashrc && jupyter notebook --notebook-dir=/tf --ip 0.0.0.0 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''"]
CMD ["bash", "-c", "source /etc/bash.bashrc && jupyter server --ip 0.0.0.0 --no-browser --allow-root --ServerApp.token='' --ServerApp.password=''"]
