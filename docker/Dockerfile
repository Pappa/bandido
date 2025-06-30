FROM tensorflow/tensorflow:2.19.0-jupyter

RUN apt-get update && \
    apt-get -y upgrade

RUN apt-get install -y libcairo2-dev pkg-config python3-dev

RUN pip install pip --upgrade
RUN pip install setuptools --upgrade

RUN pip install \
    tf_agents==0.19.0 \
    tf_keras==2.19.0 \
    numpy==2.1.3 \
    pandas==2.3.0 \
    pycairo

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

#7d6b1799c5a861809a33f0b9de8c930d13d5aeec7bf6c7ff⁠