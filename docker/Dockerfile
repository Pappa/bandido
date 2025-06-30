FROM tensorflow/tensorflow:latest-jupyter
# COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt-get update && \
    apt-get -y upgrade

RUN pip install pip --upgrade
RUN pip install setuptools --upgrade

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=UTF-8 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

RUN mkdir -p /app/bandido
COPY src /app/bandido/src
COPY pyproject.toml /app/bandido
COPY README.md /app/bandido
WORKDIR /app/bandido
RUN pip install .