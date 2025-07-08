# bandido - multi-armed bandit experiments

```bash
docker build -t bandido:latest -f docker/Dockerfile .
```

```bash
docker run --rm --runtime=nvidia --gpus all -it --mount src=`pwd`/notebooks,target=/tf/notebooks,type=bind -p 8888:8888 bandido:latest
```