docker run --rm --gpus all --name=dex_ai8x-training -p 45555:22 -v $(dirname $PWD):/home/$USER/git --ipc=host -itd dex_ai8x-training


# --ipc=host is required to increase shared memory (https://github.com/ultralytics/yolov3/issues/283)