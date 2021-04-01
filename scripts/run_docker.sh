#docker run -it --ulimit memlock=-1 --ulimit stack=67108864 -v $COCO_DIR:/coco --ipc=host --gpus=all encoding -v /root/code:/data/code xxx /bin/bash
docker run -it --ulimit memlock=-1 --ulimit stack=67108864 -v /home/pengbo/workspace/dataset:/workspace/encoding/data --ipc=host --gpus=all encoding
docker run -it --ulimit memlock=-1 --ulimit stack=67108864 -v /home/pengbo/workspace/dataset:/workspace/encoding/data --ipc=host --gpus=all encoding
