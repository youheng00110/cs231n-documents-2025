#Windows系统
#!/bin/bash
if [ ! -d "coco_captioning" ]; then
    curl -O "http://cs231n.stanford.edu/coco_captioning.zip"
    unzip coco_captioning.zip
    rm coco_captioning.zip
fi