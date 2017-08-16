#!/bin/sh
MODEL=/home/shiyemin/code/flownet2/models/FlowNet2/FlowNet2_weights.caffemodel.h5
PROTO=/home/shiyemin/code/flownet2/models/FlowNet2/FlowNet2_deploy.prototxt.template
INPUT=/home/shiyemin/data/ucf101/Videos
OUTPUT=/home/shiyemin/data/ucf101/frames_net

trap 'echo you hit Ctrl-C/Ctrl-\, now exiting..; pkill -P $$; exit' SIGINT SIGQUIT

source set-env.sh
GPUS=(0 1 2 3 4 5 6 7)
for((j=0;j<4;j++));do
    for((i=0;i<${#GPUS[@]};i++));do
        GPU_ID=${GPUS[$i]}
        echo "Using GPU " $GPU_ID
        CUDA_VISIBLE_DEVICES=$GPU_ID python scripts/extract-flownet.py $MODEL $PROTO $INPUT $OUTPUT --bound 20 &
        sleep 2
    done
done
wait
