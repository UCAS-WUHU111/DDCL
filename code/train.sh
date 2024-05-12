#!/usr/bin/env bash


GRAPH="pearson_graph"
ARCH="DDCL"
LABEL_KEY="labels"
IMG_KEY="imgs_kde_quantil"
X_KEY="X_obs"
EXP_PATH="/results"

# use test_data for training and testing
# this is just for demonstration of the code, change data files to adapt to your dataset


python /code/DDCL_train.py \
                    --EXP_PATH ${EXP_PATH}\
                    --TRAIN_FILE /data/snr10.000/pearson_graph_test.tfrecord\
                    --TEST_FILE /data/snr10.000/pearson_graph_test.tfrecord\
                    --BATCH_SIZE 50\
                    --ARCHITECTURE ${ARCH}\
                    --GRAPH ${GRAPH}\
                    --LABEL_KEY ${LABEL_KEY}\
                    --IMG_KEY ${IMG_KEY}\
                    --X_KEY ${X_KEY}\
                    --MAX_EDGES 35000\
                    --MAX_NODES 1500
                                   

