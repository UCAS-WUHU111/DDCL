#!/usr/bin/env bash


GRAPH="pearson_graph"
ARCH="DDCL"
LABEL_KEY="labels"
IMG_KEY="imgs_kde_quantil"
X_KEY="X_obs"
DIR="/data"
CHECKPOINT="/data/DDCL_TEST_DDCL/checkpoints/best_roc_auc_model_12_0.827796.chkpt"


python /code/DDCL_test.py \
                    --BATCH_SIZE 50\
                    --ARCHITECTURE ${ARCH}\
                    --GRAPH ${GRAPH}\
                    --LABEL_KEY ${LABEL_KEY}\
                    --IMG_KEY ${IMG_KEY}\
                    --X_KEY ${X_KEY}\
                    --MAX_EDGES 35000\
                    --MAX_NODES 1500
                                   

