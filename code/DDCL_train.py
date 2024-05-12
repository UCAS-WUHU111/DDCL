import os
import sys
import shutil
import glob
import argparse
import math
from tqdm import tqdm
import h5py
import time
import socket
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import dgl
from sklearn.metrics import roc_auc_score, roc_curve
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from subprocess import call
import tensorflow as tf
from types import SimpleNamespace
from collections import OrderedDict
import matplotlib.pyplot as plt
from pathlib import Path
import json
import torchmetrics

import dataset_dali
import architectures

def convert_dali2graph(architecture, batch, X, __max_z__, label_key='labels', img_key='imgs_kde_quantil'):
    """
    Function to convert DALI dataset to DGL graph
    Depending on the architecture, it will return the DGL graph and/or image representations and the labels
    :param architecture: str, name of the architecture
    :param batch: list, batch returned from DALI pipeline
    :param X: np.ndarray, data matrix, size (num_samples, num_nodes)
    :param __max_z__: int, maximum number of nodes in all subgraphs
    :param label_key: str, name of the label key
    :param img_key: str, name of the image key
    """
    outputs = []
    if architecture in ['DDCL', 'DDCL_GNN']:
        graph_edge_list_length = np.asarray(batch[0].get('n_subgraph_edges'), dtype=int)
        src = np.asarray(batch[0].get('shuffled_subgraph_src_nodes'), dtype=int)
        dst = np.asarray(batch[0].get('shuffled_subgraph_dst_nodes'), dtype=int)

        num_unique_nodes = np.asarray(batch[0].get('num_unique_nodes'), dtype=int)
        unique_nodes = np.asarray(batch[0].get('unique_nodes_shuffled_order'), dtype=int)
        drnl_label = torch.FloatTensor(np.asarray(batch[0].get('drnl_labels'), dtype=int))


        graphs = []
        X_features = []
        for i in range(graph_edge_list_length.shape[0]):
            src_nodes = np.concatenate((src[i,:int(graph_edge_list_length[i])], np.arange(num_unique_nodes[i])))
            dst_nodes = np.concatenate((dst[i,:int(graph_edge_list_length[i])], np.arange(num_unique_nodes[i])))
            g = dgl.graph((src_nodes, dst_nodes))
            graphs.append(g)

            z_labels = drnl_label[i, :int(num_unique_nodes[i])].type(torch.long)
            z_labels = F.one_hot(z_labels, __max_z__ + 1).type(torch.float)

            current_unique_nodes = unique_nodes[i, :int(num_unique_nodes[i])]
            X_feature = torch.from_numpy(X[:, current_unique_nodes].T)
            X_features.append(torch.cat((z_labels, X_feature), dim=1))
            
        overall_graph = dgl.batch(graphs)
        X = torch.cat(X_features)
        outputs.append(overall_graph)
        outputs.append(X)

    if architecture in ['DDCL', 'DDCL_CNN']:
        imgs = torch.FloatTensor(batch[0].get(img_key))
        outputs.append(imgs)
   
    labels = torch.FloatTensor(np.asarray(batch[0].get(label_key), dtype=int))
    outputs.append(labels)
    return outputs

def main():
    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('--GPUS', default=4, type=int, help='number of gpus per node')
    parser.add_argument('--EPOCHS', default=200, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--BATCH_SIZE', default=75, type=int, help='number of sample in a batch')
    parser.add_argument('--EXP_PATH', type=str, required=True, help='path to directory directory of experiments')
    parser.add_argument('--WORKERS', default=8, type=int)

    #data
    parser.add_argument('--TRAIN_FILE', type=str, required=True, help='path to training dataset')
    parser.add_argument('--TEST_FILE', type=str, required=True, help='path to testing dataset')
    parser.add_argument('--GRAPH', type=str, default="pearson_graph", choices=['lasso_graph', 'pearson_graph'], help='Select prior graph (only neccessary for DDCL and DDCL_GNN): lasso_graph, pearson_graph')
    parser.add_argument('--MAX_EDGES', type=int, default=35000, help='Max number of edges per subgraph')
    parser.add_argument('--MAX_NODES', type=int, default=1500, help='Max number of nodes per subgraph')
    parser.add_argument('--LABEL_KEY', type=str, default='labels', help='Key under which labels are stored')
    parser.add_argument('--IMG_KEY', type=str, default='imgs_kde_quantil', help='Key under which images are stored')
    parser.add_argument('--X_KEY', type=str, default='X_obs', help='Key under which observational and interventional data is stored')

    # model, optimizer, scheduler
    parser.add_argument('--ARCHITECTURE', type=str, default="DDCL", choices=['DDCL', 'DDCL_CNN', 'DDCL_GNN'], help='Select architecture: DDCL, DDCL_CNN, DDCL_GNN')    
    parser.add_argument('--LR', type=float, default=1e-4, help='Learning rate of optimizer')
    parser.add_argument('--WEIGHT_DECAY', type=float, default=1e-4, help='Weight decay of optimizer')
    parser.add_argument('--REDUCTION_FACTOR', type=float, default=0.2, help='Reduction factor of optimizer')
    parser.add_argument('--REDUCTION_PATIENCE_LEVEL', type=int, default=10, help='Number of epochs to wait before reducing learning rate')
    parser.add_argument('--REDUCTION_MIN_LR', type=float, default=1e-8, help='Minimum learning rate to reduce learning rate')
    parser.add_argument('--GNN_NUM_LAYERS', type=int, default=3, help='Number of layers in GNN')
    parser.add_argument('--GNN_NUM_CHANNELS', type=int, default=32, help='Number of channels in CNN')

    # restore from checkpoint
    parser.add_argument('--RECOVER', type=eval, default=False, help='Set true if you want to recover the model')
    parser.add_argument('--CHECKPOINT', type=str, default=None, help='Path to checkpoint you want to load')
    args = parser.parse_args()
   
    # prepare data paths
    train_path = Path(args.TRAIN_FILE)
    args.TRAIN_IDX_FILE = train_path.with_suffix('.idx').as_posix()
    args.META_FILE = Path(train_path.parent, 'exp.h5').as_posix()
    test_path = Path(args.TEST_FILE)
    args.TEST_IDX_FILE = test_path.with_suffix('.idx').as_posix()
    
   
    # load meta data -  same for train and test
    with h5py.File(args.META_FILE, 'r') as hf:
        train_max_z = int(np.max(hf['pearson_graph/SEAL_train_data/drnl_labels']))
        test_max_z = int(np.max(hf['pearson_graph/SEAL_test_data/drnl_labels']))
        args.__MAX_Z__ = int(max(train_max_z, test_max_z))
    
    # setup logging directories
    training_begin = time.strftime("%d_%m_%y___%H_%M_%S")
    args.LOG_PATH = os.path.join(args.EXP_PATH, "DDCL_%s_%s"%(args.ARCHITECTURE, training_begin))
    args.CHECKPOINT_PATH = os.path.join(args.LOG_PATH, 'checkpoints')
    if not os.path.exists(args.LOG_PATH):
        os.makedirs(args.LOG_PATH)
    if not os.path.exists(args.CHECKPOINT_PATH):
        os.makedirs(args.CHECKPOINT_PATH)

    # DALI data loading code
    tfrecord2idx_script = "tfrecord2idx"
    print('train file', args.TRAIN_FILE, file=sys.stderr)
    print('test files', args.TEST_FILE, file=sys.stderr)
    if not os.path.isfile(args.TRAIN_IDX_FILE):
        call([tfrecord2idx_script, args.TRAIN_FILE, args.TRAIN_IDX_FILE])
    if not os.path.isfile(args.TEST_IDX_FILE):
        call([tfrecord2idx_script, args.TEST_FILE, args.TEST_IDX_FILE])


    assert args.ARCHITECTURE in [ 'DDCL', 'DDCL_CNN', 'DDCL_GNN'], "No valid architecture specified!"

    # save experiment details 
    with open(os.path.join(args.LOG_PATH, 'hyper_parameters'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)

    os.environ['MASTER_ADDR'] = '127.0.0.1'
    mp.spawn(train, nprocs=args.GPUS, args=(args,))  #use DDP_spawn to spawn multiple processes, see https://pytorch.org/docs/stable/distributed.html#distributed-data-parallel for alternatives

def train(GPU, args):
    print('Architecture', args.ARCHITECTURE, 'start on GPU', GPU, flush=True)
     
    # setup backend
    args.MASTER_PORT = int(os.environ.get("MASTER_PORT", 8738))
    args.MASTER_ADDR = os.environ.get("MASTER_ADDR")
    args.N_NODES = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NNODES", 1)))
    args.NODE_RANK = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))
    args.WORLD_SIZE = args.GPUS * args.N_NODES
    RANK = args.NODE_RANK * args.GPUS + GPU 
    backend = 'nccl'
    args.NODE_NAME = socket.gethostname()
    args.NODE_IP = socket.gethostbyname(args.NODE_NAME)

    tcp_store = dist.TCPStore(args.MASTER_ADDR, args.MASTER_PORT, args.WORLD_SIZE, RANK == 0)
    dist.init_process_group(backend, 
                            store=tcp_store, 
                            rank=RANK, 
                            world_size=args.WORLD_SIZE
    )   
    GPU_IDX = GPU
    GPU = torch.device("cuda", GPU)
    torch.cuda.set_device(GPU)
    dist.barrier() # synchronize all processes

    random_state = 0
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    os.environ['PYTHONHASHSEED'] = '0'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # start logging
    log_file = open(os.path.join(args.LOG_PATH, 'log.txt'), "w")

    #load meta data
    if args.ARCHITECTURE in ['DDCL', 'DDCL_GNN']:
        with h5py.File(args.META_FILE, 'r') as meta_hf:
            X = np.asarray(meta_hf[args.X_KEY])
            num_unique_nodes = np.asarray(meta_hf[args.GRAPH + '/SEAL_train_data/num_unique_nodes'])
        input_features = X.shape[0] + args.__MAX_Z__ + 1    

    # setup key to extract data from DALI
    if args.ARCHITECTURE in ['DDCL']:
        key_list = [
            'n_subgraph_edges', 
            'num_unique_nodes',
            'shuffled_subgraph_src_nodes',
            'shuffled_subgraph_dst_nodes',
            'unique_nodes_shuffled_order',
            'drnl_labels',
            args.IMG_KEY,
            'labels'
            ]
    elif args.ARCHITECTURE in ['DDCL_GNN']:
        key_list = [
            'n_subgraph_edges', 
            'num_unique_nodes',
            'shuffled_subgraph_src_nodes',
            'shuffled_subgraph_dst_nodes',
            'unique_nodes_shuffled_order',
            'drnl_labels',
            'labels'
            ]
    elif args.ARCHITECTURE in ['DDCL_CNN']:
        key_list = [
            args.IMG_KEY,
            'labels'
            ]
    else:
        raise NotImplementedError('Architecture not allow, check that ...')


    # DALI training pipeline
    train_pipe = dataset_dali.TFRecordPipeline(
        architecture=args.ARCHITECTURE, 
        batch_size=args.BATCH_SIZE, 
        num_threads=8, 
        device_id=GPU_IDX, 
        num_gpus=1,
        tfrecord=args.TRAIN_FILE, 
        tfrecord_idx=args.TRAIN_IDX_FILE,
        num_shards=args.WORLD_SIZE, 
        shard_id=RANK, 
        is_shuffle=True, 
        scalar_shape=[1,],
        src_dst_shape=[args.MAX_EDGES,],
        nodes_shape=[args.MAX_NODES,],
        imgs_kde_quantil_shape=[3,64,64],
        label_key=args.LABEL_KEY,
        img_key=args.IMG_KEY
        )
    train_pipe.build()
    train_loader = DALIGenericIterator(train_pipe, 
                                    key_list,
                                    size=int(train_pipe.epoch_size("Reader") / args.WORLD_SIZE),
                                    last_batch_padded=False, fill_last_batch=False, auto_reset=True)
    train_loader_len = int(math.ceil(train_loader._size / args.BATCH_SIZE))

    dist.barrier()    
    
    # DALI test pipeline
    test_pipe = dataset_dali.TFRecordPipeline(
        architecture=args.ARCHITECTURE, 
        batch_size=args.BATCH_SIZE, 
        num_threads=8, 
        device_id=GPU_IDX, 
        num_gpus=1,
        tfrecord=args.TEST_FILE, 
        tfrecord_idx=args.TEST_IDX_FILE,
        num_shards=args.WORLD_SIZE, 
        shard_id=RANK, 
        is_shuffle=False, 
        scalar_shape=[1,],
        src_dst_shape=[args.MAX_EDGES,],
        nodes_shape=[args.MAX_NODES,],
        imgs_kde_quantil_shape=[3,64,64],
        label_key='labels',
        img_key=args.IMG_KEY
        )
    test_pipe.build()
    test_loader = DALIGenericIterator(test_pipe, 
                                    key_list, 
                                    size=int(test_pipe.epoch_size("Reader") / args.WORLD_SIZE),
                                    last_batch_padded=False, fill_last_batch=False, auto_reset=True)
    test_loader_len = int(math.ceil(test_loader._size / args.BATCH_SIZE))

    dist.barrier() 

    # create model, optimizer, scheduler for training
    if args.ARCHITECTURE == 'DDCL':
        model = architectures.DDCL(
            input_features=input_features, 
            hidden_channels=args.GNN_NUM_CHANNELS,
            num_layers=args.GNN_NUM_LAYERS, 
            num_unique_nodes=num_unique_nodes
        )
    elif args.ARCHITECTURE == 'DDCL_GNN':
        model = architectures.DDCL_GNN(
            input_features=input_features, 
            hidden_channels=args.GNN_NUM_CHANNELS,
            num_layers=args.GNN_NUM_LAYERS, 
            num_unique_nodes=num_unique_nodes
            )
    elif args.ARCHITECTURE == 'DDCL_CNN':
        model = architectures.DDCL_CNN()
    else:
        raise NotImplementedError('Model not implemented')
    
    if args.RECOVER:
        checkpoint = torch.load(args.CHECKPOINT_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        START_EPOCH = checkpoint['epoch'] + 1

    model = model.float().cuda(GPU)
    model = DDP(model, device_ids=[GPU])

    optimizer = torch.optim.Adam(
        params=model.parameters(), 
        lr=args.LR, 
        weight_decay=args.WEIGHT_DECAY
        )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        'max', 
        factor=args.REDUCTION_FACTOR, 
        patience=args.REDUCTION_PATIENCE_LEVEL, 
        min_lr=args.REDUCTION_MIN_LR
        )
    if args.RECOVER:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # initialize tracking metrics
    best_test_auc = 0.
    train_loss_metric = torchmetrics.MeanMetric().cuda(GPU)
    test_loss_metric = torchmetrics.MeanMetric().cuda(GPU)
    test_AUROC = torchmetrics.AUROC(num_classes=2, task='binary').cuda(GPU)

    for epoch in range(START_EPOCH, EPOCHS):
        #training
        model.train()
        if RANK == 0:
            train_pbar = tqdm(enumerate(train_loader), total=train_loader_len, desc='Epoch %s / %s TRAINING'%(epoch +1, EPOCHS))
        else:
            train_pbar = tqdm(enumerate(train_loader), total=train_loader_len, desc='Epoch %s / %s TRAINING'%(epoch +1, EPOCHS), disable=True)
        
        for i, train_batch in train_pbar:
            if args.ARCHITECTURE in ['DDCL']:
                train_batch_graph, train_X, train_imgs, train_labels = convert_dali2graph(args.ARCHITECTURE, train_batch, X, args.__MAX_Z__, label_key=args.LABEL_KEY, img_key=args.IMG_KEY)
                train_batch_graph = train_batch_graph.to(GPU)
                train_X = train_X.float().cuda(GPU)
                train_imgs = train_imgs.float().cuda(GPU)
                train_labels = train_labels.squeeze(dim=1).type(torch.long).cuda(GPU)
                optimizer.zero_grad()
                start = time.time()
                logits = model(train_batch_graph, train_X, train_imgs)
            elif args.ARCHITECTURE in ['DDCL_GNN']:
                train_batch_graph, train_X, train_labels = convert_dali2graph(args.ARCHITECTURE, train_batch, X, args.__MAX_Z__,  label_key=args.LABEL_KEY)
                train_batch_graph = train_batch_graph.to(GPU)
                train_X = train_X.float().cuda(GPU)
                train_labels = train_labels.squeeze(dim=1).type(torch.long).cuda(GPU)
                optimizer.zero_grad()
                logits = model(train_batch_graph, train_X)
            elif args.ARCHITECTURE in ['DDCL_CNN']:
                train_imgs = torch.FloatTensor(train_batch[0].get(args.IMG_KEY))
                train_imgs = train_imgs.cuda(GPU)
                train_labels = torch.FloatTensor(np.asarray(train_batch[0].get(args.LABEL_KEY), dtype=int)).squeeze(dim=1).type(torch.long).cuda(GPU)
                optimizer.zero_grad()
                logits = model(train_imgs)
                
            log_probs = F.log_softmax(logits, dim=1)
            loss = F.nll_loss(log_probs, train_labels)
            loss.backward()
            optimizer.step()
            
            # update loss tracking and progress bar
            train_loss_metric(loss)
            avg_train_loss = train_loss_metric.compute()

            postfix_str = 'loss: %f avg_loss: %f' %(loss.item(), avg_train_loss.item())
            train_pbar.set_postfix_str(postfix_str)

        # average training loss and reset metric
        avg_train_loss = train_loss_metric.compute()
        train_loss_metric.reset()


        #test
        with torch.no_grad():
            model.eval()
            if RANK == 0:
                test_pbar = tqdm(enumerate(test_loader), total=test_loader_len, desc='Epoch %s / %s TESTING'%(epoch +1, EPOCHS))
            else:
                test_pbar = tqdm(enumerate(test_loader), total=test_loader_len, desc='Epoch %s / %s TESTING'%(epoch +1, EPOCHS), disable=True)

            for i, test_batch in test_pbar:
                if args.ARCHITECTURE in ['DDCL']:
                    test_batch_graph, test_X, test_imgs, test_labels = convert_dali2graph(args.ARCHITECTURE, test_batch, X, args.__MAX_Z__,  label_key='labels', img_key=args.IMG_KEY)
                    test_batch_graph = test_batch_graph.to(GPU)
                    test_X = test_X.float().cuda(GPU)
                    test_imgs = test_imgs.float().cuda(GPU)
                    test_labels = test_labels.squeeze(dim=1).type(torch.long).cuda(GPU)
                    optimizer.zero_grad()
                    start_test = time.time()
                    test_logits = model(test_batch_graph, test_X, test_imgs)
                    end_test = time.time()
                elif args.ARCHITECTURE in ['DDCL_GNN']:
                    test_batch_graph, test_X, test_labels = convert_dali2graph(args.ARCHITECTURE, test_batch, X, args.__MAX_Z__,  label_key='labels')
                    test_batch_graph = test_batch_graph.to(GPU)
                    test_X = test_X.float().cuda(GPU)
                    test_labels = test_labels.squeeze(dim=1).type(torch.long).cuda(GPU)
                    optimizer.zero_grad()
                    test_logits = model(test_batch_graph, test_X)
                elif args.ARCHITECTURE in ['DDCL_CNN']:
                    test_imgs = torch.FloatTensor(test_batch[0].get(args.IMG_KEY))
                    test_imgs = test_imgs.cuda(GPU)
                    test_labels = torch.FloatTensor(np.asarray(test_batch[0].get('labels'), dtype=int)).squeeze(dim=1).type(torch.long).cuda(GPU)
                    optimizer.zero_grad()
                    test_logits = model(test_imgs)
 

                test_log_probs = F.log_softmax(test_logits, dim=1)
                test_loss = F.nll_loss(test_log_probs, test_labels)

                test_loss_metric(test_loss)
                test_AUROC(test_log_probs[:, 1], test_labels)

            # average test metrics and reset for next epoch
            avg_test_loss = test_loss_metric.compute()
            test_auroc = test_AUROC.compute()
            test_loss_metric.reset()
            test_AUROC.reset()
            
            # update lr
            scheduler.step(test_auroc)

            if RANK == 0:
                output_str = 'Epoch %s avg_train_loss: %f avg_test_loss: %f test_roc_auc: %f lr: %.3e \n' % (epoch+1,
                                                                                           avg_train_loss.item() if avg_train_loss.item() != 0 else -100.,
                                                                                           avg_test_loss.item(),
                                                                                           test_auroc.item(),
                                                                                           optimizer.__getattribute__('param_groups')[0]['lr']
                                                                                            )
                tqdm.write(output_str)
                log_file.write(output_str)
                log_file.flush()

            if RANK == 0: 
                if test_auroc > best_test_auc + 0.001:
                    best_test_auc = test_auroc
                    filelist = glob.glob(os.path.join(args.CHECKPOINT_PATH, "best_roc_auc_model_*.chkpt"))
                    for f in filelist:
                        os.remove(f)
                    torch.save({'model_state_dict': model.module.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'scheduler_state_dict': scheduler.state_dict(),
                                'epoch': epoch},
                               os.path.join(args.CHECKPOINT_PATH, 'best_roc_auc_model_%s_%f.chkpt' % (epoch, best_test_auc)))
                

    print('finished')

if __name__ == "__main__":
    main()