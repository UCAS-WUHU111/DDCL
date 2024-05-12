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

   
def create_output_image(run_dict: dict) -> None:
    res = []
    for k, d in run_dict.items():
        snr, _, auc = d.values()
        res.append([snr, auc])
    res = np.asarray(res)
    res = res[res[:, 0].argsort()[::-1]]
    
    fig, ax = plt.subplots(1, 1)
    ax.plot(np.arange(res.shape[0]), res[:, 1], label='DDCL', marker="X")
    ax.set_xticks(np.arange(res.shape[0]))
    ax.set_xticklabels(res[:, 0].astype(str).tolist())
    ax.legend()
    ax.set_ylim([0.35, 1.0])
    ax.set_xlabel('SNR')
    ax.set_ylabel('AUC')
    ax.set_title('Tanh - Direct causal effects')
    plt.text(0.05, 0.05, 'The results are not averaged \nas presented in the corresponding paper!', horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, color='red')
    fig.savefig('/results/DDCL_Tanh_DirectCausalEffects.png')
    
        
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--BATCH_SIZE', default=75, type=int,
                        help='number of sample in a batch')
    parser.add_argument('--WORKERS', default=8, type=int)
    parser.add_argument('--ARCHITECTURE', type=str)
    parser.add_argument('--GRAPH', type=str, help='lasso_graph or pearson_pearson')
    parser.add_argument('--MAX_EDGES', type=int)
    parser.add_argument('--MAX_NODES', type=int)
    parser.add_argument('--LABEL_KEY', type=str, default='labels')
    parser.add_argument('--IMG_KEY', type=str, default='imgs_kde_quantil')
    parser.add_argument('--X_KEY', type=str)
    args = parser.parse_args()

    run_dict = OrderedDict(
        {
        "snr_10.000": {"snr": 10.0, "data_path": "/data/snr10.000"},
        "snr_6.000": {"snr": 6.0, "data_path": "/data/snr6.000"},
        "snr_4.000": {"snr": 4.0, "data_path": "/data/snr4.000"},
        "snr_2.000": {"snr": 2.0, "data_path": "/data/snr2.000"},
        "snr_1.000": {"snr": 1.0, "data_path": "/data/snr1.000"},
        "snr_0.750": {"snr": 0.75, "data_path": "/data/snr0.750"},
        "snr_0.500": {"snr": 0.5, "data_path": "/data/snr0.500"},
        "snr_0.250": {"snr": 0.25, "data_path": "/data/snr0.250"},
        "snr_0.100": {"snr": 0.1, "data_path": "/data/snr0.100"},
        }
    )

    args.MASTER_PORT = int(os.environ.get("MASTER_PORT", 8738))
    args.MASTER_ADDR = os.environ.get("MASTER_ADDR", "127.0.0.1")
    args.NODE_RANK = int(os.environ.get("RANK", 0))
    args.WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    rank = 0
    GPU = rank
    backend = 'nccl'

    tcp_store = dist.TCPStore(
        args.MASTER_ADDR, 
        args.MASTER_PORT, 
        args.WORLD_SIZE, 
        rank == 0
        )
    dist.init_process_group(
        backend, 
        store=tcp_store, 
        rank=rank, 
        world_size=args.WORLD_SIZE
        )
             
    GPU_IDX = GPU
    GPU = torch.device("cuda", GPU)
    torch.cuda.set_device(GPU)
   

    random_state = 0
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    os.environ['PYTHONHASHSEED'] = '0'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    for name, config in run_dict.items():
        config_args = SimpleNamespace(**{**vars(args), **config})

        config_args.TEST_FILE = os.path.join(config_args.data_path, config_args.GRAPH+'_test.tfrecord')
        config_args.TEST_IDX_FILE = os.path.join(config_args.data_path, config_args.GRAPH+'_test.idx')
        config_args.META_FILE = os.path.join(config_args.data_path, 'exp.h5')
        config_args.CHECKPOINT_PATH = os.path.join(config_args.data_path, 'checkpoint', 'model_checkpoint.chkpt')
        with h5py.File(config_args.META_FILE, 'r') as hf:
                train_max_z = np.max(hf['pearson_graph/SEAL_train_data/drnl_labels'])
                test_max_z = np.max(hf['pearson_graph/SEAL_test_data/drnl_labels'])
                config_args.__MAX_Z__ = max(train_max_z, test_max_z)
    

        # DALI data loading code
        tfrecord2idx_script = "tfrecord2idx"
        if not os.path.isfile(config_args.TEST_IDX_FILE):
            call([tfrecord2idx_script, config_args.TEST_FILE, config_args.TEST_IDX_FILE])

        assert config_args.ARCHITECTURE in [ 'DDCL', 'DDCL_CNN', 'DDCL_GNN'], "No valid architecture specified!"

        if config_args.ARCHITECTURE in ['DDCL', 'DDCL_GNN']:
            with h5py.File(config_args.META_FILE, 'r') as meta_hf:
                X = np.asarray(meta_hf[config_args.X_KEY])
                num_unique_nodes = np.asarray(meta_hf[config_args.GRAPH + '/SEAL_train_data/num_unique_nodes'])
            input_features = X.shape[0] + config_args.__MAX_Z__ + 1

        if config_args.ARCHITECTURE in ['DDCL']:
            key_list = [
                'n_subgraph_edges', 
                'num_unique_nodes',
                'shuffled_subgraph_src_nodes',
                'shuffled_subgraph_dst_nodes',
                'unique_nodes_shuffled_order',
                'drnl_labels',
                config_args.IMG_KEY,
                'labels'
                ]
        elif config_args.ARCHITECTURE in ['DDCL_GNN']:
            key_list = [
                'n_subgraph_edges', 
                'num_unique_nodes',
                'shuffled_subgraph_src_nodes',
                'shuffled_subgraph_dst_nodes',
                'unique_nodes_shuffled_order',
                'drnl_labels',
                'labels'
                ]
        elif config_args.ARCHITECTURE in ['DDCL_CNN']:
            key_list = [
                config_args.IMG_KEY,
                'labels'
                ]
        else:
            raise NotImplementedError('Architecture not allow, check that ...')

        test_pipe = dataset_dali.TFRecordPipeline(
            architecture=config_args.ARCHITECTURE, 
            batch_size=config_args.BATCH_SIZE, 
            num_threads=8, 
            device_id=GPU_IDX, 
            num_gpus=1,
            tfrecord=config_args.TEST_FILE, 
            tfrecord_idx=config_args.TEST_IDX_FILE,
            num_shards=config_args.WORLD_SIZE, 
            shard_id=rank, 
            is_shuffle=False, 
            scalar_shape=[1,],
            src_dst_shape=[config_args.MAX_EDGES,],
            nodes_shape=[config_args.MAX_NODES,],
            imgs_kde_quantil_shape=[3,64,64],
            label_key='labels',
            img_key=config_args.IMG_KEY
            )
        test_pipe.build()
        test_loader = DALIGenericIterator(
            test_pipe, 
            key_list, 
            size=int(test_pipe.epoch_size("Reader") / config_args.WORLD_SIZE),
            last_batch_padded=False, fill_last_batch=False, auto_reset=True
            )
        test_loader_len = int(math.ceil(test_loader._size / config_args.BATCH_SIZE))

        if config_args.ARCHITECTURE == 'DDCL':
            model = architectures.DDCL(
                input_features=input_features, 
                hidden_channels=32,
                num_layers=3, 
                num_unique_nodes=num_unique_nodes
            )
        elif config_args.ARCHITECTURE == 'DDCL_GNN':
            model = architectures.DDCL_GNN(
                input_features=input_features, 
                hidden_channels=32,
                num_layers=3,
                num_unique_nodes=num_unique_nodes
                )
        elif config_args.ARCHITECTURE == 'DDCL_CNN':
            model = architectures.DDCL_CNN()
        else:
            raise NotImplementedError('Model not implemented')
        
        checkpoint = torch.load(config_args.CHECKPOINT_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])


        model = model.float().cuda(GPU)
        model = DDP(model, device_ids=[GPU])
            
        with torch.no_grad():
            model.eval()

            test_total_loss = 0.
            test_total_samples = 0.

            y_pred, y_true = [], []

            test_pbar = tqdm(enumerate(test_loader), total=test_loader_len, desc='TEST')
            for i, test_batch in test_pbar:
                if config_args.ARCHITECTURE in ['DDCL']:
                    test_batch_graph, test_X, test_imgs, test_labels = convert_dali2graph(config_args.ARCHITECTURE, test_batch, X, config_args.__MAX_Z__,  label_key='labels', img_key=config_args.IMG_KEY)
                    test_batch_graph = test_batch_graph.to(GPU)
                    test_X = test_X.float().cuda(GPU)
                    test_imgs = test_imgs.float().cuda(GPU)
                    test_labels = test_labels.squeeze(dim=1).type(torch.long).cuda(GPU)
                    test_logits = model(test_batch_graph, test_X, test_imgs)
                elif config_args.ARCHITECTURE in ['DDCL_GNN']:
                    test_batch_graph, test_X, test_labels = convert_dali2graph(config_args.ARCHITECTURE, test_batch, X, config_args.__MAX_Z__,  label_key='labels')
                    test_batch_graph = test_batch_graph.to(GPU)
                    test_X = test_X.float().cuda(GPU)
                    test_labels = test_labels.squeeze(dim=1).type(torch.long).cuda(GPU)
                    test_logits = model(test_batch_graph, test_X)
                elif config_args.ARCHITECTURE in ['DDCL_CNN']:
                    test_imgs = torch.FloatTensor(test_batch[0].get(config_args.IMG_KEY))
                    test_imgs = test_imgs.cuda(GPU)
                    test_labels = torch.FloatTensor(np.asarray(test_batch[0].get('labels'), dtype=int)).squeeze(dim=1).type(torch.long).cuda(GPU)
                    test_logits = model(test_imgs)

                test_log_probs = F.log_softmax(test_logits, dim=1)
                test_loss = F.nll_loss(test_log_probs, test_labels)
                y_pred.append(F.softmax(test_logits, dim=1).cpu())
                y_true.append(test_labels.cpu())
                test_total_loss += test_loss.item() * float(test_labels.shape[0])
                test_total_samples += test_labels.shape[0]

            avg_test_loss = test_total_loss / float(test_total_samples)


            y_pred = torch.cat(y_pred, axis=0)
            y_true = torch.cat(y_true, axis=0).type(torch.float)
            test_roc_auc = roc_auc_score(y_true=y_true, y_score=y_pred[:, 1])


            output_str = '%s: avg_test_loss: %f test_roc_auc: %f \n'%(name, avg_test_loss, test_roc_auc)
            print(output_str)
            run_dict[name]['auc'] = test_roc_auc

    create_output_image(run_dict)

if __name__ == "__main__":
    main()
