import nvidia.dali.ops as ops
import nvidia.dali.tfrecord as tfrec
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIClassificationIterator as PyTorchIterator
from nvidia.dali.plugin.pytorch import DALIGenericIterator


class TFRecordPipeline(Pipeline):
    def __init__(
            self, 
            architecture: str, 
            batch_size: int, 
            num_threads: int, 
            device_id: int, 
            num_gpus: int, 
            shard_id: int, 
            num_shards: int, 
            tfrecord: str, 
            tfrecord_idx: str,
            scalar_shape: list = [1,],
            src_dst_shape: list = [456981,], 
            nodes_shape: list = [5535,],
            imgs_kde_quantil_shape: list = [3, 64, 64],
            label_key: str = 'labels',
            img_key: str = 'imgs_kde_quantil',
            exec_pipelined: bool = False, 
            exec_async: bool = False, 
            is_shuffle: bool = False
        ) -> None:
        super(TFRecordPipeline, self).__init__(
            batch_size, 
            num_threads, 
            device_id, 
            exec_pipelined=False, 
            exec_async=False, 
            prefetch_queue_depth=8
            )
        """
        DALI data pipeline

        :architecture: str, architecture of the model
        :batch_size: int, batch size
        :num_threads: int, number of threads
        :device_id: int, device id
        :num_gpus: int, number of gpus
        :shard_id: int, shard id, see https://docs.nvidia.com/deeplearning/dali/user-guide/docs/advanced_topics_sharding.html for more details
        :num_shards: int, number of shards
        :tfrecord: str, path to the tfrecord file
        :tfrecord_idx: str, path to the tfrecord index file
        :scalar_shape: list, shape of the scalars, needed for extraction
        :src_dst_shape: list, shape of the source and destination nodes, needed for extraction
        :nodes_shape: list, shape of the nodes, needed for extraction
        :imgs_kde_quantil_shape: list, shape of the imgs_kde_quantil, needed for extraction
        :label_key: str, label key
        :img_key: str, img key
        :exec_pipelined: bool, execute pipeline in pipelined mode
        :exec_async: bool, execute pipeline in async mode
        :is_shuffle: bool, shuffle the dataset
        """

        self.architecture = architecture
        self.extract_graph = False
        self.extract_imgs = False
        self.label_key = label_key
        self.img_key = img_key
        
        assert self.architecture in ['DDCL', 'DDCL_CNN', 'DDCL_GNN']
        if self.architecture in ['DDCL']:
            self.extract_graph = True
            self.extract_imgs = True
        elif self.architecture in ['DDCL_CNN']:
            self.extract_imgs = True
        elif self.architecture in ['DDCL_GNN']:
            self.extract_graph = True
            
        self.input = ops.TFRecordReader(
            path = tfrecord, 
            index_path = tfrecord_idx,
            random_shuffle=is_shuffle,
            pad_last_batch = True,
            shard_id=shard_id,
            num_shards=num_shards,
            prefetch_queue_depth=8,
            features = {
                "n_subgraph_edges" : tfrec.FixedLenFeature([], tfrec.string, ""),
                "num_unique_nodes" : tfrec.FixedLenFeature([], tfrec.string, ""),
                "labels" : tfrec.FixedLenFeature([], tfrec.string, ""),
                "shuffled_subgraph_src_nodes" : tfrec.FixedLenFeature([], tfrec.string, ""),
                "shuffled_subgraph_dst_nodes" : tfrec.FixedLenFeature([], tfrec.string, ""),
                "unique_nodes_shuffled_order" : tfrec.FixedLenFeature([], tfrec.string, ""),
                "drnl_labels" : tfrec.FixedLenFeature([], tfrec.string, ""),
                "imgs_kde_quantil" : tfrec.FixedLenFeature([], tfrec.string, ""),
                }
            )

        self.decode_float32 = ops.PythonFunction(function=self.extract_view_float32, num_outputs=1)
        self.decode_int32 = ops.PythonFunction(function=self.extract_view_int32, num_outputs=1)
        self.decode_int16 = ops.PythonFunction(function=self.extract_view_int16, num_outputs=1)
        self.decode_uint8 = ops.PythonFunction(function=self.extract_view_uint8, num_outputs=1)
        self.reshape_scalar = ops.Reshape(shape=scalar_shape)
        self.reshape_src_dst = ops.Reshape(shape=src_dst_shape)
        self.reshape_nodes = ops.Reshape(shape=nodes_shape)
        self.reshape_imgs = ops.Reshape(shape=imgs_kde_quantil_shape)

    def extract_view_float32(self, data):
        """ Helper function to extract the float32 view from the TFRecordReader
        """
        ext_data = data.view('<f4')
        return ext_data

    def extract_view_int32(self, data):
        """ Helper function to extract the int32 view from the TFRecordReader
        """
        ext_data = data.view('<i4')
        return ext_data

    def extract_view_int16(self, data):
        """ Helper function to extract the int16 view from the TFRecordReader
        """
        ext_data = data.view('<i2')
        return ext_data
    
    def extract_view_uint8(self, data):
        """ Helper function to extract the uint8 view from the TFRecordReader
        """
        ext_data = data.view('<u1')
        return ext_data
    
    def define_graph(self):
        """ Defines the computation performed at every call to the data pipeline. Binary tfrecord data needs reshaping, see helper functions.
            Depending on the architecture, the graph and the causal images will be extracted or not. 
            If the architecture is DDCL, the graph and the causal images will be extracted.
            If the architecture is DDCL_CNN, the causal images will be extracted.
            If the architecture is DDCL_GNN, the graph will be extracted.
        """

        inputs = self.input(name="Reader")
        output = []
        if self.extract_graph:
            n_subgraph_edges = self.reshape_scalar(self.decode_int32(inputs['n_subgraph_edges']))
            num_unique_nodes = self.reshape_scalar(self.decode_int16(inputs['num_unique_nodes']))
            shuffled_subgraph_src_nodes = self.reshape_src_dst(self.decode_int16(inputs['shuffled_subgraph_src_nodes']))
            shuffled_subgraph_dst_nodes = self.reshape_src_dst(self.decode_int16(inputs['shuffled_subgraph_dst_nodes']))
            unique_nodes_shuffled_order = self.reshape_nodes(self.decode_int16(inputs['unique_nodes_shuffled_order']))
            drnl_labels = self.reshape_nodes(self.decode_int32(inputs['drnl_labels']))
            output = [n_subgraph_edges, num_unique_nodes, shuffled_subgraph_src_nodes, shuffled_subgraph_dst_nodes, unique_nodes_shuffled_order, drnl_labels]
        if self.extract_imgs:
            imgs_kde_quantil = self.reshape_imgs(self.decode_float32(inputs[self.img_key]))
            output.append(imgs_kde_quantil)

        labels = self.reshape_scalar(self.decode_uint8(inputs[self.label_key]))
        output.append(labels)

        return output
            