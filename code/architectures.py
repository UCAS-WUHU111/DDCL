import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn

class DGCNN_partial(torch.nn.Module):
    """ Graph Convolutional Neural Network for feature extraction
        GNN tower network
    """
    def __init__(self, input_features, hidden_channels, num_layers, num_unique_nodes, GNN=dgl.nn.pytorch.conv.GraphConv, k=0.6):
        super(DGCNN_partial, self).__init__()
        """ Initialize network
            :input_features: int, number of input features per node
            :hidden_channels: int, number of hidden channels per layer
            :num_layers: int, number of hidden layers
            :num_unique_nodes: int, number of unique nodes
            :GNN: graph convolutional neural network class
            :k: float, percentile of nodes to keep
        """
        if k < 1:  # Transform percentile to number.
            num_nodes = sorted(num_unique_nodes.tolist())
            k = num_nodes[int(math.ceil(k * len(num_nodes))) - 1]
            k = max(10, k)
        self.k = int(k)

        self.gcnn1 = GNN(input_features, hidden_channels)
        self.gcnn2 = GNN(hidden_channels, hidden_channels)
        self.gccn3 = GNN(hidden_channels, hidden_channels)
        self.gccn4 = GNN(hidden_channels, 1)

        total_latent_dim = hidden_channels * num_layers + 1
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=total_latent_dim,
                               stride=total_latent_dim)
        self.avgpool1d = nn.AvgPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32,
                               kernel_size=math.ceil(self.k / 2), stride=1)

        self.sortpool = dgl.nn.pytorch.glob.SortPooling(k=self.k)

    def forward(self, overall_graph, X):
        """ Forward pass
            :overall_graph: dgl.DGLGraph, overall graph
            :X: torch.Tensor, input features
        """

        # add self-loops to the graph needed so that GraphConv works see docs
        overall_graph_conv = dgl.add_self_loop(overall_graph)
        g_out1 = torch.tanh(self.gcnn1(graph=overall_graph_conv, feat=X))
        g_out2 = torch.tanh(self.gcnn2(graph=overall_graph_conv, feat=g_out1))
        g_out3 = torch.tanh(self.gccn3(graph=overall_graph_conv, feat=g_out2))
        g_out4 = torch.tanh(self.gccn4(graph=overall_graph_conv, feat=g_out3))


        # concatenate all computed node features
        g_out = torch.cat((g_out1, g_out2, g_out3, g_out4), dim=-1)

        # Global pooling.
        x = self.sortpool(overall_graph, g_out)
        x = x.unsqueeze(1)  # [num_graphs, 1, k * hidden]
        x = F.relu(self.conv1(x))

        if self.k % 2 != 0:
            x = self.avgpool1d(F.pad(x, (0, 1)))
        else:
            x = self.avgpool1d(x)

        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # [num_graphs, dense_dim]
        return x





######################################################################################################################
##########  ResNet with full preactivation, normalized and augmented data, average pooling and dropout ############
######################################################################################################################

class _ResNet_Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, prev_layer_depth, expansion=4, stride_3x3=1,  padding_3x3=1, conv_identity=False, stride_conv_identity=1, PReLU=True):
        super(_ResNet_Bottleneck, self).__init__()
        """ ResNet Bottleneck
            :inplanes: int, number of input channels
            :planes: int, number of output channels
            :prev_layer_depth: int, number of previous layers
            :expansion: int, expansion factor
            :stride_3x3: int, stride of 3x3 convolution
            :padding_3x3: int, padding of 3x3 convolution
            :conv_identity: bool, whether to use identity mapping
            :stride_conv_identity: int, stride of identity mapping
            :PReLU: bool, whether to use PReLU activation
        """            

        self.outplanes = planes*expansion
        self.conv_identity = conv_identity
        self.conv1 = nn.Conv2d(in_channels=inplanes, out_channels=planes, kernel_size=1)
        self.conv1_bn = nn.BatchNorm2d(prev_layer_depth)
        self.conv2 = nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=3, stride=stride_3x3, padding=padding_3x3)
        self.conv2_bn = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(in_channels=planes, out_channels=expansion*planes, kernel_size=1)
        self.conv3_bn = nn.BatchNorm2d(planes)
        if conv_identity:
            self.conv_identity_layer = nn.Conv2d(inplanes, planes*expansion, kernel_size=1, stride=stride_conv_identity)
        self._PReLU = PReLU
        if self._PReLU:
            self.conv1_PReLU = nn.PReLU()
            self.conv2_PReLU = nn.PReLU()
            self.conv3_PReLU = nn.PReLU()

    def forward(self, x, activation=F.relu):
        """ Forward pass
            :x: torch.Tensor, input features
        """
        identity = x
        if self._PReLU:
            out = self.conv1(self.conv1_PReLU(self.conv1_bn(x)))
            out = self.conv2(self.conv2_PReLU(self.conv2_bn(out)))
            out = self.conv3(self.conv3_PReLU(self.conv3_bn(out)))
        else:
            out = self.conv1(activation(self.conv1_bn(x)))
            out = self.conv2(activation(self.conv2_bn(out)))
            out = self.conv3(activation(self.conv3_bn(out)))
        if self.conv_identity:
            identity = self.conv_identity_layer(x)
        out += identity
        return out

class ResNet(nn.Module):
   def __init__(
        self, 
        ) -> None:
       super(ResNet, self).__init__()
       """ ResNet constructor
           CNN tower network
       """ 

       # define stem 
       self.conv1 = nn.Conv2d(3, 16, kernel_size=5, padding=2)
       self.conv1_bn = nn.BatchNorm2d(16)

       # define stages
       self.stage_1a = self._make_layer(inplanes=16, planes=16, prev_layer_depth=16, blocks=1, conv_identity=True)
       self.stage_1b = self._make_layer(inplanes=64, planes=16, prev_layer_depth=64, blocks=2)
       self.stage_2a = self._make_layer(inplanes=64, planes=32, prev_layer_depth=64, blocks=1)
       self.stage_2b = self._make_layer(inplanes=128, planes=32, prev_layer_depth=128, blocks=3)
       self.stage_3a = self._make_layer(inplanes=128, planes=64, prev_layer_depth=128, blocks=1)
       self.stage_3b = self._make_layer(inplanes=256, planes=64, prev_layer_depth=256, blocks=5)
       self.stage_4a = self._make_layer(inplanes=256, planes=128, prev_layer_depth=256, blocks=1)
       self.stage_4b = self._make_layer(inplanes=512, planes=128, prev_layer_depth=512, blocks=2)

       # define output transformation
       self.conv5a_bn = nn.BatchNorm2d(512)
       self.conv5a = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1)
       self.conv5b_bn = nn.BatchNorm2d(128)
       self.conv5b = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1)
       self.conv5c_bn = nn.BatchNorm2d(128)
       self.conv5c = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=1)

       self.drop = nn.Dropout(p=0.2)
       self.avgPool = nn.AvgPool2d(3, stride=2, padding=1)


   def forward(self, input, activation=F.relu):
       """ Forward pass
           :input: torch.Tensor, input features
           :activation: torch.nn.functional, activation function
        """
       output = F.relu(self.conv1_bn(self.conv1(input)))
       output = self.stage_1a(output)
       output = self.stage_1b(output)
       output = self.avgPool(output)
       output = self.stage_2a(output)
       output = self.stage_2b(output)
       output = self.avgPool(output)
       output = self.stage_3a(output)
       output = self.stage_3b(output)
       output = self.avgPool(output)
       output = self.stage_4a(output)
       output = self.stage_4b(output)

       output = F.relu(self.conv5a(self.conv5a_bn(output)))
       output = F.relu(self.conv5b(self.conv5b_bn(output)))
       output = F.relu(self.conv5c(self.conv5c_bn(output)))

       return output

   def _make_layer(self, inplanes, planes, blocks, prev_layer_depth, expansion=4, stride_3x3=1, padding_3x3=1, conv_identity=True, stride_conv_identitiy=1, PReLU=True):
        """ Helper function, make a layer of ResNet
           :inplanes: int, input channels
           :planes: int, output channels
           :blocks: int, number of blocks
           :prev_layer_depth: int, previous layer depth
           :expansion: int, expansion factor
           :stride_3x3: int, stride of 3x3 convolution
           :padding_3x3: int, padding of 3x3 convolution
           :conv_identity: bool, whether to use identity mapping
           :stride_conv_identity: int, stride of identity mapping
           :PReLU: bool, whether to use PReLU activation
        """
        layers = []
        for _ in range(blocks):
            layers.append(_ResNet_Bottleneck(inplanes, planes, prev_layer_depth=prev_layer_depth, stride_3x3=stride_3x3, padding_3x3=padding_3x3, conv_identity=conv_identity, stride_conv_identity=stride_conv_identitiy, PReLU=PReLU))

        return nn.Sequential(*layers)


#################################################################################################
##### Full DDCL
#################################################################################################
class DDCL(nn.Module):
    def __init__(self, input_features, hidden_channels, num_layers, num_unique_nodes):
        super(DDCL, self).__init__()
        """DDCL constructor
           initializes the full DDCL network. CNN tower architecture is fixed. GNN tower architecture might be varied. 
           :input_features: int, input features used in GNN
           :hidden_channels: int, number of hidden channels used in GNN
           :num_layers: int, number of GNN layers
           :num_unique_nodes: int, number of unique nodes
        """
        self.gnn = DGCNN_partial(input_features=input_features, hidden_channels=hidden_channels, num_layers=num_layers,
                          num_unique_nodes=num_unique_nodes)
        self.gnn_projection = nn.Sequential(
            nn.Linear(32, 128),
            nn.PReLU(),
            nn.BatchNorm1d(128)
        )
        
        self.cnn = ResNet()
        self.cnn_projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8192, 2048),
            nn.PReLU(),
            nn.BatchNorm1d(2048),
            nn.Linear(2048, 128),
            nn.PReLU(),
            nn.BatchNorm1d(128)
        )

        self.fusion_block = nn.Sequential(
            nn.Linear(in_features=256, out_features=128), 
            nn.PReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(in_features=128, out_features=64),
            nn.PReLU(),
            nn.Linear(in_features=64, out_features=2)
        )        
       
        

    def forward(self, graph, X, imgs):
        """ Forward pass
           :graph: torch.Tensor, graph
           :X: torch.Tensor, node features
           :imgs: torch.Tensor, image features
        """
        gnn_output = self.gnn_projection(self.gnn(graph, X))
        cnn_output = self.cnn_projection(self.cnn(imgs))
        
        output = self.fusion_block(torch.cat((gnn_output, cnn_output), dim=1))
        return output
    
#################################################################################################
##### CNN Tower Network
#################################################################################################
class DDCL_CNN(nn.Module):
    def __init__(self,):
        super(DDCL_CNN, self).__init__()
        """ CNN tower constructor
        """
        self.cnn = ResNet()
        self.cnn_projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8192, 2048),
            nn.PReLU(),
            nn.BatchNorm1d(2048),
            nn.Linear(2048, 128),
            nn.PReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(in_features=128, out_features=64),
            nn.PReLU(),
            nn.Linear(in_features=64, out_features=2)
        )

    def forward(self, imgs):
        """ Forward pass
           :imgs: torch.Tensor, image features
        """
        output = self.cnn_projection(self.cnn(imgs))
        return output


#################################################################################################
##### DDCL GNN
#################################################################################################
class DDCL_GNN(nn.Module):
    def __init__(self, input_features, hidden_channels, num_layers, num_unique_nodes):
        super(DDCL_GNN, self).__init__()
        """ GNN tower constructor
           :input_features: int, input features used in GNN
           :hidden_channels: int, number of hidden channels used in GNN
           :num_layers: int, number of GNN layers
           :num_unique_nodes: int, number of unique nodes
        """
        self.gnn = DGCNN_partial(input_features=input_features, hidden_channels=hidden_channels, num_layers=num_layers,
                          num_unique_nodes=num_unique_nodes)
        self.gnn_projection = nn.Sequential(
            nn.Linear(in_features=32, out_features=128), 
            nn.PReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(in_features=128, out_features=64),
            nn.PReLU(),
            nn.Linear(in_features=64, out_features=2)
        )

    def forward(self, graph, X):
        """ Forward pass
           :graph: torch.Tensor, graph
           :X: torch.Tensor, node features
        """
        output = self.gnn_projection(self.gnn(graph, X))
        return output

