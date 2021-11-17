import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
import collections
import numpy as np

_EPS = 1e-10

MultiGraph = collections.namedtuple('Graph', ['node_features', 'edge_sets'])


class MLP(nn.Module):
    """Two-layer fully-connected ELU net with batch norm."""

    def __init__(self, n_in, n_hid, n_out, do_prob=0.):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_out)
        self.bn = nn.BatchNorm1d(n_out)
        self.dropout_prob = do_prob

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs):
        x = inputs.view(inputs.size(0) * inputs.size(1), -1)
        x = self.bn(x)
        return x.view(inputs.size(0), inputs.size(1), -1)

    def forward(self, inputs):
        # Input shape: [num_sims, num_things, num_features]
        x = F.elu(self.fc1(inputs))
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = F.elu(self.fc2(x))
        return self.batch_norm(x)


class GraphNetBlock(nn.Module):
    def __init__(self):
        super(GraphNetBlock, self).__init__()
        """Multi-Edge Interaction Network with residual connections."""
        '''_update_edge_features'''
        self.edge_mlp = MLP(128 + 128 + 32, 128, 32)
        self.node_mlp = MLP(128 + 32, 256, 128)

    def update_node_features(self, nodes, edges, pair):
        """Aggregrates edge features, and applies node function."""

        num_nodes = nodes.shape[1]
        nodes_feature = torch.zeros((num_nodes, edges.shape[-1]))
        #nodes_feature2 = torch.zeros((num_nodes, edges.shape[-1]))
        edges = edges[0]
        nodes_feature = nodes_feature.index_add_(0, pair[:, 0], edges)
        nodes_feature = nodes_feature.index_add_(0, pair[:, 1], edges).reshape(1, num_nodes, -1)
        features = torch.cat([nodes, nodes_feature], -1)

        return self.node_mlp(features)

    def update_edge_features(self, nodes, edges, pair):
        """Aggregrates node features, and applies edge function."""
        sender = nodes[:, pair[:, 0]]
        receiver = nodes[:, pair[:, 1]]
        features = torch.concat([sender, receiver, edges], -1)
        return self.edge_mlp(features)

    def forward(self, nodes, edges, pair):
        # apply edge functions

        new_edge_sets = self.update_edge_features(nodes, edges, pair)

        # apply node function
        new_node_features = self.update_node_features(nodes, new_edge_sets, pair)

        # add residual connections
        new_node_features += nodes
        new_edge_sets = new_edge_sets + edges

        return new_node_features, new_edge_sets


class MeshGraph(nn.Module):
    def __init__(self, args, outputsize, latent_size, number_layers, message_passing_steps):
        super(MeshGraph, self).__init__()
        self.args = args
        self._latent_size = latent_size
        self._output_size = outputsize
        self._num_layers = number_layers
        self._message_passing_steps = message_passing_steps
        self.edge_feature_dim = 6  # type, length, distance
        self.node_feature_dim = 10 * 4  # x, y, dx, dy
        self.edge_hidden = 32
        
        self.hidden_dim = 128
        self.encoder_mlp_edge = MLP(self.edge_feature_dim, self.edge_hidden, self.edge_hidden)
        self.encoder_mlp_node = MLP(self.node_feature_dim, self.hidden_dim, self._latent_size)
        self.decoder_mlp = nn.Sequential(nn.Linear(self._latent_size, self.hidden_dim), nn.Tanh(),
                                         nn.Linear(self.hidden_dim, self._output_size))
        self.variance = 5e-5
        self.mse_loss = nn.MSELoss()

    def nll_gaussian(self, preds, target, add_const=False):
        variance = self.variance
        neg_log_p = ((preds - target) ** 2 / (2 * variance))
        if add_const:
            const = 0.5 * np.log(2 * np.pi * variance)
            neg_log_p += const
        return neg_log_p.sum() / (target.size(0) * target.size(1))

    def forward(self, nodes, edge_pair, edge_attr):
        """encode process decoder """
        '''Encodes node and edge features into latent features'''
        # node_set shape = [B, Time steps, Number of atoms, Position]
        # edge_set shape = [B, 2]
        # print(nodes.shape, edge_pair.shape, edge_attr.shape, "?")
        # [1, 130, 5, 2]) torch.Size([10, 2]) torch.Size([10, 2]
        # node_latents = self.encoder_mlp_node(nodes)
        # edge_set = self.encoder_mlp_edge(edge_attr)
        history_step = self.args.his
        rollout_step = self.args.pred_roll
        total_step = history_step + rollout_step
        num_balls = nodes.shape[-2]
        
        '''Decodes node features from graph'''
        edge_attr = edge_attr.reshape(1, edge_attr.shape[0], -1).double()
        
        
        for t in range(1):
            window_nodes = nodes[0] #[:, t * total_step: t * total_step + total_step][0]
            #print(window_nodes.shape,'window')

            history_nodes = window_nodes[:history_step].permute(1, 0, 2).reshape(1, num_balls, -1)
            #print(window_nodes.shape, history_nodes.shape, nodes.shape,'here')
            gt_history = window_nodes[: history_step]
            gt_future = window_nodes[history_step:]
            pos_diff = window_nodes[history_step][edge_pair[:, 0]] - window_nodes[history_step][edge_pair[:, 1]]
            pos_diff = torch.abs(pos_diff)
            edge_attr = torch.cat([edge_attr, pos_diff.reshape(1, edge_attr.shape[1], -1)], -1)

            pred_all = []
            pred_acc = []
            target_acc = []
        
            cur_position = gt_history[-1,:,:2]
            gt_cur_pos = cur_position
            cur_vel = gt_history[-1, :, 2:]
            prev_position = gt_history[-2, :, :2]
            gt_prev_pos = prev_position
            prev_vel = gt_history[-2, :, 2:]

            for pred_t in range(rollout_step):
                
                # encode
                latent_nodes = self.encoder_mlp_node(history_nodes)
                latent_edges = self.encoder_mlp_edge(edge_attr.float())

                for _ in range(self._message_passing_steps):
                    latent_nodes, latent_edges = GraphNetBlock()(latent_nodes, latent_edges, edge_pair)

                acceleration = self.decoder_mlp(latent_nodes)
                # integrate and update edge, update prev_position, update history
                prev_position = cur_position
                #print(cur_position.shape, acceleration.shape, prev_position.shape)
                cur_position = 2 * cur_position + acceleration - prev_position
                cur_vel = cur_vel + acceleration
                 
                # update edge displacement
                pos_diff = torch.abs(cur_position[:, edge_pair[:, 0]] - cur_position[:, edge_pair[:, 1]])
                edge_attr[:,:,-2:] = pos_diff
                
                pred_all.append(cur_position)
                pred_acc.append(acceleration)
                

                # update node feature with new hostory.
                history_nodes = history_nodes.reshape(1, num_balls, history_step, 4)
                new_state = torch.cat([cur_position.reshape(1, num_balls, 1, -1), cur_vel.reshape(1, num_balls, 1, -1)], -1)
                # new_state = gt_future[:, pred_t]
                history_nodes = torch.cat([history_nodes[:, :, 1:, :], new_state], -2)
                history_nodes = history_nodes.reshape(1, num_balls, -1)
            
            preds = torch.stack(pred_all, dim=1)

            loss_nll = self.nll_gaussian(preds[0], gt_future[:,:,:2])
            loss_mse = self.mse_loss(preds[0], gt_future[:,:,:2])

            return preds, loss_nll, loss_mse, gt_future
