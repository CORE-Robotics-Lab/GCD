import torch
import dgl
import torch.nn as nn
import dgl.function as fn
from dgl.nn.pytorch import Sequential


class ExampleLayer(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, graph, n_feat, e_feat):
        with graph.local_scope():
            graph.ndata['h'] = n_feat
            graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
            n_feat += graph.ndata['h']
            graph.apply_edges(fn.u_add_v('h', 'h', 'e'))
            e_feat += graph.edata['e']
            return n_feat, e_feat
g = dgl.DGLGraph()
g.add_nodes(3)
g.add_edges([0, 1, 2, 0, 1, 2, 0, 1, 2], [0, 0, 0, 1, 1, 1, 2, 2, 2])
net = Sequential(ExampleLayer(), 
                 nn.ReLU(),
                 ExampleLayer(),
                 nn.ReLU(), 
                 ExampleLayer()
                 )


n_feat = torch.rand(3, 4)
e_feat = torch.rand(9, 4)
out = net(g, n_feat, e_feat)

print(out)
