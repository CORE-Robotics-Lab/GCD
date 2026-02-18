import torch as th
from torch import nn

import dgl.function as fn
from dgl.nn import TypedLinear
from dgl.ops import edge_softmax

class HetConvGAT(nn.Module):
    def __init__(self, 
                 in_feat, 
                 out_feat, 
                 num_heads, 
                 num_rels, 
                 num_classes, 
                 l_alpha=0.2, 
                 bias=False, 
                 activation=None, # not used?
                 self_loop=True, 
                 dropout=0.0, 
                 layer_norm=False):
        
        super().__init__()

        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_heads = num_heads
        self.num_rels = num_rels
        self.num_classes = num_classes

        # Message weights for each node class
        self.own_class_msg_weights = TypedLinear(self.in_feat, self.out_feat * self.num_heads, num_classes)

        # Message weights for each class to class communication channel
        self.class2class_msg_weights = TypedLinear(in_feat, out_feat * self.num_heads, num_rels)

        # Multi-head attention weights for class to class communication
        self.class2class_att_weights = nn.ModuleList([TypedLinear(out_feat * 2, 1, num_rels)
                                                      for _ in range(self.num_heads)])

        self.leaky_relu = nn.LeakyReLU(negative_slope=l_alpha)

        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop
        self.layer_norm = layer_norm

        # bias; TODO: where do we apply this bias? not present in the HetNet paper
        if self.bias:
            raise NotImplementedError
            # self.h_bias = nn.Parameter(th.Tensor(out_feat))
            # nn.init.zeros_(self.h_bias)

        # TODO: add layer norm if needed
        # TODO: dropout

    def own_class_feature_reduction(self, nodes):
        """ Apply class wise feature reduction to the source nodes.
        This computes W_i*h_j in HetNet Eq. 2 where j indexes the destination node/agent and i is the class of agent j
        """
        ntypes = nodes.data['ntype']
        out = self.own_class_msg_weights(nodes.data['h'], ntypes, self.presorted)

        return {f'W_i*h_j_head_{i}': feat
                for i, feat in enumerate(out.view(-1, self.num_heads, self.out_feat).permute(1, 0, 2))}

    def class2class_feature_reduction(self, edges):
        """ Apply feature reduction based on class to class edge relations on the destination nodes.
        This computes W_l2i*h_k in HetNet Eq. 2 where k indexes the neighbors of agent j and l is the class of agent k
        """
        etypes = edges.data['etype']
        out = self.class2class_msg_weights(edges.src['h'], etypes, self.presorted)

        return {f'W_l2i*h_k_head_{i}': feat
                for i, feat in enumerate(out.view(-1, self.num_heads, self.out_feat).permute(1, 0, 2))}

    def class2class_attention_coefficients(self, edges):
        """ Compute attention coefficients based on class to class edge relations on the destination nodes.
        This computes alpha_{jk}^{l2i} in HetNet Eq. 4
        """
        etypes = edges.data['etype']
        e_l2i_heads = []  # one for each head
        for i in range(self.num_heads):
            feat_src = edges.data[f'W_l2i*h_k_head_{i}']
            feat_dst = edges.src[f'W_i*h_j_head_{i}']
            e_l2i = self.class2class_att_weights[i](th.cat((feat_src, feat_dst), dim=-1), etypes, self.presorted)
            e_l2i_heads.append(e_l2i)

        return {f'e_l2i_head_{i}': feat for i, feat in enumerate(e_l2i_heads)}

    def udf_u_mul_e(self, edges):
        """ Compute all the incoming messages for each node. These are the messages from its neighbors weighted by their 
        attention coefficients.
        """
        feat_heads = []
        for i in range(self.num_heads):
            feat = edges.data[f'W_l2i*h_k_head_{i}'] * edges.data[f'alpha_l2i_head_{i}']
            feat_heads.append(feat)

        return {f'm_head{i}': feat for i, feat in enumerate(feat_heads)}

    def udf_sum(self, nodes):
        """ Sums all of the incoming messages for each node. This is the aggregation step in the GAT layer.
        """
        feat_heads = []
        for i in range(self.num_heads):
            feat_heads.append(th.sum(nodes.mailbox[f'm_head{i}'], dim=1))

        return {f'feat_head_{i}': feat for i, feat in enumerate(feat_heads)}

    def udf_apply_node_update(self, nodes):
        """ Applies the weighted messages of its neighbors to the node's own features.
        """
        feat_heads = []
        for i in range(self.num_heads):
            feat_heads.append(nodes.data[f'W_i*h_j_head_{i}'] + nodes.data[f'feat_head_{i}'])

        return {f'feat_head_{i}': feat for i, feat in enumerate(feat_heads)}


    def forward(self, g, feat, etypes, ntypes, norm=None, *, presorted=False):
        """Forward computation.
        Parameters
        ----------
        g : DGLGraph
            The graph.
        feat : torch.Tensor
            A 2D tensor of node features. Shape: :math:`(|V|, D_{in})`.
        etypes : torch.Tensor or list[int]
            An 1D integer tensor of edge types. Shape: :math:`(|E|,)`.
        ntypes: torch.Tensor or list[int]
            An 1D integer tensor of node types. Shape: :math:`(|V|,)`.
        norm : torch.Tensor, optional
            An 1D tensor of edge norm value.  Shape: :math:`(|E|,)`.
        presorted : bool, optional
            Whether the edges of the input graph have been sorted by their types.
            Forward on pre-sorted graph may be faster. Graphs created
            by :func:`~dgl.to_homogeneous` automatically satisfy the condition.
            Also see :func:`~dgl.reorder_graph` for sorting edges manually.

        Returns
        -------
        torch.Tensor
            New node features. Shape: :math:`(|V|, D_{out})`.
        """
        self.presorted = presorted
        with g.local_scope():
            
            
            try: 
                g.srcdata['h'] = feat
                g.dstdata['h'] = feat
                
            except:
                print("error")
                

            g.edata['etype'] = etypes
            g.ndata['ntype'] = ntypes # this the class of each node

            # compute feature reductions
            g.apply_nodes(self.own_class_feature_reduction)
            g.apply_edges(self.class2class_feature_reduction)

            # compute attention coefficients for each class to class edge relation
            g.apply_edges(self.class2class_attention_coefficients)

            for i in range(self.num_heads):
                e_l2i = self.leaky_relu(g.edata.pop(f'e_l2i_head_{i}'))
                g.edata[f'alpha_l2i_head_{i}'] = edge_softmax(g, e_l2i)

            g.update_all(message_func=self.udf_u_mul_e, 
                         reduce_func=self.udf_sum,
                         apply_node_func=self.udf_apply_node_update)

            return th.stack([g.ndata[f'feat_head_{i}'] for i in range(self.num_heads)], dim=0)
