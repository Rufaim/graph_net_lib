import tensorflow as tf

from .processors import EdgeProcessor, GlobalProcessor, NodeProcessor



# adapted from https://github.com/deepmind/graph_nets/blob/64771dff0d74ca8e77b1f1dcd5a7d26634356d61/graph_nets/modules.py#L247

class GraphNetwork(tf.keras.Model):
    """Implementation of a Graph Network.

    See https://arxiv.org/abs/1806.01261 for more details.
    """
    def __init__(self,node_processor: NodeProcessor,
                 edge_processor: EdgeProcessor,
                 global_processor: GlobalProcessor):
        """Initializes the GraphNetwork module.

        Args:
        :param node_processor: An instance of `graphnetlib.NodeProcessor` that
                will be used to perform per-node computations.
        :param edge_processor: An instance of `graphnetlib.EdgeProcessor` that
                will be used to perform per-edge computations.
        :param global_processor: An instance of `graphnetlib.GlobalProcessor` that
                will be used to perform per-global computations.
        """
        super(GraphNetwork, self).__init__()

        assert isinstance(node_processor, NodeProcessor)
        assert isinstance(edge_processor, EdgeProcessor)
        assert isinstance(global_processor, GlobalProcessor)

        self.node_processor = node_processor
        self.edge_processor = edge_processor
        self.global_processor = global_processor

    def call(self, input_graph, training=None, mask=None):
        new_graph = self.edge_processor(input_graph, training=training, mask=mask)
        new_graph = self.node_processor(new_graph, training=training, mask=mask)
        return self.global_processor(new_graph, training=training, mask=mask)



class MessagePassingNetwork(tf.keras.Model):
    """Implementation of a Message Passing Network.

    Note that unlike Graph Network the global prediction does not include processed edges.

    See https://arxiv.org/abs/1704.01212 for more details.
    """
    def __init__(self, node_processor: NodeProcessor,
                 edge_processor: EdgeProcessor,
                 global_processor: GlobalProcessor):
        """Initializes the MessagePassingNetwork module.

        Args:
        :param node_processor: An instance of `graphnetlib.NodeProcessor` that
                will be used to perform per-node computations.
        :param edge_processor: An instance of `graphnetlib.EdgeProcessor` that
                will be used to perform per-edge computations.
        :param global_processor: An instance of `graphnetlib.GlobalProcessor` that
                will be used to perform per-global computations.
        """
        super(MessagePassingNetwork, self).__init__()

        assert isinstance(node_processor, NodeProcessor)
        assert isinstance(edge_processor, EdgeProcessor)
        assert isinstance(global_processor, GlobalProcessor)

        self.node_processor = node_processor
        self.edge_processor = edge_processor
        self.global_processor = global_processor

    def call(self, input_graph, training=None, mask=None):
        new_graph = self.edge_processor(input_graph,training=training,mask=mask)
        new_graph = self.node_processor(new_graph,training=training,mask=mask)
        new_graph = self.global_processor(new_graph,training=training,mask=mask)
        return new_graph.replace(edges=input_graph.edges)



class NonLocalNetwork(tf.keras.Model):
    """Implementation of a Non-Local neural network.

    See https://arxiv.org/abs/1711.07971 for more details.
    """
    def __init__(self, node_processor: NodeProcessor,
                 edge_processor: EdgeProcessor):
        """Initializes the NonLocalNetwork module.

        Args:
        :param node_processor: An instance of `graphnetlib.NodeProcessor` that
                will be used to perform per-node computations.
        :param edge_processor: An instance of `graphnetlib.EdgeProcessor` that
                will be used to perform per-edge computations.
        """
        super(NonLocalNetwork, self).__init__()

        assert isinstance(node_processor, NodeProcessor)
        assert isinstance(edge_processor, EdgeProcessor)

        self.node_processor = node_processor
        self.edge_processor = edge_processor

    def call(self, input_graph, training=None, mask=None):
        new_graph = self.edge_processor(input_graph, training=training, mask=mask)
        new_graph = self.node_processor(new_graph, training=training, mask=mask)
        return new_graph.replace(edges=input_graph.edges)



# adapted from https://github.com/deepmind/graph_nets/blob/64771dff0d74ca8e77b1f1dcd5a7d26634356d61/graph_nets/modules.py#L148

class RelationNetwork(tf.keras.Model):
    """Implementation of a Relation Network.

    See https://arxiv.org/abs/1706.01427 for more details.
    """
    def __init__(self, edge_processor: EdgeProcessor,
                 global_processor: GlobalProcessor):
        """Initializes the RelationNetwork module.

        Args:
        :param edge_processor: An instance of `graphnetlib.EdgeProcessor` that
                will be used to perform per-edge computations.
        :param global_processor: An instance of `graphnetlib.GlobalProcessor` that
                will be used to perform per-global computations.
        """
        super(RelationNetwork, self).__init__()

        assert isinstance(edge_processor, EdgeProcessor)
        assert isinstance(global_processor, GlobalProcessor)

        self.edge_processor = edge_processor
        self.global_processor = global_processor

    def call(self, input_graph, training=None, mask=None):
        new_graph = self.edge_processor(input_graph, training=training, mask=mask)
        new_graph = self.global_processor(new_graph, training=training, mask=mask)
        return input_graph.replace(globals=new_graph.globals)



# adapted from https://github.com/deepmind/graph_nets/blob/64771dff0d74ca8e77b1f1dcd5a7d26634356d61/graph_nets/modules.py#L414

class DeepSet(tf.keras.Model):
    """DeepSets module.
    Implementation for the model described in https://arxiv.org/abs/1703.06114
    (M. Zaheer, S. Kottur, S. Ravanbakhsh, B. Poczos, R. Salakhutdinov, A. Smola).
    See also PointNet (https://arxiv.org/abs/1612.00593, C. Qi, H. Su, K. Mo,
    L. J. Guibas) for a related model.

    This module operates on sets, which can be thought of as graphs without
    edges. The nodes features are first updated based on their value and the
    globals features, and new globals features are then computed based on the
    updated nodes features.
    """
    def __init__(self, node_processor: NodeProcessor,
                 global_processor: GlobalProcessor):
        """Initializes the NonLocalNetwork module.

        Args:
        :param node_processor: An instance of `graphnetlib.NodeProcessor` that
                will be used to perform per-node computations.
        :param global_processor: An instance of `graphnetlib.GlobalProcessor` that
                will be used to perform per-global computations.
        """
        super(DeepSet, self).__init__()

        assert isinstance(node_processor, NodeProcessor)
        assert isinstance(global_processor, GlobalProcessor)

        self.node_processor = node_processor
        self.global_processor = global_processor

    def call(self, input_graph, training=None, mask=None):
        new_graph = self.node_processor(input_graph, training=training, mask=mask)
        new_graph = self.global_processor(new_graph, training=training, mask=mask)
        return input_graph.replace(globals=new_graph.globals)



# adapted from https://github.com/deepmind/graph_nets/blob/64771dff0d74ca8e77b1f1dcd5a7d26634356d61/graph_nets/modules.py#L332

class IndependentNetwork(tf.keras.Model):
    """A graph module that applies models to the graph elements independently.

    The inputs and outputs are graphs. The corresponding models are applied to
    each element of the graph (edges, nodes and globals) in parallel and
    independently of the other elements. It can be used to encode or
    decode the elements of a graph.
    """
    def __init__(self, node_net=None, edge_net=None, global_net=None):
        """Initializes the GraphNetwork module.

        Args:
        :param node_net: An instance of `tensorflow.keras.Model` or None that
                will be used to perform independent per-node update.
        :param edge_net: An instance of `tensorflow.keras.Model` or None that
                will be used to perform per-edge computations.
        :param global_net: An instance of `tensorflow.keras.Model` or None that
                will be used to perform per-global computations.

        Note that at least one of the node_net, edge_net or global_net
        should not be set to none or else usage of the module does not make any sense.
        """
        super(IndependentNetwork, self).__init__()

        assert any(x is not None for x in [node_net, edge_net, global_net])
        assert isinstance(node_net,tf.keras.Model) or node_net is None
        assert isinstance(edge_net, tf.keras.Model) or edge_net is None
        assert isinstance(global_net, tf.keras.Model) or global_net is None

        self.node_net = node_net
        self.edge_net = edge_net
        self.global_net = global_net

    def call(self, input_graph, training=None, mask=None):
        new_nodes = input_graph.nodes
        new_edges = input_graph.edges
        new_globals = input_graph.globals

        if self.node_net is not None:
            new_nodes = self.node_net(new_nodes)
        if self.edge_net is not None:
            new_edges = self.edge_net(new_edges)
        if self.global_net is not None:
            new_globals = self.global_net(new_globals)
        return input_graph.replace(nodes=new_nodes,edges=new_edges,globals=new_globals)
