# Copyright 2021 Rufaim (https://github.com/Rufaim)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf

from .aggregators import EdgesToNodesAggregator, GlobalsAggregator



# adapted from https://github.com/deepmind/graph_nets/blob/64771dff0d74ca8e77b1f1dcd5a7d26634356d61/graph_nets/blocks.py#L386

class EdgeProcessor(tf.keras.Model):
    """Edge processing block.

    A block that updates the features of each edge in a batch of graphs based on
    (a subset of) the previous edge features, the features of the adjacent nodes,
    and the global features of the corresponding graph.

    See https://arxiv.org/abs/1806.01261 for more details.
    """
    def __init__(self, net, use_edges=True, use_receivers=True, use_senders=True, use_globals=True):
        """Initializes the edge processing block.

        :param net: `tensorflow.keras.Model` instance containing a neural network for
                    edge updates. Network is taking tensor of edges ExF1 and returning
                    new set of edges ExF2 where number of edges is flexible. Typically,
                    this would be inputing and outputing tensor of rank 2, but may also
                    operate on higher ranks.
        :param use_edges: Whether to use edge attributes. Default is True.
        :param use_receivers: Whether to use receiver node attributes. Default is True.
        :param use_senders: Whether to use sender node attributes. Default is True.
        :param use_globals: Whether to use global attributes. Default is True.

        Note that at least one of the use_edges, use_receivers, use_senders or use_globals
        keys should be set to True or else usage of the processor does not make any sense.
        """
        super(EdgeProcessor, self).__init__()

        assert isinstance(net,tf.keras.Model)
        assert any([use_edges,use_receivers,use_senders,use_globals])

        self.net = net
        self._use_edges = use_edges
        self._use_receivers = use_receivers
        self._use_senders = use_senders
        self._use_globals = use_globals

    def call(self, graph, training=None, mask=None):
        edges = []

        if self._use_edges:
            edges.append(graph.edges)
        if self._use_receivers:
            edges.append(graph.broadcast_receiver_nodes_to_edges())
        if self._use_senders:
            edges.append(graph.broadcast_sender_nodes_to_edges())
        if self._use_globals:
            edges.append(graph.broadcast_globals_to_edges())

        new_edges = self.net(tf.concat(edges, axis=-1),training=training,mask=mask)
        return graph.replace(edges=new_edges)




# adapted from https://github.com/deepmind/graph_nets/blob/64771dff0d74ca8e77b1f1dcd5a7d26634356d61/graph_nets/blocks.py#L487

class NodeProcessor(tf.keras.Model):
    """Node processing block.

    A block that updates the features of each node in batch of graphs based on
    (a subset of) the previous node features, the aggregated features of the
    adjacent edges, and the global features of the corresponding graph.

    See https://arxiv.org/abs/1806.01261 for more details.
    """
    def __init__(self, net, use_nodes=True, use_receivers=True, use_senders=False, use_globals=True,
                 receivers_aggregator=None, senders_aggregator=None):
        """Initializes the node processing module.

        Args:
        :param net: `tensorflow.keras.Model` instance containing a neural network for
                    edge updates. Network is taking tensor of edges ExF1 and returning
                    new set of edges ExF2 where number of edges is flexible. Typically,
                    this would be inputing and outputing tensorf of rank 2, but may also
                    operate on higher ranks.
        :param use_nodes: Whether to use node attributes. Default is True.
        :param use_receivers: Whether to use receiver edge attributes. Default is True.
        :param use_senders: Whether to use sender edge attributes. Default is False.
        :param use_globals: Whether to use global attributes. Default is True.
        :param receivers_aggregator: An instance of `graphnetlib.EdgesToNodesAggregator` or None. Default is None.
                    Received edge aggregation method. If None default EdgesToNodesAggregator is used.
        :param senders_aggregator: An instance of `graphnetlib.EdgesToNodesAggregator` or None. Default is None.
                    Sent edge aggregation method. If None default EdgesToNodesAggregator is used.

        Note that at least one of the use_nodes, use_receivers, use_senders or use_globals
        keys should be set to True or else usage of the processor does not make any sense.
        """
        super(NodeProcessor, self).__init__()

        assert isinstance(net, tf.keras.Model)
        assert any([use_nodes,use_receivers,use_senders,use_globals])

        if use_receivers:
            if receivers_aggregator is None:
                receivers_aggregator=EdgesToNodesAggregator()
            assert isinstance(receivers_aggregator,EdgesToNodesAggregator)
        if use_senders:
            if senders_aggregator is None:
                senders_aggregator=EdgesToNodesAggregator()
            assert isinstance(senders_aggregator,EdgesToNodesAggregator)

        self.net = net
        self._use_nodes = use_nodes
        self._use_receivers = use_receivers
        self._use_senders = use_senders
        self._use_globals = use_globals
        self._receivers_aggregator = receivers_aggregator
        self._senders_aggregator = senders_aggregator

    def call(self, graph, training=None, mask=None):
        nodes = []

        if self._use_nodes:
            nodes.append(graph.nodes)
        if self._use_receivers:
            nodes.append(self._receivers_aggregator.reduce(graph.edges,graph.receivers,graph.get_num_nodes()))
        if self._use_senders:
            nodes.append(self._senders_aggregator.reduce(graph.edges,graph.senders,graph.get_num_nodes()))
        if self._use_globals:
            nodes.append(graph.broadcast_globals_to_nodes())

        new_nodes = self.net(tf.concat(nodes, axis=-1),training=training,mask=mask)
        return graph.replace(nodes=new_nodes)




# adapted from https://github.com/deepmind/graph_nets/blob/64771dff0d74ca8e77b1f1dcd5a7d26634356d61/graph_nets/blocks.py#L605

class GlobalProcessor(tf.keras.Model):
    """Global processing block.

    A block that updates the global features of each graph in a batch based on
    (a subset of) the previous global features, the aggregated features of the
    edges of the graph, and the aggregated features of the nodes of the graph.

    See https://arxiv.org/abs/1806.01261 for more details.
    """
    def __init__(self, net, use_nodes=True, use_edges=True, use_globals=True,
                        node_aggregator=None, edge_aggregator=None):
        """Initializes the global processing module.

        Args:
        :param net: `tensorflow.keras.Model` instance containing a neural network for
                    edge updates. Network is taking tensor of edges ExF1 and returning
                    new set of edges ExF2 where number of edges is flexible. Typically,
                    this would be inputing and outputing tensorf of rank 2, but may also
                    operate on higher ranks.
        :param use_nodes: Whether to use node attributes. Default is True.
        :param use_edges: Whether to use edge attributes. Default is True.
        :param use_globals: Whether to use global attributes. Default is True.
        :param node_aggregator: An instance of `graphnetlib.GlobalsAggregator` or None. Default is None.
                    Node aggregation method. If None default EdgesToNodesAggregator is used.
        :param edge_aggregator: An instance of `graphnetlib.GlobalsAggregator` or None. Default is None.
                    Edge aggregation method. If None default EdgesToNodesAggregator is used.

        Note that at least one of the use_nodes, use_edges or use_globals
        keys should be set to True or else usage of the processor does not make any sense.
        """
        super(GlobalProcessor, self).__init__()

        assert isinstance(net, tf.keras.Model)
        assert any([use_nodes,use_edges,use_globals])

        if use_nodes:
            if node_aggregator is None:
                node_aggregator = GlobalsAggregator()
            assert isinstance(node_aggregator, GlobalsAggregator)
        if use_edges:
            if edge_aggregator is None:
                edge_aggregator = GlobalsAggregator()
            assert isinstance(edge_aggregator, GlobalsAggregator)

        self.net = net
        self._use_nodes = use_nodes
        self._use_edges = use_edges
        self._use_globals = use_globals
        self._node_aggregator = node_aggregator
        self._edge_aggregator = edge_aggregator

    def call(self, graph, training=None, mask=None):
        globs = []

        if self._use_nodes:
            globs.append(self._node_aggregator.reduce(graph.nodes,graph.get_batch_size(),graph.n_nodes))
        if self._use_edges:
            globs.append(self._edge_aggregator.reduce(graph.edges,graph.get_batch_size(),graph.n_edges))
        if self._use_globals:
            globs.append(graph.globals)

        new_globals = self.net(tf.concat(globs, axis=-1),training=training,mask=mask)
        return graph.replace(globals=new_globals)
