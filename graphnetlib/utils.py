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

from .graph import Graph


# adapted from https://github.com/deepmind/graph_nets/blob/64771dff0d74ca8e77b1f1dcd5a7d26634356d61/graph_nets/utils_tf.py#L677

def receiver_sender_idxes_fully_connected(n_node, exclude_self_edges=True):
    """Creates sets of senders and receivers for complete digraph
    where each pair of nodes is connected with an edge.

    Args:
    :param n_node: scalar integer tensor with the total number of nodes in the graph.
    :param exclude_self_edges: bool to control self-connections.
    :return: receivers - 1D integer tensor,
            senders - 1D integer tensor.
    """
    rng = tf.range(n_node)
    receivers, senders = tf.meshgrid(rng, rng)
    n_edge = n_node**2

    if exclude_self_edges:
        ind = tf.cast(1 - tf.eye(n_node), tf.bool)
        receivers = tf.boolean_mask(receivers, ind)
        senders = tf.boolean_mask(senders, ind)
        n_edge -= n_node

    receivers = tf.reshape(tf.cast(receivers, tf.int32), [n_edge])
    senders = tf.reshape(tf.cast(senders, tf.int32), [n_edge])

    return receivers, senders


# adapted from https://github.com/deepmind/graph_nets/blob/64771dff0d74ca8e77b1f1dcd5a7d26634356d61/graph_nets/utils_tf.py#L327

def concat_graphs(list_of_graphs, axis=0):
    """Concatenates graphs along a given axis.

    If `axis` == 0, then the graphs are concatenated along the (underlying) batch dimension
    including receivers, senders, n_nodes and n_edges fields of the `graphnetlib.Graph`.
    If `axis` != 0, then there is an underlying assumption that the rreceivers, senders,
    n_nodes and n_edges fields of the provided graphs in should all match,
    but this is not checked explicitly.

    Args:
    :param list_of_graphs: A iterable of `graphnetlib.Graph`-s to concatenate that
                satisfying the constraints outlined above.
    :param axis: An axis to concatenate provided graphs on. Default is 0.
    :return: a single concatenated graph
    """

    if len(list_of_graphs) == 0:
        raise ValueError("List argument `list_of_graphs` is empty")
    if len(list_of_graphs) == 1:
        return list_of_graphs[0]

    nodes_ = tf.concat([g.nodes for g in list_of_graphs],axis=axis)
    edges_ = tf.concat([g.edges for g in list_of_graphs],axis=axis)
    globals_ = tf.concat([g.globals for g in list_of_graphs],axis=axis)

    if axis != 0:
        return list_of_graphs[0].replace(nodes=nodes_, edges=edges_, globals=globals_)

    n_nodes_per_graph = tf.stack([tf.reduce_sum(g.n_nodes) for g in list_of_graphs])
    n_edges_per_graph = tf.stack([tf.reduce_sum(g.n_edges) for g in list_of_graphs])
    offset_values = tf.cumsum(tf.concat([[0], n_nodes_per_graph[:-1]], 0))
    offsets = tf.repeat(offset_values, n_edges_per_graph,axis=0)

    receivers_ = tf.concat([g.receivers for g in list_of_graphs], axis=0) + offsets
    senders_ = tf.concat([g.senders for g in list_of_graphs], axis=0) + offsets
    n_nodes_ = tf.concat([g.n_nodes for g in list_of_graphs], axis=0)
    n_edges_ = tf.concat([g.n_edges for g in list_of_graphs], axis=0)

    return Graph(nodes_,edges_,globals_,receivers_,senders_,n_nodes_,n_edges_)
