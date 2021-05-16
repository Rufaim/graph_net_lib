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

from graphnetlib import Graph, receiver_sender_idxes_fully_connected, concat_graphs


def create_data(list_of_values):
    """Generate inputs and targets for training. from values
    Graphs are fully connected.
    Target nodes are encoded as one-ohe where one is set for the smallest value.
    Target edges are set to one if node n+1 is immediately after node n in the sorted list.

    Args:
    list_of_values: list of tensors containing values to sort

    Returns:
    input_graph: a graph for input in model
    target_nodes: tensor containing target node values
    target_edges: tensor containing target edges values
    """

    inputs_graphs = create_input_data(list_of_values)

    sort_indexes = []
    ranks = []
    for g in inputs_graphs:
        sort_indexes_ = tf.argsort(g.nodes[...,0], axis=-1)
        ranks_ = tf.cast(tf.argsort(sort_indexes_, axis=-1), tf.float32) / (tf.cast(g.get_num_nodes(), tf.float32) - 1.0)
        sort_indexes.append(sort_indexes_)
        ranks.append(ranks_)

    target_nodes, target_edges = create_targets(inputs_graphs, sort_indexes)
    inputs_graph = concat_graphs(inputs_graphs)
    return inputs_graph, target_nodes, target_edges, sort_indexes, ranks



def create_input_data(list_of_values):
    inputs_graphs = []
    for vals in list_of_values:
        num_nodes = tf.shape(vals)[0]

        r,s = receiver_sender_idxes_fully_connected(num_nodes)
        num_edges = tf.shape(r)[0]
        graph = Graph(vals[:, None],
                      tf.zeros((num_edges,1),dtype=tf.float32),
                      tf.zeros((1,1),dtype=tf.float32),
                      tf.convert_to_tensor(r, dtype=tf.int32),
                      tf.convert_to_tensor(s, dtype=tf.int32),
                      tf.convert_to_tensor([len(vals)], dtype=tf.int32),
                      tf.convert_to_tensor([num_edges], dtype=tf.int32))
        inputs_graphs.append(graph)
    return inputs_graphs

def create_targets(list_of_input_graphs, list_of_sort_indexes):
    target_nodes = []
    target_edges = []
    for g,si in zip(list_of_input_graphs,list_of_sort_indexes):
        nodes = tf.one_hot(si[0], g.get_num_nodes())
        nodes = tf.expand_dims(nodes,-1)

        x = tf.stack((si[:-1], si[1:]))[None]
        y = tf.stack((g.senders, g.receivers), axis=1)[..., None]
        edges = tf.reshape(tf.cast(tf.reduce_any(tf.reduce_all(tf.equal(x, y), axis=1), axis=1),tf.float32), (-1, 1))

        target_nodes.append(tf.concat([nodes,1.0-nodes],axis=-1))
        target_edges.append(tf.concat([edges,1.0-edges],axis=-1))
    return tf.concat(target_nodes,axis=0),tf.concat(target_edges,axis=0)
