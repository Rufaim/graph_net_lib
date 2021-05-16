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


def broadcast_globals(globals, sizes):
    """Broadcasts global features to edges or nodes of a graph.

    See `graphnetlib.Graph` methods for examples of usage.

    Args:
    :param globals: tensor with global features BxF
    :param sizes: 1D integer tensor with numbers of size N containing nodes or
                edges in each of the stacked graphs in a batch.
                Typically it is graph.n_nodes or graph.n_edges.
    :return: broadcasted tensor of shape NxF
    """
    return tf.repeat(globals,repeats=sizes, axis=0)


# adapted from https://github.com/deepmind/graph_nets/blob/64771dff0d74ca8e77b1f1dcd5a7d26634356d61/graph_nets/blocks.py#L154

def broadcast_nodes_to_edges(nodes, idxs):
    """Broadcasts node features to edges set using receiver or sender indexes.

    See `graphnetlib.Graph` methods for examples of usage.

    Args:
    :param nodes: tensor with nodes features NxF
    :param idxs: 1D integer tensor of receivers or senders indexes of size E.
                Typically it is graph.receivers or graph.senders.
    :return: broadcasted tensor of shape ExF
    """
    return tf.gather(nodes, idxs, axis=0)
