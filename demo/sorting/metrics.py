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

import numpy as np


def compute_accuracy(output, target_nodes, target_edges):
    """Calculate model accuracy.

    Returns the number of correctly predicted shortest path nodes and the number
    of completely solved graphs (100% correct predictions).

    Args:
        output: A `graphnetlib.Graph` that contains the output graph.
        target_nodes: A `tensorflow.Tensor` that contains the targets for nodes.
        target_edges: A `tensorflow.Tensor` that contains the targets for edges.

    Returns:
        correct: A `float` fraction of correctly labeled nodes/edges.
        solved: A `float` fraction of graphs that are completely correctly labeled.

    Raises:
        ValueError: Nodes or edges (or both) must be used
    """
    xn = np.argmax(output.nodes.numpy(), axis=-1)
    yn = np.argmax(target_nodes.numpy(), axis=-1)
    correctly_predicted_nodes = xn == yn

    xe = np.argmax(output.edges.numpy(), axis=-1)
    ye = np.argmax(target_edges.numpy(), axis=-1)
    correctly_predicted_links = xe == ye

    predicted = np.mean(np.concatenate([correctly_predicted_nodes,correctly_predicted_links],axis=0))

    batched_nodes = np.logical_and.reduceat(correctly_predicted_nodes,output.n_nodes.numpy(),axis=0)
    batched_links = np.logical_and.reduceat(correctly_predicted_links, output.n_edges.numpy(), axis=0)
    solved = np.mean(np.logical_and(batched_nodes,batched_links))

    return predicted, solved
