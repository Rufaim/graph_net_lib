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

from graphnetlib import EdgesToNodesAggregator, broadcast_globals, broadcast_nodes_to_edges


# code for spring simulator is adapted from https://colab.research.google.com/github/deepmind/graph_nets/blob/master/graph_nets/demos/physics.ipynb

def hookes_law(receiver_nodes, sender_nodes, k, x_rest):
    """Applies Hooke's law to springs connecting some nodes.

    Args:
      receiver_nodes: Ex5 tf.Tensor of [x, y, v_x, v_y, is_fixed] features for the
        receiver node of each edge.
      sender_nodes: Ex5 tf.Tensor of [x, y, v_x, v_y, is_fixed] features for the
        sender node of each edge.
      k: Spring constant for each edge.
      x_rest: Rest length of each edge.

    Returns:
      Nx2 Tensor of the force [f_x, f_y] acting on each edge.
    """
    diff = receiver_nodes[..., 0:2] - sender_nodes[..., 0:2]
    x = tf.norm(diff, axis=-1, keepdims=True)
    force_magnitude = tf.multiply(k, (x - x_rest) / x)
    force = -1 * force_magnitude * diff
    return force



class SpringMassSimulator(object):
    """Implements a basic Physics Simulator"""

    def __init__(self, step_size):
        self._step_size = step_size
        self._aggregator = EdgesToNodesAggregator(reducer=tf.math.unsorted_segment_sum)

    def step(self,nodes,edges,globals,senders,receivers,num_nodes):
        total_nodes = tf.reduce_sum(num_nodes)
        receiver_nodes = broadcast_nodes_to_edges(nodes,receivers)
        sender_nodes = broadcast_nodes_to_edges(nodes,senders)
        spring_force_per_edge = hookes_law(receiver_nodes, sender_nodes,
                                           edges[..., 0:1],
                                           edges[..., 1:2])
        spring_force_per_node = self._aggregator.reduce(spring_force_per_edge,receivers,total_nodes)
        gravity = broadcast_globals(globals,num_nodes)

        # euler integration step
        force_per_node = spring_force_per_node + gravity
        # set forces to zero for fixed nodes
        force_per_node *= 1 - nodes[..., 4:5]
        updated_velocities = nodes[..., 2:4] + force_per_node * self._step_size
        new_pos = nodes[..., :2] + updated_velocities * self._step_size
        new_nodes = tf.concat([new_pos, updated_velocities, nodes[..., 4:5]], axis=-1)
        return new_nodes

    def __call__(self, graph):
        return self.step(graph.nodes,graph.edges,graph.globals,graph.senders,graph.receivers,graph.n_nodes)


class GraphNetSimulator(object):
    def __init__(self,model,step_size):
        self.model = model
        self._step_size = step_size

    def __call__(self, graph):
        nodes = graph.nodes
        new_graph = self.model(graph)[-1]
        new_pos = nodes[..., :2] + new_graph.nodes * self._step_size * (1 - nodes[..., 4:5])
        new_nodes = tf.concat([new_pos, new_graph.nodes, nodes[..., 4:5]], axis=-1)
        return new_nodes



def simulate(simulator,graph,num_steps):
    nodes_per_step = tf.TensorArray(dtype=graph.nodes.dtype, size=num_steps + 1, element_shape=graph.nodes.shape)
    new_graph = graph
    for i in range(num_steps):
        nodes_per_step = nodes_per_step.write(i, new_graph.nodes)
        nodes = simulator(new_graph)
        new_graph = new_graph.replace(nodes=nodes)
    nodes_per_step = nodes_per_step.write(num_steps, new_graph.nodes)
    return nodes_per_step.stack()

