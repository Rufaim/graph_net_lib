import numpy as np
import tensorflow as tf

from graphnetlib import Graph


# code is adapted from https://colab.research.google.com/github/deepmind/graph_nets/blob/master/graph_nets/demos/physics.ipynb


def base_graph(n, d):
    """Define a basic mass-spring system graph structure.

    These are n masses (1kg) connected by springs in a chain-like structure. The
    first and last masses are fixed. The masses are vertically aligned at the
    start and are d meters apart; this is also the rest length for the springs
    connecting them. Springs have spring constant 50 N/m and gravity is 10 N in
    the negative y-direction.

    Args:
      n: number of masses
      d: distance between masses (as well as springs' rest length)

    Returns:
      data_dict: dictionary with globals, nodes, edges, receivers and senders
          to represent a structure like the one above.
    """
    # Nodes
    # Generate initial position and velocity for all masses.
    # The left-most mass has is at position (0, 0); other masses (ordered left to
    # right) have x-coordinate d meters apart from their left neighbor, and
    # y-coordinate 0. All masses have initial velocity 0m/s.
    nodes = np.zeros((n, 5), dtype=np.float32)
    half_width = d * tf.cast(n,tf.float32) / 2.0
    nodes[:, 0] = np.linspace(-half_width, half_width, num=n, endpoint=False, dtype=np.float32)
    # indicate that the first and last masses are fixed
    nodes[(0, -1), -1] = 1.

    # Edges.
    edges, senders, receivers = [], [], []
    for i in range(n - 1):
        left_node = i
        right_node = i + 1
        # The 'if' statements prevent incoming edges to fixed ends of the string
        if right_node < n - 1:
            # Left incoming edge.
            edges.append([50., d])
            senders.append(left_node)
            receivers.append(right_node)
        if left_node > 0:
            # Right incoming edge.
            edges.append([50., d])
            senders.append(right_node)
            receivers.append(left_node)

    # note graph is bidirectional
    # global features where second is gravity
    # node features Nx5  [x, y, v_x, v_y, is_fixed]
    # edges features Ex2  [spring_constant, rest_length]
    # indexes for edge receivers
    # indexes for edge senders
    return Graph(nodes,
                 edges,
                 tf.expand_dims([0.0,-9.87],0),
                 tf.convert_to_tensor(receivers,dtype=tf.int32),
                 tf.convert_to_tensor(senders,dtype=tf.int32),
                 tf.shape(nodes)[:1],
                 tf.shape(edges)[:1])


def alter_data(graph, node_noise_level, edge_noise_level, global_noise_level, seed=None):
    if isinstance(seed,tf.random.Generator):
        gen = seed
    else:
        gen = tf.random.Generator.from_seed(seed=seed)

    node_position_noise = gen.uniform(
        [tf.shape(graph.nodes)[0], 2],
        minval=-node_noise_level,
        maxval=node_noise_level)
    nodes = tf.concat([graph.nodes[..., :2] + node_position_noise, graph.nodes[..., 2:]], axis=-1)
    edge_spring_constant_noise = gen.uniform(
        [tf.shape(graph.edges)[0], 1],
        minval=-edge_noise_level,
        maxval=edge_noise_level)
    edges = tf.concat([graph.edges[..., :1] + edge_spring_constant_noise, graph.edges[..., 1:]], axis=-1)
    global_gravity_y_noise = gen.uniform(
        [tf.shape(graph.globals)[0], 1],
        minval=-global_noise_level,
        maxval=global_noise_level)
    globals = tf.concat([graph.globals[...,:1], graph.globals[...,1:]+global_gravity_y_noise],axis=-1)

    graph.replace(nodes=nodes,globals=globals)
    receiver_nodes = graph.broadcast_receiver_nodes_to_edges()
    sender_nodes = graph.broadcast_sender_nodes_to_edges()
    rest_length = tf.norm(receiver_nodes[..., :2] - sender_nodes[..., :2], axis=-1, keepdims=True)
    edges = tf.concat([edges[..., :1], rest_length], axis=-1)

    return graph.replace(edges=edges)

