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
