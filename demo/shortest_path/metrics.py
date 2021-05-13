import numpy as np


def compute_accuracy(output_graph, target_graph):
    """Calculate model accuracy.

    Returns the number of correctly predicted shortest path nodes and the number
    of completely solved graphs (100% correct predictions).

    Args:
        output_graph: A `graphnetlib.Graph` that contains output graph.
        target_graph: A `graphnetlib.Graph` that contains targets.

    Returns:
        correct: A `float` fraction of correctly labeled nodes/edges.
        solved: A `float` fraction of graphs that are completely correctly labeled.

    Raises:
        ValueError: Nodes or edges (or both) must be used
    """

    xn = np.argmax(output_graph.nodes.numpy(), axis=-1)
    yn = np.argmax(target_graph.nodes.numpy(), axis=-1)
    correctly_predicted_nodes = xn == yn

    xe = np.argmax(output_graph.edges.numpy(), axis=-1)
    ye = np.argmax(target_graph.edges.numpy(), axis=-1)
    correctly_predicted_links = xe == ye

    predicted = np.mean(np.concatenate([correctly_predicted_nodes, correctly_predicted_links], axis=0))

    batched_nodes = np.logical_and.reduceat(correctly_predicted_nodes, output_graph.n_nodes.numpy(), axis=0)
    batched_links = np.logical_and.reduceat(correctly_predicted_links, output_graph.n_edges.numpy(), axis=0)
    solved = np.mean(np.logical_and(batched_nodes, batched_links))

    return predicted, solved