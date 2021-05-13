import tensorflow as tf



class GlobalsAggregator(object):
    """Global aggregator for nodes or edges aggregation
    before calculating updated global properties of the graph.
    """
    def __init__(self, reducer=tf.math.unsorted_segment_sum):
        """Initializes global aggregator.

        Supported reducers:
            * tf.math.unsorted_segment_sum
            * tf.math.unsorted_segment_mean
            * tf.math.unsorted_segment_prod

        :param reducer: one of supported reducers. Default: tf.math.unsorted_segment_sum
        """
        self._reducer = reducer

    def reduce(self, input, batch_size, nums):
        """Aggreagate nodes or edges to for global update.

        See `graphnetlib.GlobalProcessor` code for examples of usage.

        :param input: tensor of nodes or edges of shape NxF.
                    Usually it is graph.nodes or graph.edges.
        :param batch_size: scalar tensor of batch size B.
                    Ususally graph.get_batch_size() is passed.
        :param nums: 1D integer tensor with numbers of nodes or edges
                    in each of the stacked graphs in a batch.
                    Usually graph.n_nodes or graph.n_edges is fed into.
        :return: aggregated tensor of shape BxF
        """
        graph_index = tf.range(batch_size)
        indices = tf.repeat(graph_index, nums, axis=0)
        return self._reducer(input, indices, batch_size)



class EdgesToNodesAggregator(object):
    """Edges aggregator for edges aggregation before
    calculating node features update for the graph.
    """
    def __init__(self, reducer=tf.math.unsorted_segment_sum):
        """Initializes edge aggregator.

        Supported:
         * tf.math.unsorted_segment_sum
         * tf.math.unsorted_segment_mean
         * tf.math.unsorted_segment_prod

        Args:
        :param reducer: one of supported reducers. Default: tf.math.unsorted_segment_sum
        """
        self._reducer = reducer

    def reduce(self, edges, idxs, num_nodes):
        """Aggreagate edges to for nodes update step.

        See `graphnetlib.NodeProcessor` code for examples of usage.

        Args:
        :param egdes: tensor of edges of shape ExF.
                    Usually graph.edges is fed.
        :param idxs: 1D integer tensor of receivers or senders indexes of size E.
                    Usually it is graph.receivers or graph.senders.
        :param num_nodes: scalar tensor containing total number of nodes N.
                    Usually graph.get_num_nodes() is passed.

        :return: aggregated tensor of shape NxF
        """
        return self._reducer(edges,idxs,num_nodes)
