import tensorflow as tf
import numpy as np
import networkx as nx
import collections

from .broadcasters import broadcast_globals, broadcast_nodes_to_edges


GRAPH_NX_FEATURES_KEY = "features"


# adapted from https://github.com/deepmind/graph_nets/blob/64771dff0d74ca8e77b1f1dcd5a7d26634356d61/graph_nets/graphs.py#L125

class Graph(collections.namedtuple("Graph",["nodes","edges","globals","receivers","senders","n_nodes","n_edges"])):
    """Namedtuple describing `Graphs`s.
        An inheritors of `collections.namedtuple` is allowed to be used inside of tensofrlow converted code.
        For example, an instance of `Graph` can be passed into a function wrapped with `tensorflow.function`.

        An instance of this class can be constructed as
        ```
        Graph(nodes=nodes,
              edges=edges,
              globals=globals,
              receivers=receivers,
              senders=senders,
              n_node=n_node,
              n_edge=n_edge)
        ```
        where `nodes`, `edges`, `globals`, `receivers`, `senders`, `n_node` and
        `n_edge` are typically `numpy.ndarrays` or `tensorflow.Tensor`.

        Alternatively, a `Graph` instance could be constructed from any netwerkx graph using
        `Graph.from_networkx(nx_graph)`.
    """
    def __new__(cls, nodes,edges,globals,receivers,senders,n_nodes,n_edges):
        """Graph constructor. Fields of a `collections.namedtuple` are filled in the `__new__` method,
        so this method is overrided to be a constructor instead of conventional `__init__`.
        There is a set of assertions is applied before construction.

        Args:
        :param nodes: `numpy.ndarrays` or `tensorflow.Tensor` representing node features of the graph.
        :param edges: `numpy.ndarrays` or `tensorflow.Tensor` representing edge features of the graph.
        :param globals: `numpy.ndarrays` or `tensorflow.Tensor` representing global features of the graph.
        :param receivers: One-dimentional `numpy.ndarrays` or `tensorflow.Tensor` of integers representing indexes of the node
                        directed edge pointing to.
        :param senders: One-dimentional `numpy.ndarrays` or `tensorflow.Tensor` of integers representing indexes of the node
                        directed edge starting from.
        :param n_nodes: One-dimentional `numpy.ndarrays` or `tensorflow.Tensor` of integers representing number of nodes
                        in each of the stacked graphs. This argument is used for making batches of graphs with differed structure.
                        In case of manually constructing an instance for one graph, one can pass `[len(nodes)]`
        :param n_edges: One-dimentional `numpy.ndarrays` or `tensorflow.Tensor` of integers representing number of edges
                        in each of the stacked graphs. This argument is used for making batches of graphs with differed structure.
                        In case of manually constructing an instance for one graph, one can pass `[len(edges)]`
        """

        if isinstance(nodes,tf.TensorSpec):
            # inside tf.function construction call
            return super(Graph, cls).__new__(cls, nodes, edges, globals, receivers, senders, n_nodes, n_edges)
        if isinstance(nodes,tf.TensorShape):
            # inside tf.keras.Model maybe_build
            return super(Graph, cls).__new__(cls, nodes, edges, globals, receivers, senders, n_nodes, n_edges)

        nodes_ = tf.convert_to_tensor(nodes)
        edges_ = tf.convert_to_tensor(edges)
        globals_ = tf.convert_to_tensor(globals)
        receivers_ = tf.convert_to_tensor(receivers,dtype=tf.int32)
        senders_ = tf.convert_to_tensor(senders,dtype=tf.int32)
        n_nodes_ = tf.convert_to_tensor(n_nodes,dtype=tf.int32)
        n_edges_ = tf.convert_to_tensor(n_edges,dtype=tf.int32)


        tf.debugging.assert_rank_at_least(nodes_,2,message="Invalid nodes rank. Nodes shall have shape (NUM_NODES*BATCH)xFEATURES")
        tf.debugging.assert_rank_at_least(edges_, 2, message="Invalid edges rank. Edges shall have shape (NUM_EDGES*BATCH)xFEATURES")
        tf.debugging.assert_rank_at_least(globals_, 2, message="Invalid globals rank. Globals shall have shape BATCHxFEATURES")
        tf.debugging.assert_rank(receivers_, 1,message="Invalid receivers rank. Receivers shall be one-dimensional tensor")
        tf.debugging.assert_rank(senders_, 1, message="Invalid senders rank. Senders shall be one-dimensional tensor")
        tf.debugging.assert_rank(n_nodes_, 1,message="Invalid total nodes rank. Shall be one-dimensional tensor")
        tf.debugging.assert_rank(n_edges_, 1, message="Invalid total edges rank. Shall be one-dimensional tensor")

        n_nodes_shape = tf.shape(n_nodes_)

        if n_nodes_.shape.as_list()[0] is None:
            # inside tf.function with predefined input signature
            return super(Graph, cls).__new__(cls, nodes_, edges_, globals_, receivers_, senders_, n_nodes_, n_edges_)

        n_edges_shape = tf.shape(n_edges_)
        globals_shape = tf.shape(globals_)

        tf.debugging.assert_equal(n_nodes_shape, n_edges_shape, message="Mismatch of batch metadata")
        tf.debugging.assert_equal(n_nodes_shape[0], globals_shape[0], message="Global features invalid batch")

        edges_shape = tf.shape(edges_)
        receivers_shape = tf.shape(receivers_)
        senders_shape = tf.shape(senders_)

        tf.debugging.assert_equal(senders_shape[0], edges_shape[0], message="Invalid senders indexes shape")
        tf.debugging.assert_equal(receivers_shape[0], edges_shape[0], message="Invalid receivers indexes shape")
        tf.debugging.assert_equal(receivers_shape[0], senders_shape[0], message="Receiver sender indexes shape mismatch")

        num_nodes = tf.reduce_sum(n_nodes_)

        tf.debugging.Assert(tf.reduce_all([tf.reduce_min(receivers_) >= 0, tf.reduce_max(receivers_) <= num_nodes - 1]),
                            ["Invalid indexes in receiver indexes"])
        tf.debugging.Assert(tf.reduce_all([tf.reduce_min(senders_) >= 0, tf.reduce_max(senders_) <= num_nodes - 1]),
                            ["Invalid indexes in sender indexes"])

        return super(Graph, cls).__new__(cls, nodes_, edges_, globals_, receivers_, senders_, n_nodes_, n_edges_)

    @classmethod
    def from_networkx(cls, graph_nx, node_feature_shape=None, edge_feature_shape=None, globals_feature_shape=None):
        """Constructs an instance from a networkx graph.

        The networkx graph should be set up such that, for fixed shapes `node_shape`, `edge_shape` and `global_shape`:
        - `graph_nx.nodes(data=True)[i][-1]["features"]` is, for any node index i, a tensor of shape `node_shape`,
        - `graph_nx.edges(data=True)[i][-1]["features"]` is, for any edge index i, a tensor of shape `edge_shape`,
        - `graph_nx.graph["features"] is a tensor of shape `global_shape` or `None`.

        Note that both nodes and edges should present in the provided graph, or else error will be raised.
        Global feature are optional to present.

        Typical usage is:
        `graph = Graph.from_networkx(nx_graph)`
        By default features shape converted as is, but with keywords `node_feature_shape`, `edge_feature_shape`,
        `globals_feature_shape` it is possible to specify a particular shape for each type of features, depending on
        what you tensorflow model consume.

        Args:
        :param graph_nx: A networkx graph. Any graph is applicable.
        :param node_feature_shape: An iterable, specifying a node feature shape or `None`.
                                    Default is a shape specified by graph_nx.
        :param edge_feature_shape: An iterable, specifying an edge feature shape or `None`.
                                    Default is a shape specified by graph_nx.
        :param globals_feature_shape: An iterable, specifying a global feature shape or `None`.
                                    Default is a shape specified by graph_nx
        :returns: An instance of `Graph`.
        """
        if node_feature_shape is not None:
            node_feature_shape = list(node_feature_shape)
        if edge_feature_shape is not None:
            edge_feature_shape = list(edge_feature_shape)
        if globals_feature_shape is not None:
            globals_feature_shape = list(globals_feature_shape)

        try:
            num_nodes = graph_nx.number_of_nodes()
        except ValueError:
            raise TypeError("Argument `graph_nx` of wrong type {}".format(type(graph_nx)))
        if num_nodes <= 0:
            raise RuntimeError("Invalid number of nodes")

        try:
            nodes_data = [ data[GRAPH_NX_FEATURES_KEY]
                for node_i, (key, data) in enumerate(graph_nx.nodes(data=True))
                if _check_key(node_i, key) and data[GRAPH_NX_FEATURES_KEY] is not None
            ]
            if len(nodes_data) != num_nodes:
                raise ValueError("All the nodes should have features")
            nodes = tf.convert_to_tensor(nodes_data)
            if node_feature_shape is not None:
                nodes = tf.reshape(nodes,[num_nodes]+node_feature_shape)
        except KeyError:
            raise KeyError("Missing 'features' field from the graph nodes. This could be due to the node having been"
                            "silently added as a consequence of an edge addition when creating the networkx instance")

        num_edges = graph_nx.number_of_edges()
        if num_edges <= 0:
            raise RuntimeError("Invalid number of edges")

        if "index" in list(graph_nx.edges(data=True))[0][2]:
            senders, receivers, edge_attr_dicts = zip(*sorted(graph_nx.edges(data=True), key=lambda x: x[2]["index"]))
        else:
            senders, receivers, edge_attr_dicts = zip(*graph_nx.edges(data=True))

        senders = tf.convert_to_tensor(senders, dtype=tf.int32)
        receivers = tf.convert_to_tensor(receivers, dtype=tf.int32)
        edges_data = [ x[GRAPH_NX_FEATURES_KEY]
                       for x in edge_attr_dicts if x[GRAPH_NX_FEATURES_KEY] is not None]
        if len(edges_data) == 0:
            # edges exist, but do not have any features
            edges_data = np.zeros((num_edges,1),dtype=np.float32)

        if len(edges_data) != num_edges:
            raise ValueError("All the edges should have features")

        edges = tf.convert_to_tensor(edges_data)
        if edge_feature_shape is not None:
            edges = tf.reshape(edges, [num_edges] + edge_feature_shape)

        if GRAPH_NX_FEATURES_KEY in graph_nx.graph:
            globals = graph_nx.graph[GRAPH_NX_FEATURES_KEY]
        else:
            globals = tf.zeros([1,1],dtype=tf.float32)

        globals = tf.convert_to_tensor(globals)

        if globals_feature_shape is not None:
            globals = tf.reshape(globals,[1] + globals_feature_shape)

        if tf.shape(globals)[0] != 1 or tf.rank(globals) < 2:
            # unbatched global features case
            globals = tf.expand_dims(globals,0)

        return cls(nodes, edges, globals, receivers, senders, [num_nodes], [num_edges])

    def to_networkxs(self):
        """Returns a networkx graph that contains the stored data.

        The node and edge features are placed in the networkx graph's nodes and edges attribute dictionaries
        under the "features" key. Edges are added in the order they are stored in the `self.edges` tensor.
        The global features are placed under the key "features" of the the networkx graph's property.

        :returns: An instance of `networkx.OrderedMultiDiGraph`.

        """
        output_graphs_nx = []
        for g in self.batch_iterator():
            graph_nx = nx.OrderedMultiDiGraph()
            graph_nx.graph[GRAPH_NX_FEATURES_KEY] = g.globals.numpy()[0]

            nodes_list = _unstack(g.nodes)
            for i, x in enumerate(nodes_list):
                graph_nx.add_node(i, **{GRAPH_NX_FEATURES_KEY: x})

            edges_features = [{ "index": i, GRAPH_NX_FEATURES_KEY: x } for i, x in enumerate(_unstack(g.edges))]
            edges_data = zip(g.senders.numpy(), g.receivers.numpy(), edges_features)
            graph_nx.add_edges_from(edges_data)

            output_graphs_nx.append(graph_nx)
        return output_graphs_nx

    def replace(self,**kwargs):
        return self._replace(**kwargs)

    def get_batch_size(self):
        """Returns a number of independent graphs being stored in the instance."""
        return tf.shape(self.n_nodes)[0]

    def get_num_nodes(self):
        """Returns a total number of nodes in all of the graphs being stored in the instance."""
        return tf.reduce_sum(self.n_nodes)

    def get_num_edges(self):
        """Returns a total number of edges in all of the graphs being stored in the instance."""
        return tf.reduce_sum(self.n_edges)

    def get_graph_by_index(self,i):
        """Returns a specified by index `i` graph from the batch.
        Raises error is `i` is less than zero or exceed a batch size.
        """
        tf.debugging.assert_greater_equal(i,0,message="Index shall be greater than zero")
        tf.debugging.assert_less(i,self.get_batch_size(),message="Index shall be less than total number of graphs")

        graph_slice = slice(i, i + 1)
        start_slice = slice(0, graph_slice.start)
        start_node_index = tf.reduce_sum(self.n_nodes[start_slice])
        start_edge_index = tf.reduce_sum(self.n_edges[start_slice])
        end_node_index = start_node_index + tf.reduce_sum(self.n_nodes[graph_slice])
        end_edge_index = start_edge_index + tf.reduce_sum(self.n_edges[graph_slice])
        nodes_slice = slice(start_node_index, end_node_index)
        edges_slice = slice(start_edge_index, end_edge_index)

        return Graph(nodes=self.nodes[nodes_slice],
                    edges=self.edges[edges_slice],
                    globals=self.globals[graph_slice],
                    receivers=self.receivers[edges_slice]-start_node_index,
                    senders=self.senders[edges_slice]-start_node_index,
                    n_nodes=self.n_nodes[graph_slice],
                    n_edges=self.n_edges[graph_slice])

    def batch_iterator(self):
        """Iterates over stacked graphs."""
        for i in range(self.get_batch_size()):
            yield self.get_graph_by_index(i)

    def get_tensor_spec(self):
        """Returns a `tensorflow.TensorSpec` for the graph instance.
        It is useful for specifying an input signature for the `tensorflow.function` wrapper.

        Example:
        @tf.function(input_signature=[graph_input.get_tensor_spec()])
        def graph_function(graph_input):
            ...
        """
        def get_spec(tensor):
            shape = tensor.shape.as_list()
            shape[0]=None
            return tf.TensorSpec(shape=shape,dtype=tensor.dtype)

        return Graph(nodes=get_spec(self.nodes),
                      edges=get_spec(self.edges),
                      globals=get_spec(self.globals),
                      receivers=get_spec(self.receivers),
                      senders=get_spec(self.senders),
                      n_nodes=get_spec(self.n_nodes),
                      n_edges=get_spec(self.n_edges)
                     )

    def broadcast_globals_to_edges(self):
        return broadcast_globals(self.globals,self.n_edges)

    def broadcast_globals_to_nodes(self):
        return broadcast_globals(self.globals,self.n_nodes)

    def broadcast_receiver_nodes_to_edges(self):
        return broadcast_nodes_to_edges(self.nodes,self.receivers)

    def broadcast_sender_nodes_to_edges(self):
        return broadcast_nodes_to_edges(self.nodes,self.senders)




def _check_key(node_index, key):
    """Checks that node ordinal key is equal to provided integer key and raise ValueError if not.
    """
    if node_index != key:
        raise ValueError(
            "Nodes of the networkx.OrderedMultiDiGraph must have sequential integer keys consistent with the order"
            "of the nodes (e.g. `list(graph_nx.nodes)[i] == i`), found node with index {} and key {}"
            .format(node_index, key))

    return True

def _unstack(x):
    """Similar to `tensorflow.unstack`.
    """
    num_splits = int(x.shape[0])
    return [np.squeeze(x, 0) for x in np.split(x, num_splits, axis=0)]
