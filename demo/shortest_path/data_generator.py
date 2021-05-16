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
import collections
import itertools
import networkx as nx

from scipy import spatial
from graphnetlib import Graph, concat_graphs

DISTANCE_WEIGHT_NAME ="distance"



def set_diff(seq0, seq1):
    """Return the set difference between 2 sequences as a list."""
    return list(set(seq0) - set(seq1))

def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

def to_one_hot(indices, num_classes):
    return np.eye(num_classes)[indices]


def generate_connected_graph(generator,num_nodes,theta,rate):
    """Generates graph ensures all nodes are connected

    Args:
        generator: A numpy random generator
        num_nodes: An integer number of nodes in the graph
        theta: A `float` threshold parameters for the geographic threshold graph's threshold.
        rate: A rate parameter for the node weight exponential sampling distribution.

    Returns:
        combined_graph: A generated graph
    """
    pos_array = generator.uniform(size=(num_nodes, 2))
    pos = dict(enumerate(pos_array))
    weight = dict(enumerate(generator.exponential(rate, size=num_nodes)))
    geo_graph = nx.geographical_threshold_graph(num_nodes, theta, pos=pos, weight=weight)

    distances = spatial.distance.squareform(spatial.distance.pdist(pos_array))
    i_, j_ = np.meshgrid(range(num_nodes), range(num_nodes), indexing="ij")
    weighted_edges = list(zip(i_.ravel(), j_.ravel(), distances.ravel()))
    mst_graph = nx.Graph()
    mst_graph.add_weighted_edges_from(weighted_edges, weight=DISTANCE_WEIGHT_NAME)
    mst_graph = nx.minimum_spanning_tree(mst_graph, weight=DISTANCE_WEIGHT_NAME)
    # Put geo_graph's node attributes into the mst_graph.
    for i in mst_graph.nodes():
        mst_graph.nodes[i].update(geo_graph.nodes[i])

    # Compose the graphs.
    combined_graph = nx.compose_all([mst_graph, geo_graph.copy()])
    # Put all distance weights into edge attributes.
    for i, j in combined_graph.edges():
        combined_graph.get_edge_data(i, j).setdefault(DISTANCE_WEIGHT_NAME, distances[i, j])
    return combined_graph


def add_shortest_path(generator, graph, min_length=1):
    pair_to_length_dict = {}

    lengths = nx.all_pairs_shortest_path_length(graph)
    for x, yy in lengths:
        for y in yy:
            if yy[y] >= min_length:
                pair_to_length_dict[(x, y)] = yy[y]

    if len(pair_to_length_dict) == 0:
        raise ValueError("All shortest paths are below the minimum length")

    # Computes probabilities per pair, to enforce uniform sampling of each
    # shortest path lengths.
    # The counts of pairs per length.
    node_pairs = list(pair_to_length_dict.keys())
    counts = collections.Counter(pair_to_length_dict.values())
    prob_per_length = 1.0 / len(counts)
    probabilities = [ prob_per_length / counts[pair_to_length_dict[x]] for x in node_pairs ]

    i = generator.choice(len(node_pairs), p=probabilities)
    start, end = node_pairs[i]
    path = nx.shortest_path(graph, source=start, target=end, weight=DISTANCE_WEIGHT_NAME)

    digraph = graph.to_directed()

    # Add the "start", "end", and "solution" attributes to the nodes and edges.
    digraph.add_node(start, start=True)
    digraph.add_node(end, end=True)
    digraph.add_nodes_from(set_diff(digraph.nodes(), [start]), start=False)
    digraph.add_nodes_from(set_diff(digraph.nodes(), [end]), end=False)
    digraph.add_nodes_from(set_diff(digraph.nodes(), path), solution=False)
    digraph.add_nodes_from(path, solution=True)
    path_edges = list(pairwise(path))
    digraph.add_edges_from(set_diff(digraph.edges(), path_edges), solution=False)
    digraph.add_edges_from(path_edges, solution=True)
    return digraph


def graph_to_input_target(graph):
    """Returns 2 graphs with input and target feature vectors for training.

    Args:
        graph: An `nx.DiGraph` instance.

    Returns:
        The input `nx.DiGraph` instance.
        The target `nx.DiGraph` instance.

    Raises:
        ValueError: unknown node type
    """

    def create_feature(attr, fields):
        return np.hstack([np.array(attr[field], dtype=float) for field in fields])

    input_node_fields = ("pos", "weight", "start", "end")
    input_edge_fields = ("distance",)
    target_node_fields = ("solution",)
    target_edge_fields = ("solution",)

    input_graph = graph.copy()
    target_graph = graph.copy()

    solution_length = 0
    for node_index, node_feature in graph.nodes(data=True):
        input_graph.add_node(node_index, features=create_feature(node_feature, input_node_fields))
        target_node = to_one_hot(create_feature(node_feature, target_node_fields).astype(int), 2)[0]
        target_graph.add_node(node_index, features=target_node)
        solution_length += int(node_feature["solution"])

    for receiver, sender, features in graph.edges(data=True):
        input_graph.add_edge(
            sender, receiver, features=create_feature(features, input_edge_fields))
        target_edge = to_one_hot(
            create_feature(features, target_edge_fields).astype(int), 2)[0]
        target_graph.add_edge(sender, receiver, features=target_edge)

    input_graph.graph["features"] = np.array([0.0])
    target_graph.graph["features"] = np.array([solution_length], dtype=np.int32)

    return input_graph, target_graph

def generate_graphs(num_examples, num_nodes_min_max, theta=20.0, rate=1.0, min_path_length=1, seed=None):
    """Generate graphs for training and testing.

    Args:
        rand: A random seed (np.RandomState instance).
        num_examples: Total number of graphs to generate.
            num_nodes_min_max: A 2-tuple with the [lower, upper) number of nodes per
            graph. The number of nodes for a graph is uniformly sampled within this
            range.
        theta: (optional) A `float` threshold parameters for the geographic
            threshold graph's threshold. Default= the number of nodes.
        rate: (optional) A rate parameter for the node weight exponential sampling
            distribution. Default= 1.0.
        min_path_length: (optional) A minimal length for a generated target path.
            Paths shorter than the value will not be choosen. Default= 1.
        seed: (optional)

    Returns:
        input_graphs: The list of input graphs.
        target_graphs: The list of output graphs.
        graphs: The list of generated graphs.
    """
    if isinstance(seed,np.random.RandomState):
        gen = seed
    else:
        gen = np.random.RandomState(seed=seed)

    input_graphs = []
    target_graphs = []
    graphs = []
    for _ in range(num_examples):
        num_nodes = gen.randint(*num_nodes_min_max)
        graph = generate_connected_graph(gen,num_nodes,theta,rate)
        graph = add_shortest_path(gen, graph, min_path_length)
        input_graph, target_graph = graph_to_input_target(graph)

        input_graphs.append(Graph.from_networkx(input_graph))
        target_graphs.append(Graph.from_networkx(target_graph))
        graphs.append(graph)

    input_graphs = concat_graphs(input_graphs)
    target_graphs = concat_graphs(target_graphs)
    return input_graphs, target_graphs, graphs