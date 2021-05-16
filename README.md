# GraphNetLib

GraphNetLib is a library for making graph networks with Tensorflow 2.x.

### Graph network

Graph networks operates on graphs.
They are fed with a graph and output a graph as well.
A graph is a structure having node (*V*), edge(*E*) and global(*u*) features.
To learn more about graph networks, see paper: [Relational inductive biases, deep learning, and graph networks](https://arxiv.org/abs/1806.01261).

![Graph network](https://github.com/Rufaim/graph_net_lib/raw/master/imgs/graph_nets.png)

## Installation

GraphNetLib can be installed using `pip`. This installation is compatible with Linux/Mac OS X, and Python 3.6+.

Please note that package is not currently registered in [PyPI](https://pypi.org/) and require installation from this repository directly.

```shell
python3 -m pip install git+https://github.com/Rufaim/graph_net_lib
```

The package supports both  CPU and GPU versions of Tensorflow.

## Examples

Let's consider a small usage example of creating a graph data and processing it with a graph network.

```python
import tensorflow as tf
import graphnetlib as gnl

# generate your node features
nodes_ = get_graph_nodes()
# generate edge features.
# receivers and senders are 1d tensors of integers representing indexes of
# corresponding outputting and receiving nodes.
edges_, senders_, receivers_ = get_graph_edges()
# generate global features
globals_ = get_graph_features()
# create the Graph structure
graph_data = gnl.Graph(nodes_,
                 edges_,
                 globals_,
                 receivers_,
                 senders_,
                 tf.shape(nodes_)[:1], # number of nodes in each graph in the batch. In the example we consider only one graph 
                 tf.shape(edges_)[:1]) # number of edges in each graph in the batch. In the example we consider only one graph

# Create the graph network
np = gnl.NodeProcessor(tf.keras.Sequential([
    tf.keras.layers.Dense(32,activation=tf.nn.relu),
    tf.keras.layers.Dense(32,activation=tf.nn.relu),
    tf.keras.layers.LayerNormalization()
]))
# Setup of all processors is very explicit to give user more of control
ep = gnl.EdgeProcessor(tf.keras.Sequential([
    tf.keras.layers.Dense(32,activation=tf.nn.relu),
    tf.keras.layers.Dense(32,activation=tf.nn.relu),
    tf.keras.layers.LayerNormalization()
]))
gp = gnl.GlobalProcessor(tf.keras.Sequential([
    tf.keras.layers.Dense(32,activation=tf.nn.relu),
    tf.keras.layers.Dense(32,activation=tf.nn.relu),
    tf.keras.layers.LayerNormalization()
]))
graph_network = gnl.GraphNetwork(node_processor=np,
                                    edge_processor=ep,
                                    global_processor=gp)


# Process the graph with the graph network
processed_graph_data = graph_network(graph_data)
```

### Demo

The repository includes three [demo examples](https://github.com/Rufaim/graph_net_lib/tree/master/demo) of how to use the package.
Those demos are similar to [DeepMind's Graph Nets library](https://github.com/deepmind/graph_nets/tree/a265c907be646abc6521ddc162cb61b9b8bacfb0/graph_nets/demos)

##### [Shortest path demo](https://github.com/Rufaim/graph_net_lib/tree/master/demo/shortest_path)

The "shortest path demo" shows how to train graph network to label nodes and edges on the shortest path between two nodes.
Data are generated randomly, but it is always ensured that final graph is connected.

![shortest path generated data](https://github.com/Rufaim/graph_net_lib/raw/master/imgs/shortest_path/data_example.png)

Over a sequence of message-passing steps, the model refines its prediction of the shortest path.

![shortest path prediction](https://github.com/Rufaim/graph_net_lib/raw/master/imgs/shortest_path/prediciton.png)

##### [Physics demo](https://github.com/Rufaim/graph_net_lib/tree/master/demo/physics)

The "physics demo" predicts a physics of a randomly generated mass-spring systems.
A graph network is trained to predict the evolution of the system after a fixed timestep.
The network predicitons are fed to the network to rollout the whole dynamics of the system.

<img src="https://github.com/Rufaim/graph_net_lib/raw/master/imgs/physics_demo/test_9.gif" width="40%"> <img src="https://github.com/Rufaim/graph_net_lib/raw/master/imgs/physics_demo/test_4.gif" width="40%">

##### [Sorting demo](https://github.com/Rufaim/graph_net_lib)

The "sort demo" graph network is trained to sort a list of random numbers.

![sorting demo elements](https://github.com/Rufaim/graph_net_lib/raw/master/imgs/sorting_demo/elements.png)

The network is trained to classify edges if a sender node (columns in the figure) is standing before a receiver node (rows) in the sorted list.

True conntections | Predicted conntections
--- | ---
<img src="https://github.com/Rufaim/graph_net_lib/raw/master/imgs/sorting_demo/graphs_true.png" width="60%"> | <img src="https://github.com/Rufaim/graph_net_lib/raw/master/imgs/sorting_demo/graphs_predicted.png" width="60%">

## Copyrigth notice

This implementation is based on adapted code of the DeepMind's [Graph Nets library](https://github.com/deepmind/graph_nets).
All rights for the original implementation belong to DeepMind.
