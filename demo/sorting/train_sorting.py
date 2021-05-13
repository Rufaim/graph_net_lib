import tensorflow as tf
from graphnetlib import GraphNetwork,IndependentNetwork,NodeProcessor,EdgeProcessor,GlobalProcessor

from model import EncodeProcessDecoder
from data_generator import create_data
from visualize import plot_graph_edges, plot_ranked_inputs, plot_losses
from metrics import compute_accuracy



NUM_TRAIN_ITERATIONS = 10000
NUM_PROCESSING_STEPS = 10
NUM_TRAIN_EXAMPLES = 32
NUM_TEST_EXAMPLES = 100
NUM_ELEMENTS_TRAIN_RANGE = (8,17)
NUM_ELEMENTS_TEST_RANGE = (16, 33)
LEARNING_RATE = 1e-3
LOG_ITERATION = 10
SEED = 42


def sample_batch(n,rng,tf_generator):
    num_elements = tf_generator.uniform((n,), rng[0], rng[1], dtype=tf.int32)
    elements = [tf_generator.uniform(shape=[num]) for num in num_elements]
    return create_data(elements)


tf_generator = tf.random.Generator.from_seed(SEED)

### Model
kernel_init = tf.keras.initializers.glorot_normal(seed=SEED)

encoder = IndependentNetwork(node_net=tf.keras.Sequential([
    tf.keras.layers.Dense(16,activation=tf.nn.relu,kernel_initializer=kernel_init),
    tf.keras.layers.Dense(16,activation=tf.nn.relu,kernel_initializer=kernel_init),
    tf.keras.layers.LayerNormalization()
    ]),
    edge_net=tf.keras.Sequential([
    tf.keras.layers.Dense(16,activation=tf.nn.relu,kernel_initializer=kernel_init),
    tf.keras.layers.Dense(16,activation=tf.nn.relu,kernel_initializer=kernel_init),
    tf.keras.layers.LayerNormalization()]),
    global_net=tf.keras.Sequential([
    tf.keras.layers.Dense(16,activation=tf.nn.relu,kernel_initializer=kernel_init),
    tf.keras.layers.LayerNormalization()
    ]))
decoder = IndependentNetwork(node_net=tf.keras.Sequential([
    tf.keras.layers.Dense(16,activation=tf.nn.relu,kernel_initializer=kernel_init),
    tf.keras.layers.Dense(16,activation=tf.nn.relu,kernel_initializer=kernel_init),
    tf.keras.layers.Dense(2,kernel_initializer=kernel_init),
    ]),
    edge_net=tf.keras.Sequential([
    tf.keras.layers.Dense(16,activation=tf.nn.relu,kernel_initializer=kernel_init),
    tf.keras.layers.Dense(16,activation=tf.nn.relu,kernel_initializer=kernel_init),
    tf.keras.layers.Dense(2,kernel_initializer=kernel_init),
]))

np = NodeProcessor(tf.keras.Sequential([
    tf.keras.layers.Dense(16,activation=tf.nn.relu,kernel_initializer=kernel_init),
    tf.keras.layers.Dense(16,activation=tf.nn.relu,kernel_initializer=kernel_init),
    tf.keras.layers.LayerNormalization()
]))
ep = EdgeProcessor(tf.keras.Sequential([
    tf.keras.layers.Dense(16,activation=tf.nn.relu,kernel_initializer=kernel_init),
    tf.keras.layers.Dense(16,activation=tf.nn.relu,kernel_initializer=kernel_init),
    tf.keras.layers.LayerNormalization()
]))
gp = GlobalProcessor(tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation=tf.nn.relu, kernel_initializer=kernel_init),
    tf.keras.layers.Dense(16, activation=tf.nn.relu, kernel_initializer=kernel_init),
    tf.keras.layers.LayerNormalization()
]))
core = GraphNetwork(node_processor=np,edge_processor=ep,global_processor=gp)
model = EncodeProcessDecoder(encoder=encoder,core=core,decoder=decoder,num_steps=NUM_PROCESSING_STEPS)
###


### Test data generation
test_graph, test_target_nodes, test_target_edges, test_sort_indexes, test_ranks = sample_batch(NUM_TEST_EXAMPLES,NUM_ELEMENTS_TEST_RANGE,tf_generator)
###

def loss_function(out_gs,target_nodes,target_edges):
    return tf.stack([
        tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=target_nodes,logits=og.nodes),axis=0) +
        tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=target_edges,logits=og.edges),axis=0)
        for og in out_gs
    ])

def get_test_metrics():
    out_gs = model(test_graph)
    loss = tf.reduce_mean(loss_function(out_gs, test_target_nodes, test_target_edges))
    p, s = compute_accuracy(out_gs[-1], test_target_nodes, test_target_edges)
    return loss, p, s

@tf.function(input_signature=[test_graph.get_tensor_spec(),
                              tf.TensorSpec(shape=[None,2], dtype=tf.float32),
                              tf.TensorSpec(shape=[None,2], dtype=tf.float32)])
def train_step(graph,t_nodes,t_edges):
    with tf.GradientTape() as tape:
        out_gs = model(graph)
        loss = tf.reduce_mean(loss_function(out_gs, t_nodes, t_edges))
    grads = tape.gradient(loss, model.trainable_variables)
    return loss, grads, out_gs

optimizer = tf.optimizers.Adam(LEARNING_RATE)

train_losses = []
test_losses = []
train_fraction_predicted = []
test_fraction_predicted = []
train_fraction_solved = []
test_fraction_solved = []

for it in range(NUM_TRAIN_ITERATIONS):
    # Sample train data
    batch_graphs, batch_target_nodes, batch_target_edges,_,_ = sample_batch(NUM_TRAIN_EXAMPLES,NUM_ELEMENTS_TRAIN_RANGE,tf_generator)

    loss, grads, outs = train_step(batch_graphs,batch_target_nodes,batch_target_edges)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    if it % LOG_ITERATION == 0:
        p, s = compute_accuracy(outs[-1], batch_target_nodes, batch_target_edges)
        loss_,p_,s_ = get_test_metrics()
        train_losses.append(loss.numpy())
        test_losses.append(loss_.numpy())
        train_fraction_predicted.append(p)
        test_fraction_predicted.append(p_)
        train_fraction_solved.append(s)
        test_fraction_solved.append(s_)
        print("Iter: {} | Train loss: {} | Train predict: {} | Train solved: {} |\n\tTest loss: {} | Test predicted: {} | Test solved: {}".format(it,
                                loss.numpy(),p, s, loss_.numpy(),p_,s_))


tg = test_graph.get_graph_by_index(0)

out_g = model(tg)[-1]
plot_graph_edges(tg,out_g.edges.numpy()[:,0],test_sort_indexes[0].numpy(),
                 title="Element-to-element links for sorted elements",savefile="graphs_predicted.png")
plot_graph_edges(tg,test_target_edges.numpy()[:tg.get_num_edges(),0],test_sort_indexes[0].numpy(),
                 title="Element-to-element links for sorted elements",savefile="graphs_true.png")
plot_ranked_inputs(tg.nodes.numpy()[:,0],test_sort_indexes[0].numpy(),test_ranks[0].numpy(),
                   title="Nodes ranks",savefile="elements.png")
plot_losses(train_losses,test_losses,train_fraction_predicted,test_fraction_predicted,train_fraction_solved,test_fraction_solved,
                    title="Training metrics",savefile="metrics.png")
