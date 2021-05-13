import tensorflow as tf
import numpy as np
from graphnetlib import GraphNetwork,IndependentNetwork,NodeProcessor,EdgeProcessor,GlobalProcessor, concat_graphs
import matplotlib.pyplot as pyplot

from data_generator import generate_graphs
from visualize import GraphPlotter, plot_losses, plot_predicted_graphs
from model import EncodeProcessDecoder
from metrics import compute_accuracy



NUM_TRAIN_ITERATIONS = 10000
NUM_PROCESSING_STEPS = 10
NUM_NODES_TRAIN_RANGE = (8, 17)
NUM_NODES_TEST_RANGE = (16, 33)
NUM_TRAIN_EXAMPLES = 32
NUM_TEST_EXAMPLES = 100
NODE_SAMPLING_RATE = 1.0
MIN_SOLUTION_PATH_LENGTH = 1
THETA = 20
LEARNING_RATE = 1e-3
LOG_ITERATION = 10
SEED = 42


generator = np.random.RandomState(seed=SEED)

### Test data generation
test_inputs,test_targets,test_raw_graphs = generate_graphs(NUM_TEST_EXAMPLES,NUM_NODES_TEST_RANGE,THETA,
                                                           NODE_SAMPLING_RATE,MIN_SOLUTION_PATH_LENGTH,generator)
###

### Plot examples of test data
fig = pyplot.figure(figsize=(15, 15))
fig.suptitle("Examples of Test Data",fontsize=27,fontweight="bold")
for j, graph in enumerate(test_raw_graphs[:min(NUM_TEST_EXAMPLES,16)]):
    plotter = GraphPlotter(graph)
    ax = fig.add_subplot(4, 4, j + 1)
    plotter.draw_graph_with_solution(ax)
fig.tight_layout()
fig.savefig("data_example.png",dpi=250)
pyplot.close(fig)
###

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


def loss_function(out_gs,target_gs):
    return tf.stack([
        tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=target_gs.nodes,logits=og.nodes),axis=0) +
        tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=target_gs.edges,logits=og.edges),axis=0)
        for og in out_gs
    ])

def get_test_metrics():
    out_gs = model(test_inputs)
    loss = tf.reduce_mean(loss_function(out_gs, test_targets))
    p, s = compute_accuracy(out_gs[-1], test_targets)
    return loss, p, s

@tf.function(input_signature=[test_inputs.get_tensor_spec(),
                              test_targets.get_tensor_spec()])
def train_step(graph,target):
    with tf.GradientTape() as tape:
        out_gs = model(graph)
        loss = tf.reduce_mean(loss_function(out_gs, target))
    grads = tape.gradient(loss, model.trainable_variables)
    return loss, grads, out_gs

optimizer = tf.optimizers.Adam(LEARNING_RATE)

train_losses = []
test_losses = []
train_fraction_predicted = []
test_fraction_predicted = []
train_fraction_solved = []
test_fraction_solved = []

### Training
for it in range(NUM_TRAIN_ITERATIONS):
    train_inputs, train_targets, _ = generate_graphs(NUM_TEST_EXAMPLES, NUM_NODES_TEST_RANGE, THETA,
                                                              NODE_SAMPLING_RATE, MIN_SOLUTION_PATH_LENGTH, generator)

    loss, grads, outs = train_step(train_inputs,train_targets)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    if it % LOG_ITERATION == 0:
        p, s = compute_accuracy(outs[-1], train_targets)
        loss_, p_, s_ = get_test_metrics()
        train_losses.append(loss.numpy())
        test_losses.append(loss_.numpy())
        train_fraction_predicted.append(p)
        test_fraction_predicted.append(p_)
        train_fraction_solved.append(s)
        test_fraction_solved.append(s_)
        print("Iter: {} | Train loss: {} | Train predict: {} | Train solved: {} |\n\tTest loss: {} | Test predicted: {} | Test solved: {}".format(
                it, loss.numpy(), p, s, loss_.numpy(), p_, s_))


p_inp = concat_graphs([test_inputs.get_graph_by_index(i) for i in range(10)])
p_target = concat_graphs([test_targets.get_graph_by_index(i) for i in range(10)])
p_raw = test_raw_graphs[:10]

out_g = model(p_inp)
plot_predicted_graphs(out_g,p_target,p_raw,savefile="prediciton.png")
plot_losses(train_losses,test_losses,train_fraction_predicted,test_fraction_predicted,train_fraction_solved,test_fraction_solved,
                    title="Training metrics",savefile="metrics.png")