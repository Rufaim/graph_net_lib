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
from graphnetlib import concat_graphs, GraphNetwork, \
                        IndependentNetwork, NodeProcessor, EdgeProcessor,GlobalProcessor

from model import EncodeProcessDecoder
from data_generator import base_graph, alter_data
from simulator import SpringMassSimulator, GraphNetSimulator, simulate
from visualize import visualize_run



NUM_TRAIN_ITERATION = 100000
NUM_PROCESSING_STEPS = 10
NUM_TIMESTEPS = 300
TIMESTEP_SIZE = 0.01
LEARNING_RATE = 1e-3
NUM_TRAIN_SAMPLES = 256
NUM_TEST_SAMPLES = 100
NUM_MASSES_RANGE = (5,9)
DISTANCE_BETWEEN_MASSES_RANGE = (0.1,0.2)
NODE_NOISE = 1.0
EDGE_NOISE = 10.0
GLOBAL_NOISE = 1.0
LOG_ITERATION = 100
SEED = 42


tf_generator = tf.random.Generator.from_seed(SEED)
sim = SpringMassSimulator(TIMESTEP_SIZE)

def generate_train_data():
    num_masses_train = tf_generator.uniform((NUM_TRAIN_SAMPLES,), NUM_MASSES_RANGE[0], NUM_MASSES_RANGE[1],dtype=tf.int32)
    dist_between_masses_train = tf_generator.uniform((NUM_TRAIN_SAMPLES,), DISTANCE_BETWEEN_MASSES_RANGE[0],
                                                     DISTANCE_BETWEEN_MASSES_RANGE[1], dtype=tf.float32)
    train_data = [alter_data(base_graph(n, d), NODE_NOISE, EDGE_NOISE, GLOBAL_NOISE, tf_generator)
                  for n, d in zip(num_masses_train, dist_between_masses_train)]
    train_data = concat_graphs(train_data)
    train_data_node_rollouts = simulate(sim, train_data, NUM_TIMESTEPS)
    return train_data, train_data_node_rollouts

### Test data generation
# Initital train data
train_data, train_data_node_rollouts = generate_train_data()

# Test data
test_data_4 = [base_graph(4, 0.5)] * NUM_TEST_SAMPLES
test_data_4 = [alter_data(g,NODE_NOISE, EDGE_NOISE, 0.0, tf_generator) for g in test_data_4]
test_data_4 = concat_graphs(test_data_4)
test_data_4_rollout = simulate(sim,test_data_4,NUM_TIMESTEPS)

test_data_9 = [base_graph(9, 0.5)] * NUM_TEST_SAMPLES
test_data_9 = [alter_data(g,NODE_NOISE, EDGE_NOISE, 0.0, tf_generator) for g in test_data_9]
test_data_9 = concat_graphs(test_data_9)
test_data_9_rollout = simulate(sim,test_data_9,NUM_TIMESTEPS)
###

### Model
kernel_init = tf.keras.initializers.glorot_normal(seed=SEED)

encoder = IndependentNetwork(node_net=tf.keras.Sequential([
    tf.keras.layers.Dense(16,activation=tf.nn.relu,kernel_initializer=kernel_init),
    tf.keras.layers.Dense(32,activation=tf.nn.relu,kernel_initializer=kernel_init),
    tf.keras.layers.LayerNormalization()
    ]),
    edge_net=tf.keras.Sequential([
    tf.keras.layers.Dense(16,activation=tf.nn.relu,kernel_initializer=kernel_init),
    tf.keras.layers.Dense(32,activation=tf.nn.relu,kernel_initializer=kernel_init),
    tf.keras.layers.LayerNormalization()]))
decoder = IndependentNetwork(node_net=tf.keras.Sequential([
    tf.keras.layers.Dense(32,activation=tf.nn.relu,kernel_initializer=kernel_init),
    tf.keras.layers.Dense(16,activation=tf.nn.relu,kernel_initializer=kernel_init),
    tf.keras.layers.Dense(2,kernel_initializer=kernel_init),
    ]))

np = NodeProcessor(tf.keras.Sequential([
    tf.keras.layers.Dense(32,activation=tf.nn.relu,kernel_initializer=kernel_init),
    tf.keras.layers.Dense(32,activation=tf.nn.relu,kernel_initializer=kernel_init),
    tf.keras.layers.LayerNormalization()
]))
ep = EdgeProcessor(tf.keras.Sequential([
    tf.keras.layers.Dense(32,activation=tf.nn.relu,kernel_initializer=kernel_init),
    tf.keras.layers.Dense(32,activation=tf.nn.relu,kernel_initializer=kernel_init),
    tf.keras.layers.LayerNormalization()
]))
gp = GlobalProcessor(tf.keras.Sequential([
    tf.keras.layers.Dense(2, activation=tf.nn.relu, kernel_initializer=kernel_init),
    tf.keras.layers.Dense(2, activation=tf.nn.relu, kernel_initializer=kernel_init),
    tf.keras.layers.LayerNormalization()
]))
core = GraphNetwork(node_processor=np,edge_processor=ep,global_processor=gp)
model = EncodeProcessDecoder(encoder=encoder,core=core,decoder=decoder,num_steps=NUM_PROCESSING_STEPS)
###


def loss_function(out,target):
    l = tf.reduce_sum((out - target[..., 2:4])**2, axis=-1)
    return tf.reduce_mean(l)

@tf.function(input_signature=[train_data.get_tensor_spec(),
            tf.TensorSpec(shape=[None]+train_data_node_rollouts[0].shape.as_list()[1:], dtype=train_data_node_rollouts[0].dtype)])
def train_step(inps,targets):
    with tf.GradientTape() as tape:
        outs = model(inps)
        loss = tf.reduce_mean([loss_function(o.nodes,targets) for o in outs])

    grads = tape.gradient(loss,model.trainable_variables)
    optimizer.apply_gradients(zip(grads,model.trainable_variables))
    return loss


graph_sim = GraphNetSimulator(model,TIMESTEP_SIZE)

optimizer = tf.optimizers.Adam(LEARNING_RATE)
for iter in range(NUM_TRAIN_ITERATION):
    time = tf_generator.uniform([],minval=0,maxval=NUM_TIMESTEPS-1,dtype=tf.int32)

    step_data = train_data.replace(nodes=train_data_node_rollouts[time])
    step_targets = train_data_node_rollouts[time+1]

    loss = train_step(step_data,step_targets)

    if iter % LOG_ITERATION == 0:
        predicted_rollout_4 = simulate(graph_sim,test_data_4,NUM_TIMESTEPS)
        loss_4 = loss_function(predicted_rollout_4[..., 2:4], test_data_4_rollout)

        predicted_rollout_9 = simulate(graph_sim, test_data_9,NUM_TIMESTEPS)
        loss_9 = loss_function(predicted_rollout_9[..., 2:4], test_data_9_rollout)

        print("Iter {}| Loss train: {:.5}| Loss size 4: {:.5}| Loss size 9: {:.5}".format(iter,loss.numpy(),loss_4.numpy(),loss_9.numpy()))

    if (iter + 1) % (NUM_TIMESTEPS*10) == 0:
        train_data, train_data_node_rollouts = generate_train_data()


td_4 = test_data_4.get_graph_by_index(0)
predicted_4_rollout = simulate(graph_sim,td_4,500)
true_4_rollout = simulate(sim,td_4,500)
visualize_run(true_4_rollout.numpy(),rollout_predicted=predicted_4_rollout.numpy(),title="Test 4 nodes",video_filename="test_4.gif")

td_9 = test_data_9.get_graph_by_index(0)
predicted_9_rollout = simulate(graph_sim,td_9,500)
true_9_rollout = simulate(sim,td_9,500)
visualize_run(true_9_rollout.numpy(),rollout_predicted=predicted_9_rollout.numpy(),title="Test 9 nodes",video_filename="test_9.gif")
