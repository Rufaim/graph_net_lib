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



class EncodeProcessDecoder(tf.keras.Model):
    def __init__(self,encoder,core,decoder,num_steps=1):
        super(EncodeProcessDecoder, self).__init__()
        self.encoder = encoder
        self.core = core
        self.decoder = decoder
        self.num_steps = num_steps

    def call(self, input_graph, training=None, mask=None):
        outs = []
        inp = self.encoder(input_graph)
        prev_latent = inp.replace(
                    nodes=tf.concat([inp.nodes,inp.nodes],axis=-1),
                    edges=tf.concat([inp.edges,inp.edges],axis=-1)
                    )
        for i in range(self.num_steps):
            latent = self.core(prev_latent)
            prev_latent = latent.replace(
                    nodes=tf.concat([inp.nodes,latent.nodes],axis=-1),
                    edges=tf.concat([inp.edges,latent.edges],axis=-1)
                    )
            outs.append(self.decoder(prev_latent))
        return outs

