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

