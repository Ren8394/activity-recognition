import tensorflow as tf

class MLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.denseIn = tf.keras.layers.Dense(units=100)
        self.dense = tf.keras.layers.Dense(units=100, activation=tf.nn.relu)
        self.denseOut = tf.keras.layers.Dense(units = 7) # Output class = 6 + 1(other)

    def call(self, inputs):
        x = self.denseIn(inputs)
        x = self.dense(x)
        x = self.dense(x)
        x = self.denseOut(x)
        outputs = tf.nn.softmax(x)
        
        return outputs
