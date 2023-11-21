import tensorflow as tf
from tensorflow import keras 
from keras import layers


class Network(tf.keras.Model):
    def __init__(self):
        super(Network,self).__init__()
        self.conv1 = layers.Conv2D(6,5,activation="relu")
        self.pool = layers.MaxPool2D()
        self.conv2 = layers.Conv2D(16,5,activation="relu")
        self.fc1 = layers.Dense(120,activation="relu")
        self.fc2 = layers.Dense(84,activation="relu")
        self.logits = layers.Dense(10,activation="softmax")
    
    def call(self,x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = layers.Flatten()(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.logits(x)
        return x





if __name__ == "__main__" :
    model = Network()
    x=tf.random.uniform((64,32,32,3))
    model(x)
    print(model.summary())

    model.compile(
        optimizer= tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["acc"]
    )
    

