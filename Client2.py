import numpy as np
import flwr as fl
import sys
import tensorflow as tf 
from Network import Network 
import matplotlib.pyplot as plt


def getDist(y):
    x = [i for i in range(10)]
    plt.bar(x,y)
    plt.show()

def dist_data(dist,x,y):
    dx = []
    dy = []
    count = [0 for _ in range(10)]

    for i in range(len(x)):
        if count[y[i]] < dist[y[i]] :
            dx.append(x[i])
            dy.append(y[i])
            count[y[i]] += 1 
    
    return np.array(dx) , np.array(dy)

(x_train,y_train) , (x_test,y_test) = tf.keras.datasets.mnist.load_data()

x_train, x_test = x_train[...,tf.newaxis]/255 , x_test[...,tf.newaxis]/255 

dist = [0, 10, 10, 10, 4000, 3000, 4000, 5000, 10, 4500]

x_train , y_train = dist_data(dist,x_train,y_train)


model = Network()

model.compile(
        optimizer= tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["acc"]
    )


class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config)  :
        return model.get_weights()
    def fit(self,parameters,config):
        model.set_weights(parameters)
        model.fit(
            x_train,
            y_train,
            epochs=1,
            validation_data=(x_test,y_test),
            )
        return model.get_weights(),len(x_train) , {}
    
    def evaluate(self, parameters,config):
        model.set_weights(parameters)
        loss,acc = model.evaluate(x_test,y_test,verbose = 0)
        return loss,len(x_test),{"accuracy":float(acc)}
    
client = FlowerClient()

fl.client.start_numpy_client(
    server_address="localhost:"+str(sys.argv[1]),
    client=client
)





