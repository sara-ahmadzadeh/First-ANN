# -*- coding: utf-8 -*-
"""
Created on Sun May  8 18:10:53 2016
@author: Sara
"""
import pandas as pd
from pybrain import FeedForwardNetwork
from pybrain import LinearLayer, SigmoidLayer
from pybrain import FullConnection
import numpy as np
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
import matplotlib.pylab as plt


dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m')
data = pd.read_csv('AirPassengers.csv', parse_dates=['Month'], index_col='Month', date_parser=dateparse)
ts = data['#Passengers']

# Make a new FFN object:
n = FeedForwardNetwork()

# Constructing the input, output and hidden layers:
inLayer = LinearLayer(3)
hiddenLayer = SigmoidLayer(4)
outLayer = LinearLayer(1)

# Adding layers to the network:
n.addInputModule(inLayer)
n.addModule(hiddenLayer)
n.addOutputModule(outLayer)

# determining how neurons should be connected:
in_to_hidden = FullConnection(inLayer, hiddenLayer)
hidden_to_out = FullConnection(hiddenLayer, outLayer)

# Adding connections to the network
n.addConnection(in_to_hidden)
n.addConnection(hidden_to_out)

# Final step that makes our MLP usable
n.sortModules()

# Create samples Train & Validation
training_samples = []
validation_samples = []

for i in range((len(ts)-4)*2/3):
    training_samples.append(ts[i:i+4])
np.random.shuffle(training_samples)

# Normalize
max_sample = np.max(ts)
training_samples = np.array(training_samples)/float(max_sample)

inputs_training = training_samples[:, 0:3]
target_training = training_samples[:, 3]
target_training = target_training.reshape(-1, 1)

for i in range((len(ts)-4)*2/3, len(ts)-4, 1):
    validation_samples.append(ts[i:i+4])

# Normalize    
validation_samples = np.array(validation_samples)/float(max_sample)

inputs_validation = validation_samples[:, 0:3]
target_validation = validation_samples[:, 3]
target_validation = target_validation.reshape(-1, 1)

ds = SupervisedDataSet(len(inputs_training), len(target_training))
ds.setField('input', inputs_training)
ds.setField('target', target_training)

trainer = BackpropTrainer(n, ds)
trainer.trainUntilConvergence(verbose=True, validationProportion=0.15, maxEpochs=100, continueEpochs=10)

for i in range(700):
    mse = trainer.train()
    rmse = np.sqrt(mse)
    print "training RSME, epoch {}: {}".format(i+1, rmse)

#==============================================================================
# trainer = BackpropTrainer(n, ds, learningrate=0.01, momentum=0.1)
# 
# for epoch in range(1, 100000000):
#     if epoch % 10000000 == 0:
#         error = trainer.train()
#         print 'Epoch: ', epoch
#         print 'Error: ', error
#==============================================================================

predict = np.array([n.activate(x) for x in inputs_validation])

plt.plot(predict, color='blue', label='predict')
plt.plot(target_validation, color='red', label='target')
plt.legend(loc='best')

print 'True'
