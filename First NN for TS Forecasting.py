# -*- coding: utf-8 -*-
"""
Created on Sun May  8 18:10:53 2016

@author: Sara
"""
import pandas as pd
from pybrain import FeedForwardNetwork
#from pybrain.tools.shortcuts     import buildNetwork
from pybrain import LinearLayer, TanhLayer#, SigmoidLayer
from pybrain import FullConnection
import numpy as np
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
import matplotlib.pylab as plt


def set_nn(inp,hid1,out):
    # Make a new FFN object:
    n = FeedForwardNetwork()
    
    # Constructing the input, output and hidden layers:
    inLayer = LinearLayer(inp)
    hiddenLayer1 = TanhLayer(hid1)
#    hiddenLayer2 = TanhLayer(hid2)
    outLayer = LinearLayer(out)
    
    # Adding layers to the network:
    n.addInputModule(inLayer)
    n.addModule(hiddenLayer1)
#    n.addModule(hiddenLayer2)
    n.addOutputModule(outLayer)
    
    # determining how neurons should be connected:
    in_to_hidden = FullConnection(inLayer, hiddenLayer1)
#    hid_to_hid = FullConnection(hiddenLayer1,hiddenLayer2)
    hidden_to_out = FullConnection(hiddenLayer1, outLayer)
    
    # Adding connections to the network
    n.addConnection(in_to_hidden)
#    n.addConnection(hid_to_hid)
    n.addConnection(hidden_to_out)
    
    # Final step that makes our MLP usable
    n.sortModules()
    return n
        

def set_input_data(timeseries):
    # Create samples Train & Validation
    training_samples = []
        
    for i in range((len(timeseries)-4)*2/3):
        training_samples.append(timeseries[i:i+4])
    np.random.shuffle(training_samples)
    
    # Normalize
    max_sample = np.max(timeseries)
    training_samples = np.array(training_samples)/float(max_sample)
    
    #logarithm
    training_samples = np.log(training_samples)
    
    inputs_training = training_samples[:, 0:3]
    target_training = training_samples[:, 3]
    target_training = target_training.reshape(-1, 1)
    return inputs_training, target_training


def set_output_data(timeseries): 
    validation_samples = []
    for i in range((len(timeseries)-4)*2/3, len(timeseries)-4, 1):
        validation_samples.append(timeseries[i:i+4])
    
    # Normalize  
    max_sample = np.max(timeseries)
    validation_samples = np.array(validation_samples)/float(max_sample)
    
    #logarithm
    validation_samples = np.log(validation_samples)
    
    inputs_validation = validation_samples[:, 0:3]
    target_validation = validation_samples[:, 3]
    target_validation = target_validation.reshape(-1, 1)
    return inputs_validation, target_validation


def train_with_shuffle(inp, targ, nn, epoch):
    
    
    ds = SupervisedDataSet(len(inp), len(targ))
    ds.setField('input', inp)
    ds.setField('target', targ)
    
    trainer = BackpropTrainer(nn, ds)
    trainer.trainUntilConvergence(verbose=True, validationProportion=0.15, maxEpochs=100, continueEpochs=10)

    for i in range(epoch):
        mse = trainer.train()
        rmse = np.sqrt(mse)
        print "training RSME, epoch {}: {}".format(i+1, rmse)
    return nn



if __name__ == '__main__':
        
    dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m')
    data = pd.read_csv('AirPassengers.csv', parse_dates=['Month'], index_col='Month', date_parser=dateparse)
    ts = data['#Passengers']    
    
    n = set_nn(3,4,1)
#    n = buildNetwork(3, 5, 4, 1, bias=True, hiddenclass=TanhLayer)
    inp_train,targ_train = set_input_data(ts)
    inp_valid, targ_valid = set_output_data(ts)
    train_with_shuffle(inp_train,targ_train,n,700)
    
#==============================================================================
#     ds = SupervisedDataSet(len(inp_train), len(targ_train))
#     ds.setField('input', inp_train)
#     ds.setField('target', targ_train)
#     
#     trainer = BackpropTrainer(n, ds)
#     trainer.trainUntilConvergence(verbose=True, validationProportion=0.15, maxEpochs=100, continueEpochs=10)
# 
#     for i in range(700):
#         mse = trainer.train()
#         rmse = np.sqrt(mse)
#         print "training RSME, epoch {}: {}".format(i+1, rmse)
#==============================================================================
    
    #==============================================================================
    # trainer = BackpropTrainer(n, ds, learningrate=0.01, momentum=0.1)
    # 
    # for epoch in range(1, 100000000):
    #     if epoch % 10000000 == 0:
    #         error = trainer.train()
    #         print 'Epoch: ', epoch
    #         print 'Error: ', error
    #==============================================================================
    train = np.array([n.activate(x) for x in inp_train])
    
    plt.subplot(131)
    plt.plot(train, color='blue', label='train')
    plt.plot(targ_train, color='red', label='target')
    plt.legend(loc='best')   
   
    predict = np.array([n.activate(x) for x in inp_valid])
    
    plt.subplot(132)
    plt.plot(predict, color='blue', label='predict')
    plt.plot(targ_valid, color='red', label='target')
    plt.legend(loc='best')
    
    samples = []
    for i in range(len(ts)-3):
        samples.append(ts[i:i+3])
    # Normalize
    max_sample = np.max(ts)
    samples = np.array(samples)/float(max_sample)
    normal_ts = np.array(ts)/float(max_sample)

    predict_final = np.array([n.activate(x) for x in samples])
    
    plt.subplot(133)
    plt.plot(predict_final, color='blue', label='predict')
    plt.plot(normal_ts, color='red', label='target')
    plt.legend(loc='best')
    
    print 'True'
