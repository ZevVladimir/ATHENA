import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import time

import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from data_and_loading_functions import standardize, build_ml_dataset

curr_snapshot = "190"
curr_hdf5_file = "sparta_190.hdf5"

data_location = "/home/zvladimi/MLOIS/calculated_info/" + "calc_from_" + curr_hdf5_file + "/"
save_location = "/home/zvladimi/MLOIS/training_data/" + "data_for_" + curr_hdf5_file + "/"

np.random.seed(11)

num_splits = 20
param_list = ['Orbit_Infall', 'Scaled_radii', 'Radial_vel', 'Tangential_vel']
with h5py.File((data_location + "all_particle_properties" + curr_snapshot + ".hdf5"), 'r') as all_particle_properties:
    total_num_particles = all_particle_properties["PIDS"][:].shape[0]    
    random_indices = np.random.choice(total_num_particles, total_num_particles)
    use_num_particles = int(np.floor(total_num_particles/num_splits))

for i in range(1):
    t1 = time.time()
    print("Split:", (i+1), "/",num_splits)
    
    dataset = build_ml_dataset(save_location, data_location, i, random_indices, use_num_particles, curr_snapshot, param_list)

    t2 = time.time()
    print("Loaded data", t2 - t1, "seconds")

    X_train, X_test, y_train, y_test = train_test_split(dataset[:,1:], dataset[:,0], test_size=0.30, random_state=0)


Xtrain = torch.from_numpy(X_train)
ytrain = torch.from_numpy(y_train).type(torch.CharTensor)

Xtest = torch.from_numpy(X_test)
ytest = torch.from_numpy(y_test).type(torch.CharTensor)

batch_size = 100
n_iters = 10000
num_epochs = n_iters / (len(X_train) / batch_size)
num_epochs = int(num_epochs)

train = TensorDataset(Xtrain, ytrain)
test = TensorDataset(Xtest, ytest)

train_loader = DataLoader(train, batch_size = batch_size, shuffle = False)
test_loader = DataLoader(test, batch_size = batch_size, shuffle = False)

class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(RNNModel, self).__init__()
        
        # Number of hidden dimensions
        self.hidden_dim = hidden_dim
        
        # Number of hidden layers
        self.layer_dim = layer_dim
        
        # RNN
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='relu')
        
        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        
        # Initialize hidden state with zeros
        h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))
            
        # One time step
        out, hn = self.rnn(x, h0)
        out = self.fc(out[:, -1, :]) 
        return out

input_dim = 7
hidden_dim = 100
layer_dim = 1
output_dim = 1

model = RNNModel(input_dim, hidden_dim, layer_dim, output_dim)

error = nn.BCEWithLogitsLoss
learning_rate = 0.01

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

loss_list = []
iteration_list = []
accuracy_list = []
count = 0

batch_size = 100
n_iters = 8000
num_epochs = n_iters / ((X_train.shape[1]) / batch_size)
num_epochs = int(num_epochs)

train = TensorDataset(Xtrain,ytrain)
test = TensorDataset(Xtest,ytest)

# data loader
train_loader = DataLoader(train, batch_size = batch_size, shuffle = False)
test_loader = DataLoader(test, batch_size = batch_size, shuffle = False)

for epoch in range(num_epochs):
    for i, (halos, labels) in enumerate(train_loader):

        train  = Variable(halos)
        labels = Variable(labels)
            
        # Clear gradients
        optimizer.zero_grad()
        
        # Forward propagation
        outputs = model(train)
        
        # Calculate softmax and ross entropy loss
        loss = error(outputs, labels)
        
        # Calculating gradients
        loss.backward()
        
        # Update parameters
        optimizer.step()
        
        count += 1
        
        if count % 250 == 0:
            # Calculate Accuracy         
            correct = 0
            total = 0
            # Iterate through test dataset
            for halos, labels in test_loader:
                halos = Variable(halos)
                
                # Forward propagation
                outputs = model(halos)
                
                # Get predictions from the maximum value
                predicted = torch.max(outputs.data, 1)[1]
                
                # Total number of labels
                total += labels.size(0)
                
                correct += (predicted == labels).sum()
            
            accuracy = 100 * correct / float(total)
            
            # store loss and iteration
            loss_list.append(loss.data)
            iteration_list.append(count)
            accuracy_list.append(accuracy)
            if count % 500 == 0:
                # Print Loss
                print('Iteration: {}  Loss: {}  Accuracy: {} %'.format(count, loss.data[0], accuracy))

plt.plot(iteration_list,loss_list)
plt.xlabel("Number of iteration")
plt.ylabel("Loss")
plt.title("RNN: Loss vs Number of iteration")
plt.show()

# visualization accuracy 
plt.plot(iteration_list,accuracy_list,color = "red")
plt.xlabel("Number of iteration")
plt.ylabel("Accuracy")
plt.title("RNN: Accuracy vs Number of iteration")
plt.savefig('graph.png')
plt.show()