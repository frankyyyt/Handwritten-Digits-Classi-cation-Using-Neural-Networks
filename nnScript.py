import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import math
import os
import sys
import pickle

def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W
    
    
    
def sigmoid(z):
    
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    # // Notice : cause the dataset is very huge, 
    # //you may find the warning RuntimeWarning: overflow encountered in exp
    res=1.0/(1.0+np.exp(-z))
    return res    
    
    

def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - divide the original data set to training, validation and testing set
           with corresponding labels
     - convert original data set from integer to double by using double()
           function
     - normalize the data to [0, 1]
     - feature selection"""
    
    mat = loadmat('mnist_all.mat') #loads the MAT object as a Dictionary
    
    #Pick a reasonable size for validation data
    validation_data = np.zeros([0,784],dtype=np.double)
    validation_label = np.zeros((0,1), dtype=np.uint8)

    test_label = np.zeros((0,1), dtype=np.uint8)
    test_data = np.zeros((0, 784), dtype = np.double)

    train_label = np.zeros((0,1), dtype=np.uint8)
    train_data = np.zeros([0, 784], dtype = np.double)
    train=train_data
    # stack  all the 60000 train data into train matrixs and partition  it into two parts
    for i in range(10):
        A = mat.get('train'+str(i))
        
        train=np.vstack((train,A))
        train=np.double(train)
    # print train_label.shape
    # I want to break it into two random parts, one matrix with 1000 rows and second with the rest

        a = range(A.shape[0])
        aperm = np.random.permutation(a)
        A1 = A[aperm[0:1000],:]
        A2 = A[aperm[1000:],:]
    
        # len=A1.shape[0]
        
        # create the validation label for each i
        validationl=np.full([A1.shape[0],1],i,dtype=int)
        validation_label=np.vstack((validation_label,validationl))
        validation_data=np.vstack((validation_data,A1))
        
        #create the train label for each i
        trainl=np.full([A2.shape[0],1],i,dtype=int)
        train_label=np.vstack((train_label,trainl))
        train_data= np.vstack((train_data,A2))      
    # print validation_data.shape
    # print train_label.shape

# ///////////////////////////
# ////////////////////////
# I want to break it into two random parts, one matrix with 10000 rows and second with the rest
    # a = range(train.shape[0])
    # aperm = np.random.permutation(a)
    # A1 = train[aperm[0:10000],:]
    # A2 = train[aperm[10000:],:]
    # validation_data=np.vstack((validation_data,A1))
    # train_data= np.vstack((train_data,A2))  
# /////////////////////////////////////    # 
    # normalized the element of the matrices
    train_data = train_data/ 255 
    validation_data = validation_data/ 255
    # pass the test after the normalization
    # print validation_data.shape
    # print train_data[1,:]   
    
    for i in range(10):
        A = mat.get('test'+str(i))
        test_data=np.vstack((test_data,A))

        testl=np.full([A.shape[0],1],i,dtype=int)
        test_label=np.vstack((test_label,testl))
    
    return train_data, train_label, validation_data, validation_label, test_data, test_label
    
    
    

def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""
    
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args
    
    w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0  
    
    #Your code here
    #
    #
    #
    #
    #
    # CALCULATE THE error using matrix     
    train_size = train_data.shape[0]
    trainlabel=np.ones((train_size,1),np.float16)

    train= np.hstack((train_data,trainlabel))

    print 'layerone'
    layerone = np.dot(train,w1.T) 
    print layerone.shape
    
    mid = sigmoid(layerone)
    print 'layertwo'
    layertwo = np.hstack(((sigmoid(layerone)),np.ones((train_size,1),np.float16)))
    
    outlabel = sigmoid(np.dot(layertwo,w2.T))
    
    # initialize the target label value
    tv = np.zeros((train_size,n_class))
    
    # convert the target label to vector 
    for i in range (0,train_label.shape[0]):
       tv.ravel()[i*n_class+train_label.item(i,0)] = 1
    # print "target label ",tv
    
    # for implement the feature selection, just eliminate the unchanged data

    # calculation the error every time update the gradient
    J = -np.mean(np.sum(tv*np.log(outlabel)+(1-tv)*np.log(1-outlabel)))
    #using Regularization lambdaval in Neural Network
    obj_val = J + (np.sum(w1*w1)+np.sum(w2*w2))*lambdaval/(2*train_size)
    # print obj_val

    grad_w1 = (np.dot(((np.dot((outlabel-tv),w2[:,0:(n_hidden)]))*(1-mid)*mid).T,train)+lambdaval*w1)/train_size
    grad_w2 = (np.dot(layertwo.T,(outlabel-tv)).T+lambdaval*(w2*w2))/train_size  
    
    #Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    #you would use code similar to the one below to create a flat array
    #obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)

    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    
    # print obj_grad
    
    return (obj_val,obj_grad)
    



def nnPredict(w1,w2,data):
    
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels""" 
    
    # labels: there are many ways to predict the label 
    # this is an another way to predict the result labels which differe from 
    labels=np.zeros((0,1),dtype=np.uint8)
    tem=np.zeros((0,n_hidden), dtype=np.float)

    for i in range(data.shape[0]): 
        a=np.dot(w1[:,:n_input],data[i,:])    
        tem=np.vstack((tem,a))       
    z=sigmoid(tem)

    for i in range(z.shape[0]):
        b=(np.dot(w2[:,:n_hidden],z[i,:]))
        lab=sigmoid(b).tolist()
        max_value =np.max(lab)
        max_index = lab.index(max_value)
        labels=np.vstack((labels,max_index))

    return labels
    



"""**************Neural Network Script Starts here********************************"""

train_data, train_label, validation_data,validation_label, test_data, test_label = preprocess();


#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]; 

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 40;
				   
# set the number of nodes in output unit
n_class = 10;				   

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)

# set the regularization hyper-parameter
lambdaval = 0.8;


args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter' : 100}    # Preferred value 50.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)

#In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
#and nnObjGradient. Check documentation for this function before you proceed.
#nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))


#Test the computed parameters

predicted_label = nnPredict(w1,w2,train_data)

#find the accuracy on Training Dataset

print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1,w2,validation_data)

#find the accuracy on Validation Dataset

print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')


predicted_label = nnPredict(w1,w2,test_data)

#find the accuracy on Validation Dataset

print('\n Test set Accuracy:' + str(100*np.mean((predicted_label == test_label).astype(float))) + '%')

# ///using the params.pickle to help optimize the performance 
parameter = (n_hidden, w1, w2, lambdaval)
file_name='params.pickle'
with open(file_name, 'wb') as params:
    #     pickle.Pickler(params).dump(parameter)
    pickle.dump(parameter,params)
    

