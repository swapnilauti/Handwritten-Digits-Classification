"""
Implementation of a Multilayer Perceptron Neural Network and evaluation of its
performance in classifying handwritten digits.
Input Dataset : original dataset from MNIST. There are 10 matrices for testing set and 10
matrices for training set, which corresponding to 10 digits.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import matplotlib.pyplot as plt
import time
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
    
    """# Input: z can be a scalar, a vector or a matrix
    # Output: return the sigmoid of input z"""
    return (1 / (1 + np.exp(-1 * z)))
    
def feature_selection(data):
	mask = []
	min_list=np.amin(data,axis=0)
	max_list=np.amax(data,axis=0)
	for i in range(data.shape[1]):
		if min_list[i]==max_list[i]:
		    mask.append(i)
	return mask

def preprocess():
    """ This function loads the MNIST data set from file 'mnist_all.mat'.

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
	"""
    
    mat = loadmat('mnist_all.mat')

    train_data = np.array([])
    train_label = np.array([])
    validation_data = np.array([])
    validation_label = np.array([])
    test_data = np.array([])
    test_label = np.array([])

    for i in range(10):
        cur_data = mat['train'+str(i)]
        cur_label = np.repeat(i, mat['train'+str(i)].shape[0])
        np.random.shuffle(cur_data)
        cur_training_size = int(cur_data.shape[0]*83/100)
        train_data = np.append(train_data, cur_data[:cur_training_size])
        train_label = np.append(train_label, cur_label[:cur_training_size])
        validation_data = np.append(validation_data, cur_data[cur_training_size:])
        validation_label = np.append(validation_label, cur_label[cur_training_size:])

    train_data = np.reshape(train_data, (-1, 784))
    train_data /= 255
    validation_data = np.reshape(validation_data, (-1, 784))
    validation_data /= 255

    mask_list=feature_selection(train_data)
    train_data=np.delete(train_data,mask_list,axis=1)
    validation_data=np.delete(validation_data,mask_list,axis=1)

    for i in range(10):
        test_data = np.append(test_data, mat['test'+str(i)])
        test_label = np.append(test_label, np.repeat(i, mat['test'+str(i)].shape[0]))
    test_data = np.reshape(test_data, (-1, 784))
    test_data /= 255

    test_data=np.delete(test_data,mask_list,axis=1)
    return train_data, train_label, validation_data, validation_label, test_data, test_label

    

def nnObjFunction(params, *args):
    """
	% nnObjFunction computes the value of objective function (negative log 
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
    """
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args
    
    w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0

    num_data = training_data.shape[0]
    training_label_K = np.zeros((training_data.shape[0], n_class))
    count = 0
    for label in training_label:
        training_label_K[count][int(label)] = 1
        count += 1
    #FeedForward

    biased_training_data = np.concatenate((training_data, np.tile([1], (training_data.shape[0], 1))), 1)
    z_Matrix = sigmoid(np.dot(biased_training_data, np.transpose(w1)))

    #baised value to Z Matrix
    z_baised_Matrix = np.concatenate((z_Matrix, np.tile([1], (z_Matrix.shape[0], 1))), 1)

    o_Matrix = sigmoid(np.dot(z_baised_Matrix,np.transpose(w2)))

    #Feedforward Ends

    #BackPropagation with regularization
    gradient_w1 = np.zeros((n_hidden + 1, n_input + 1))
    gradient_w2 = np.zeros((n_class, n_hidden + 1))

    delta_L = np.subtract(training_label_K, o_Matrix) * np.subtract(1, o_Matrix) * o_Matrix

    gradient_w2 = -1 * np.add(gradient_w2, np.dot(np.transpose(delta_L), z_baised_Matrix))

    gradient_w1 = -1 * np.add(gradient_w1, np.dot(np.transpose(np.dot(delta_L, w2) * np.subtract(1, z_baised_Matrix) * z_baised_Matrix), biased_training_data))

    obj_val = np.sum((np.sum(np.square(np.subtract(training_label_K, o_Matrix)), axis=1) / 2)) / num_data 

    gradient_w1 = gradient_w1[:-1, :]

    #regularization

    obj_val = obj_val + (lambdaval * (np.sum(np.square(w1)) + np.sum(np.square(w2))))/(2 * num_data)

    gradient_w1 = (gradient_w1 + (lambdaval * w1)) / num_data 

    gradient_w2 = (gradient_w2 + (lambdaval * w2)) / num_data 


    #gradient_w2 = gradient_w2[:, :-1]

    obj_grad = np.concatenate((gradient_w1.flatten(), gradient_w2.flatten()), 0)
    print(obj_val)
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

    biased_data = np.concatenate((data, np.tile([1], (data.shape[0], 1))), 1)
    z_Matrix = sigmoid(np.dot(biased_data, np.transpose(w1)))

    #add baised value to Z Matrix
    z_baised_Matrix = np.concatenate((z_Matrix, np.tile([1], (z_Matrix.shape[0], 1))), 1)

    o_Matrix = sigmoid(np.dot(z_baised_Matrix, np.transpose(w2)))
    labels = np.argmax(o_Matrix, axis=1)
    return labels
    



file = open('output.txt', 'w')


train_data, train_label, validation_data,validation_label, test_data, test_label = preprocess();


#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1];

#hidden_val_list = [4, 8, 12, 16, 20]
#lambda_list = [0, .2, .4, .6, .8, 1]
n_hidden = 8
lambdaval = .4
# set the number of nodes in hidden unit (not including bias unit)
# set the number of nodes in output unit
n_class = 10;
# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);
# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)
# set the regularization hyper-parameter
# lambdaval = 0.4;
args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)
opts = {'maxiter': 100}  # Preferred value.
time_stamp=time.time()
nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
file.write('\n Time Taken for Minimize function : '+str(time.time()-time_stamp))

w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
# Test the computed parameters
predicted_label = nnPredict(w1, w2, train_data)
file.write('\n Lambda : ' + str(lambdaval))
file.write('\n Hidden Node : ' + str(n_hidden))
# find the accuracy on Training Dataset
file.write('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')
predicted_label = nnPredict(w1, w2, validation_data)
# find the accuracy on Validation Dataset
file.write('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')
predicted_label = nnPredict(w1, w2, test_data)
# find the accuracy on Validation Dataset
file.write('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')
params_val={'n_hidden':n_hidden,'w1':w1,'w2':w2,'lambdaval':lambdaval}
pickle.dump(params_val,open("params.pickle","wb"))
