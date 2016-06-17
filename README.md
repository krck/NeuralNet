
<h1> Neural Network to recognize handwritten digits </h1><br>

<b> About the Neural Network </b> <br>
This is a simple, fully connected, feed forward <br>
and backpropagation Neural Network written in C++ <br>

<b> About MNIST: </b> <br>
http://yann.lecun.com/exdb/mnist/ <br>
Yann LeCun (Courant Institute, NYU), Corinna Cortes (Google Labs, New York) <br>
and Christopher J.C. Burges (Microsoft Research, Redmond) <br>
<br>
The MNIST database of handwritten digits, has a training set of 60,000 examples, <br>
and a test set of 10,000 examples. It is a subset of a larger set available from NIST. <br>
The digits have been size-normalized and centered in a fixed-size image (28x28) <br>
<br>
It is a good database for people who want to try learning techniques <br>
and pattern recognition methods on real-world data while spending minimal <br> 
efforts on preprocessing and formatting. <br>

<b> TODO </b>
- [x] Fix the neural net
- [x] Add test Function and test Export
- [ ] Add Network Export & Import
- [ ] Add batch learning (choose between batch and stochastic approach)
- [ ] Add multi-threading
- [x] Enable multiple training iterations
- [x] Cleaned up main() Function
- [ ] Further cleanup and optimization


<b> Project source: </b>

1. basicApp.cpp: 
    * "main" function, calling "train" and "test"
2. Settings.h:
    * Set the path to the MNIST files
    * Set the neural network topology (Layers / Neurons)
    * Set training iterations and training speed (ETA / ALPHA)
3. NeuralNet.h:
    * Calling feed forward, with the training data, for each layer in the net
    * Backpropagate to get the networks error and call the gradient calculation for each layer
    * Calling feed forward, with the testing data, to create an test-results output file
4. Layer.h:
    * Passing the input values to the input layer neurons
    * Forward propagating the input values throug each neuron (calculating the sigmoid curve)
    * Calculating the gradients and weights of each neuron, when backpropagating
    * Returning the output values of the output layer neurons
    * Returning the overall network error based on the output layer neurons
5. Neuron.h:
    * Neuron struct holding its output value and gradient
    * Neuron struct holding all the weights for each connected neuron in the next layer
6. MNIST.h
    * Parsing MNIST files to structs that hold the pixel values, labels and expected outputs
