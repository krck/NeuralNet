//  NeuralNet.h
/*************************************************************************************
 *  Neural Network to process handwritten digits form the MNIST dataset              *
 *-----------------------------------------------------------------------------------*
 *  Copyright (c) 2016, Peter Baumann                                                *
 *  All rights reserved.                                                             *
 *                                                                                   *
 *  Redistribution and use in source and binary forms, with or without               *
 *  modification, are permitted provided that the following conditions are met:      *
 *    1. Redistributions of source code must retain the above copyright              *
 *       notice, this list of conditions and the following disclaimer.               *
 *    2. Redistributions in binary form must reproduce the above copyright           *
 *       notice, this list of conditions and the following disclaimer in the         *
 *       documentation and/or other materials provided with the distribution.        *
 *    3. Neither the name of the organization nor the                                *
 *       names of its contributors may be used to endorse or promote products        *
 *       derived from this software without specific prior written permission.       *
 *                                                                                   *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND  *
 *  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED    *
 *  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE           *
 *  DISCLAIMED. IN NO EVENT SHALL PETER BAUMANN BE LIABLE FOR ANY                    *
 *  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES       *
 *  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;     *
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND      *
 *  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT       *
 *  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS    *
 *  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                     *
 *                                                                                   *
 *************************************************************************************/

#pragma once

#include "MNIST.h"
#include "Layer.h"

class NeuralNetOOP {
private:
    double netError;
    double recentAverageError;
    std::vector<Layer> layers;
    
public:
    NeuralNetOOP(const Topology& topology) : netError(0.0), recentAverageError(0.0) {
        // Network needs at least 2 Layers (1 Input & 1 Output)
        if(topology.size() >= 2) {
            // Create every Layer in the Net (1 Input, X Hidden, 1 Output)
            this->layers.push_back(Layer(topology[0], topology[1], LayerType::Input));
            for(ulong i = 1; i < topology.size() - 1; i++) {
                this->layers.push_back(Layer(topology[i], topology[i + 1], LayerType::Hidden));
            }
            this->layers.push_back(Layer(topology.back(), 0, LayerType::Output));
        } else {
            std::cout <<"ERROR: Trying to create a Network wiht less than 2 Layers" <<std::endl;
        }
    }
    
    
    // Train the NeuralNet by feeding forward all the input data
    // and then backpropagating with the according output data
    void train(const MNIST& mnist) {
        for(int i = 0; i < TRAINING_ITER; i++) {
            for(const MNISTchar& mc : mnist.trainingData) {
                feedForward(mc.pixelData);
                backPropagate(mc.output);
            }
        }
    }

    
private:
    void feedForward(const std::vector<double>& inputValues) {
        // Pass the input values to the input Layer
        this->layers.front().setInputValues(inputValues);
        // Loop throug each (hidden and output) Layer to call "feedForward"
        for(ulong i = 1; i < this->layers.size(); i++) { this->layers[i].feedForward(this->layers[i-1]); }
    }
    
    
    void backPropagate(const std::vector<double>& expOutputs) {
        // Get the overall net error and calculate the recent average measurement
        this->netError = this->layers.back().getError(expOutputs);
        this->recentAverageError = (recentAverageError * SMOOTHING_FACTOR + netError) / (SMOOTHING_FACTOR + 1.0);
        // Gradients: While training the net, gradients will push the Neuron outputs
        // in a direction that will reduce the overall error value
        // Calculate output layer gradients
        this->layers.back().calculateGradients(expOutputs);
        // Calcualte hidden layer gradients
        // (Loop backwards from the penultimate Layer to the second layer ... through all hidden Layers)
        for (ulong i = (this->layers.size() - 2); i > 0; i--) { this->layers[i].calculateGradients(this->layers[i+1]); }
        // Update the connection weights
        // (Loop from the output Layer backwards to the first hidden layer / Input Layer has no in coming weights)
        for (ulong i = (layers.size() - 1); i > 0; i--) { this->layers[i].updateWeights(this->layers[i-1]); }
    }
    
    
public:
    void exportNeuralNet(const std::string& exportPath) const { }
    void importNeuralNet(const std::string& importPath) { }
    
    
    // Feed the test data to the net and write all results to a file
    void test(MNIST& mnist, const std::string& resultsPath) {
        double errSum = 0.0f;
        int recognizeCount = 0;
        std::vector<std::string> outputStrings = std::vector<std::string>();
        // Add some general output with the overall error values
        outputStrings.push_back("Overall Network Error:\t\t");
        outputStrings.push_back("Correctly recognised digits:\t");
        outputStrings.push_back("\n\nTest Data Digits:\n");
        // Add some output for ever Digit in the testData
        for(const auto& t : mnist.testData) {
            outputStrings.push_back("----------------------------------");
            // feed forward the testData
            feedForward(t.pixelData);
            // get the Neural Nets results
            errSum += this->netError;
            const auto result = this->layers.back().getResults();
            // generate the current test digit as an ASCII picture
            auto asciiDigit = mnist.MNISTcharToASCII(t);
            outputStrings.insert(outputStrings.end(), asciiDigit.begin(), asciiDigit.end());
            // generate the exprected / actual results table
            outputStrings.push_back("\n");
            for(ulong i = 0; i < t.output.size(); i++) {
                outputStrings.push_back(std::to_string((int)t.output[i]) + "\t\t\t" + std::to_string((double)result[i]));
            }
            // generate the networks guess
            outputStrings.push_back("\n");
            ulong num = -1, count = 0;
            std::string tmp = "";
            // get the Number with the highest possibility ( >= 0.8 )
            // and check if the other are as low as expected ( <= 0.2 )
            for(ulong i = 0; i < result.size(); i++) {
                if(result[i] >= 0.8f) { num = i; }
                if(result[i] <= 0.2f) { count++; }
            }
            tmp += (count >= 9) ? " definitely a: " : " very likely a: ";
            tmp += "\t" + std::to_string(num);
            outputStrings.push_back("This is" + tmp);
            // Check if the estimated digit is correct
            if(num == t.label) {
                recognizeCount++;
                outputStrings.push_back("Network guessed:\tCORRECT");
            } else {
                outputStrings.push_back("Network guessed:\tWRONG");
            }
            outputStrings.push_back("----------------------------------");
        }
        // Calculate the overall Error
        errSum = (errSum / mnist.testData.size());
        outputStrings[0] += std::to_string(errSum);
        // Calculate percentage of correctly recognized digits
        double percent = ((double)recognizeCount / mnist.testData.size()) * 100.0f;
        outputStrings[1] += (std::to_string((int)percent) + "%  (" + std::to_string(recognizeCount)
                                    + " / " + std::to_string(mnist.testData.size()) + ")");
        // WRITE ALL THE TEST OUTPUTS TO A FILE
        std::fstream file (resultsPath, std::ifstream::out | std::ifstream::binary);
        if (file.is_open()) { for(const auto& line : outputStrings) { file << line + "\n"; } }
        file.close();
    }
    
};
