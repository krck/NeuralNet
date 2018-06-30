//  NeuralNetVec.h
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

#include "NetMath.h"

class NeuralNetVec {
    
private:
    const Topology _layers;
    const size_t _layerCount, _lastLayer, _hiddenLayerCount;
    const float _learningRate;
    std::vector<Vector> _neuronVectors;         // H
    std::vector<Matrix> _weights;               // W
    std::vector<Vector> _biases;                // B
    std::vector<Matrix> _weightDeltas;          // dEdW
    std::vector<Vector> _biasDeltas;            // dEdB
    
public:
    NeuralNetVec(const Topology& layers, float learningRate)
    : _layers(layers), _layerCount(layers.size()), _lastLayer(_layerCount - 1), _hiddenLayerCount(_layerCount - 2), _learningRate(learningRate) {
        _neuronVectors = std::vector<Vector>(_layerCount);        // Each Layer has a neuron vector (Input/Hidden/Output)
        _weights = std::vector<Matrix>(_lastLayer);                // Each Layer holds the weights for the next layers neurons (-1 for Output)
        _biases = std::vector<Vector>(_lastLayer);                // Each Layer holds the biases for the next layers neurons (-1 for Output)
        _weightDeltas = std::vector<Matrix>(_lastLayer);
        _biasDeltas = std::vector<Vector>(_lastLayer);
        
        // Initialize all weight matrices and bias vectors with random values
        for (size_t i = 0; i < _lastLayer; i++) {
            // Weight matrix (rows = nextLayerNeurons, columns = thisLayerNeurons)
            _weights[i] = Matrix(_layers[i + 1] * _layers[i]);
            for (size_t mc = 0; mc < _weights[i].size(); mc++) {
                _weights[i][mc] = random_0_1;
            }
            
            // Bias vectors (length = nextLayerNeutrons)
            _biases[i] = Vector(_layers[i + 1]);
            for (size_t bc = 0; bc < _biases[i].size(); bc++) {
                _biases[i][bc] = random_0_1;
            }
        }
    }
    
    
    void Train(size_t iterations, const std::vector<Vector>& trainingInput, const std::vector<Vector>& trainingOutput) {
        for (size_t i = 0; i < iterations; i++) {
            for (size_t t = 0; t < trainingInput.size(); t++) {
                FeedForward(trainingInput[t]);
                BackPropagate(trainingOutput[t]);
            }
        }
    }
    
    
    // Take the net input and return the net output
    Vector FeedForward(const Vector& input) {
        _neuronVectors[0] = input; // Set the input layer == the input
        
        for (size_t i = 1; i < _layerCount; i++) {
            _neuronVectors[i] = CalculateDotSigmoid(_weights[i - 1], _neuronVectors[i - 1], _biases[i - 1], _layers[i]);
        }
        
        return _neuronVectors.back();
    }
    
    
    void BackPropagate(const Vector& expectedOutput) {
        // Calculate Error here (MSE) ... (Not needed)
        
        // Calculate the bias gradients
        
        // tmp = H[hiddenLayersCount].dot(W[hiddenLayersCount]).add(B[hiddenLayersCount]).applyFunction(sigmoidePrime)
        // dEdB[hiddenLayersCount] = H[hiddenLayersCount + 1].subtract(_neuronVectors.back()).multiply(tmp)
        Vector tmp = CalculateDotSigmoidPrime(_weights[_hiddenLayerCount], _neuronVectors[_hiddenLayerCount], _biases[_hiddenLayerCount], _layers[_lastLayer]);
        _biasDeltas[_hiddenLayerCount] = CalculateLastBiasDelta(_neuronVectors.back(), expectedOutput, tmp);
        
        for (long i = _hiddenLayerCount - 1; i >= 0; i--)
        {
            //dEdB[i] = dEdB[i + 1].dot(W[i + 1].transpose()).multiply(    H[i].dot(W[i]).add(B[i]).applyFunction(sigmoidePrime)   );
            const Vector sigPresult = CalculateDotSigmoidPrime(_weights[i], _neuronVectors[i], _biases[i], _layers[i + 1]);
            _biasDeltas[i] = CalculateBiasDelta(_biasDeltas[i + 1], _weights[i + 1], _layers[i + 2], _layers[i + 1], sigPresult);
        }
        
        // Calculate the weight gradients and update all weights and biases
        for (size_t i = 0; i < _lastLayer; i++) {
            // dEdW[i] = H[i].transpose().dot(dEdB[i])
            _weightDeltas[i] = CalculateWeightDelta(_neuronVectors[i], _biasDeltas[i]);
            
            // W[i] = W[i].subtract(dEdW[i].multiply(learningRate))
            // B[i] = B[i].subtract(dEdB[i].multiply(learningRate))
            _weights[i] = UpdateWeight(_weights[i], _weightDeltas[i], _learningRate);
            _biases[i] = UpdateBias(_biases[i], _biasDeltas[i], _learningRate);
        }
    }
    
    // Feed the test data to the net and write all results to a file
    void test(MNIST& mnist, const std::string& resultsPath) {
        float errSum = 0.0f;
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
            const auto result = FeedForward(t.pixelData);
            // generate the current test digit as an ASCII picture
            auto asciiDigit = mnist.MNISTcharToASCII(t);
            outputStrings.insert(outputStrings.end(), asciiDigit.begin(), asciiDigit.end());
            // generate the exprected / actual results table
            outputStrings.push_back("\n");
            for(ulong i = 0; i < t.output.size(); i++) {
                outputStrings.push_back(std::to_string((int)t.output[i]) + "\t\t\t" + std::to_string((float)result[i]));
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
        float percent = ((float)recognizeCount / mnist.testData.size()) * 100.0f;
        outputStrings[1] += (std::to_string((int)percent) + "%  (" + std::to_string(recognizeCount)
                             + " / " + std::to_string(mnist.testData.size()) + ")");
        // WRITE ALL THE TEST OUTPUTS TO A FILE
        std::fstream file (resultsPath, std::ifstream::out | std::ifstream::binary);
        if (file.is_open()) { for(const auto& line : outputStrings) { file << line + "\n"; } }
        file.close();
    }
    
};
