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
// ----------------------------------------------------------------------------
// -------------- Neuron.h and NeuralNet.h currently based on the -------------
// --------- tutorial by: David Miller, http://millermattson.com/dave ---------
// --- See the associated video for instructions: http://vimeo.com/19569529 ---
// ----------------------------------------------------------------------------

#pragma once
#ifndef NeuralNet_h
#define NeuralNet_h

#include "Settings.h"
#include "Neuron.h"
#include "MNIST.h"

class NeuralNet {
private:
    double netError;
    double recentAverageError;
    std::vector<Layer> layers;
    
public:
    NeuralNet(const Topology& topology) : netError(0.0), recentAverageError(0.0) {
        for (ulong i = 0; i < topology.size(); ++i) {
            this->layers.push_back(Layer());
            // Get the number of Neurons in the next layer (If next is the Output Layer, then 0)
            const ulong numOutputs = (i == topology.size() - 1 ? 0 : topology[i + 1]);
            // Add all the Neuros to each layer (+ 1 to add the Bias Neuron in every Layer)
            for (ulong j = 0; j < (topology[i] + 1); ++j) {
                this->layers.back().push_back( Neuron(numOutputs, j) );
            }
            // Set the Bias Neurons output value to 1.0 (Last Neuron of each Layer)
            this->layers.back().back().setOutputValue(1.0f);
        }
    }
    
    void feedForward(const std::vector<double>& inputValues) {
        // check if there is one input Value for each input Neuron (- Bias)
        if(inputValues.size() == (this->layers[0].size() - 1)) {
            // Assign all input values to input Neurons
            for (ulong i = 0; i < inputValues.size(); ++i) {
                this->layers[0][i].setOutputValue(inputValues[i]);
            }
            // Forward Propagate:
            // Loop throug each Layer (and each Neuron of the Layer) and call "feedForward"
            // (Start at 1 because the Input-Layer values are assigned already)
            for(ulong i = 1; i < this->layers.size(); ++i) {
                Layer& previousLayer = this->layers[i-1];
                for(ulong j = 0; j < this->layers[i].size() - 1; ++j) {
                    this->layers[i][j].feedForward(previousLayer);
                }
            }
        }
    }
    
    void backPropagate(const std::vector<double>& expOutputs) {
        // Calculate the overall net error
        // RMS (Root Mean Square Error) of all output neuron errors
        // Value going to be minimized throug the back propagation (hopefully ...)
        this->netError = 0.0;
        Layer& outputLayer = this->layers.back();
        // Add up all the deltas between actual output and expected output
        // ( - 1 because Output-Layer Bias Neuron does not count)
        for(ulong i = 0; i < outputLayer.size() - 1; ++i) {
            const double delta = expOutputs[i] - outputLayer[i].getOutputValue();
            // netError stores the sum of the squares of all output value deltas
            this->netError += delta * delta;
        }
        // Get the average (squared) error delta
        this->netError /= (outputLayer.size() - 1);
        // Take the square root, to get the RMS
        this->netError = sqrt(netError);
        // Implement a recent average measurement
        this->recentAverageError = (recentAverageError * SMOOTHING_FACTOR + netError) / (SMOOTHING_FACTOR + 1.0);
        // Gradients
        // Calculate output layer gradients
        for(ulong i = 0; i < (outputLayer.size() - 1); ++i) {
            outputLayer[i].calculateOutputGradient(expOutputs[i]);
        }
        // Calcualte hidden layer gradients
        // (Loop backwards from the penultimate Layer to the second layer ... through all hidden Layers)
        for (ulong i = (this->layers.size() - 2); i > 0; --i) {
            Layer& hidden = this->layers[i];
            Layer& next = this->layers[i+1];
            for (ulong n = 0; n < hidden.size(); ++n) { hidden[n].calculateHiddenGradient(next); }
        }
        // Update the connection weights
        // (Loop from the output Layer backwards to the first hidden layer / Input Layer has no weights coming in)
        for (ulong i = (layers.size() - 1); i > 0; --i) {
            Layer& currLayer = layers[i];
            Layer& prevLayer = layers[i - 1];
            for (unsigned n = 0; n < (currLayer.size() - 1); ++n) {
                currLayer[n].updateInputWeights(prevLayer);
            }
        }
    }
    
    // Get all the Output Layer Neurons Values
    std::vector<double> getResults() const {
        std::vector<double> tmpres;
        const Layer& outputLayer = this->layers.back();
        // Exclude last neuron (bias neuron)
        for (ulong i = 0; i < outputLayer.size() - 1; ++i) {
            tmpres.push_back(outputLayer[i].getOutputValue());
        }
        return tmpres;
    }
    
    // GETTER - SETTER
    inline double getNetError(void) const { return this->netError; }
    inline double getRecentAverageError(void) const { return this->recentAverageError; }
    
};

#endif
