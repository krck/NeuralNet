//  Neuron.h
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
#ifndef Neuron_h
#define Neuron_h

#include "Settings.h"


struct Connection {
    double weight;
    double deltaWeight;
    Connection() : weight(RAND_0to1), deltaWeight(0.0f) {}
};


class Neuron {
private:
    const ulong index;                          // Index of the Neuron in it's Layer
    double outputValue;                         // Value of the Neuron given to all Neurons in the next Layer
    double gradient;                            // used by the backpropagation
    std::vector<Connection> outputWeights;      // Output weight values for all connected Neurons
    
public:
    Neuron(ulong numOutputs, ulong ind) : index(ind), outputValue(0.0f), gradient(0.0f) {
        // Fully connected Net: One Connection for each Neuron in the next Layer
        this->outputWeights = std::vector<Connection>(numOutputs);
    }
    
    // Called for each Neuron to do the math (!!!11!)
    void feedForward(const Layer& prevLayer) {
        double sum = 0.0f;
        // Sum up all Outputs from the previous Layer (with the Bias Neuron)
        for(const Neuron& n : prevLayer) { sum += (n.getOutputValue() * n.getOutputWeight(this->index).weight); }
        // Apply (activation /) transfer Function to shape the output value
        this->outputValue = transferFunction(sum);
    }
    
    // (One of several ways to calculate the gradient)
    // While Training the Net: The gradient pushes Neuron outputs in
    // the direction that will reduce the overall error value
    void calculateOutputGradient(double targetValue) {
        const double delta = targetValue - this->outputValue;
        // Multiply the delta by the derivative of the output value
        this->gradient = delta * transferFunctionDerivative(this->outputValue);
    }
    
    // Basically the same as OutputGradient
    void calculateHiddenGradient(const Layer& next) {
        const double dow = sumDOW(next);
        this->gradient = dow * transferFunctionDerivative(this->outputValue);
    }
    
    void updateInputWeights(Layer& prev) {
        // The weights to be updated are in the Connection container
        // in the neurons of the previous Layer
        // (Update all weights, including Bias)
        for (ulong i = 0; i < prev.size(); i++) {
            Neuron& n = prev[i];    // for(Neuron n : prev) does not work ... (?!?)
            const double oldDeltaWeight = n.getOutputWeight(this->index).deltaWeight;
            // Individual input, magnified by the gradient and the learning rate (ETA n),
            // and also add momentum (alpha): a fraction of the previous delta weight
            const double newDeltaWeight = (ETA * n.getOutputValue() * this->gradient) + (ALPHA * oldDeltaWeight);
            n.getOutputWeight(this->index).deltaWeight = newDeltaWeight;
            n.getOutputWeight(this->index).weight += newDeltaWeight;
        }
    }
    
private:
    // Transfer Function in this case: Return hyperbolic tangent of the sum
    // tanh curve will have an output range from -1.0 to 1.0
    inline double transferFunction(double sum) const { return tanh(sum); }
    
    // Not the actual (but a approximated) hyperbolic tangent derivative is used here (!)
    inline double transferFunctionDerivative(double sum) const { return (1.0 - (sum * sum)); }
    
    double sumDOW(const Layer& next) const {
        double sum = 0.0f;
        // Sum up all contributions of the errors, to the Neurons in the next Layer
        for(ulong i = 0; i < next.size() - 1; ++i) { sum += outputWeights[i].weight * next[i].getGradient(); }
        return sum;
    }
    
public:
    // GETTER - SETTER
    inline void setOutputValue(double value) { this->outputValue = value; }
    inline double getOutputValue() const { return  this->outputValue; }
    inline double getGradient() const { return  this->gradient; }
    inline Connection& getOutputWeight(ulong i) { return this->outputWeights[i]; }
    inline Connection getOutputWeight(ulong i) const { return this->outputWeights[i]; }
    
};

#endif
