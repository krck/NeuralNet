//  Neuron.h
/*************************************************************************
 * Neural Network to process handwritten digits form the MNIST dataset
 *------------------------------------------------------------------------
 * Copyright (c) 2016 Peter Baumann
 *
 * This software is provided 'as-is', without any express or implied
 * warranty. In no event will the authors be held liable for any damages
 * arising from the use of this software.
 *
 * Permission is granted to anyone to use this software for any purpose,
 * including commercial applications, and to alter it and redistribute it
 * freely, subject to the following restrictions:
 *
 * 1. The origin of this software must not be misrepresented; you must not
 *    claim that you wrote the original software. If you use this software
 *    in a product, an acknowledgment in the product documentation would
 *    be appreciated but is not required.
 * 2. Altered source versions must be plainly marked as such, and must not
 *    be misrepresented as being the original software.
 * 3. This notice may not be removed or altered from any source
 *    distribution.
 *
 *************************************************************************/
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
