//  Layer.h
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
#ifndef Layer_h
#define Layer_h

#include "Settings.h"
#include "Neuron.h"


enum class LayerType {Input, Hidden, Output};


class Layer {
private:
    const LayerType type;
    std::vector<Neuron> neurons;
    
public:
    Layer(ulong nCount, ulong nCountNext, LayerType ltype) : type(ltype) {
        // Add all the Neurons to the Layer
        for(ulong i = 0; i < nCount; i++) { neurons.push_back(Neuron(nCountNext, i)); }
        // Add one Bias Neuron to the Layer and set the output value to 1.0
        neurons.push_back(Neuron(nCountNext, neurons.size()));
        neurons.back().outputValue = 1.0f;
    }
    
    
    void setInputValues(const std::vector<double>& inputValues) {
        if(this->type == LayerType::Input) {
            // Check if there is one input Value for each input Neuron (- Bias)
            if(inputValues.size() == this->getNeuronCountNoBias()) {
                // Assign all InputValues to the InputNeurons
                for(ulong i = 0; i < inputValues.size(); i++) { this->neurons[i].outputValue = inputValues[i]; }
            } else { std::cout <<"ERROR: Ammount of InputValues not equal to InputNeurons" <<std::endl; }
        } else { std::cout <<"ERROR: Trying to set InputValues on a non Input-Layer" <<std::endl; }
    }
    
    
    void feedForward(const Layer& prev) {
        // Forward Propagate the inputValues throug each Neuron of the Layer (- Bias)
        for(ulong i = 0; i < this->getNeuronCountNoBias(); i++) {
            const ulong index = this->neurons[i].index;
            double sum = 0.0f;
            // Sum up all Outputs from the previous Layer (with the Bias Neuron)
            for(const Neuron& n : prev.getNeurons()) { sum += (n.outputValue * n.outputWeights[index].weight); }
            // Apply the sigmoid function to shape the output value (curve between 0.0 and 1.0)
            this->neurons[i].outputValue = sigmoidFunction(sum);
        }
    }
    
    
    // Calculate the new Gradients of the Output-Layer Neurons
    void calculateGradients(const std::vector<double>& expOutputs) {
        if(this->type == LayerType::Output) {
            for(ulong i = 0; i < this->getNeuronCountNoBias(); i++) {
                const double outputVal = this->neurons[i].outputValue;
                const double delta = expOutputs[i] - outputVal;
                this->neurons[i].gradient = delta * gradientFunction(outputVal);
            }
        } else { std::cout <<"ERROR: Trying to calculate OutputGradients on a non Output-Layer" <<std::endl; }
    }
    
    
    // Calculate the new Gradients of the Hidden-Layer Neurons
    void calculateGradients(const Layer& next) {
        if(this->type == LayerType::Hidden) {
            for(ulong i = 0; i < this->getNeuronCount(); i++) {
                const double outputVal = this->neurons[i].outputValue;
                double dow = 0.0f;
                // Sum up all contributions of the errors, to the Neurons in the next Layer
                for(ulong j = 0; j < next.getNeuronCountNoBias(); j++) {
                    dow += (this->neurons[i].outputWeights[j].weight * next.getNeuron(j).gradient);
                }
                this->neurons[i].gradient = dow * gradientFunction(outputVal);
            }
        } else { std::cout <<"ERROR: Trying to calculate HiddenGradients on a non Hidden-Layer" <<std::endl; }
    }
    
    
    void updateWeights(Layer& prev) {
        // All the "incoming" weights are stored in the Connection vector
        // of the Neurons in the previous layer (fully connected)
        for (ulong n = 0; n < this->getNeuronCountNoBias(); n++) {
            for (ulong j = 0; j < prev.getNeuronCount(); j++) {
                Neuron& prevNeuron = prev.getNeuron(j);
                const double oldDeltaWeight = prevNeuron.outputWeights[this->neurons[n].index].deltaWeight;
                // Individual input, magnified by the gradient and the learning rate (ETA),
                // and also add momentum (ALPHA): a fraction of the previous delta weight
                const double newDeltaWeight = (ETA * prevNeuron.outputValue * this->neurons[n].gradient) + (ALPHA * oldDeltaWeight);
                prevNeuron.outputWeights[this->neurons[n].index].deltaWeight = newDeltaWeight;
                prevNeuron.outputWeights[this->neurons[n].index].weight += newDeltaWeight;
            }
        }
        
    }
    
    
    // Calculate the overall Output Layer error
    double getError(const std::vector<double>& expOutputs) const {
        double tmperror = 0.0f;
        if(this->type == LayerType::Output) {
            if(expOutputs.size() == this->getNeuronCountNoBias()) {
                // Add up all the deltas between actual outputs and expected outputs (- 1 Bias)
                for(ulong i = 0; i < this->getNeuronCountNoBias(); i++) {
                    const double delta = expOutputs[i] - this->neurons[i].outputValue;
                    // tmperror stores the sum of the squares of all output value deltas
                    tmperror += (delta * delta);
                }
                // Get the average (squared) error delta
                tmperror /= this->getNeuronCountNoBias();
                // Take the square root, to get the RMS (Root Mean Square Error)
                tmperror = sqrt(tmperror);
            } else { std::cout <<"ERROR: Ammount of OutputValues not equal to Expected OutputValues" <<std::endl; }
        } else { std::cout <<"ERROR: Trying to get Results from a non Output-Layer" <<std::endl; }
        return tmperror;
    }
    
    
    std::vector<double> getResults() const {
        std::vector<double> tmpres;
        if(this->type == LayerType::Output) {
            // Get all the neuron outputs from the output layer (- Bias)
            for(ulong i = 0; i < this->getNeuronCountNoBias(); i++) {
                tmpres.push_back(this->neurons[i].outputValue);
            }
        } else { std::cout <<"ERROR: Trying to get Results from a non Output-Layer" <<std::endl; }
        return tmpres;
    }
    
    
    // GETTER - SETTER
    inline size_t getNeuronCount() const { return  this->neurons.size(); }
    inline size_t getNeuronCountNoBias() const { return  (this->neurons.size() - 1); }
    inline const std::vector<Neuron> & getNeurons() const { return this->neurons; }
    inline const Neuron& getNeuron(ulong index) const { return this->neurons[index]; }
    inline Neuron& getNeuron(ulong index) { return this->neurons[index]; }
    
private:
    // The Sigmoid Function (having an S shaped curve) produces a output value between 0.0 and 1.0
    // Definition:  s(t) = 1 / 1 + e^-t
    inline double sigmoidFunction(double sum) const { return  (1.0f / (1.0f + exp(sum * -1.0f))); }
    
    // OLD "transferFunctionDerivative": return (1.0 - (sum * sum));
    inline double gradientFunction(double sum) const { return (exp(sum * -1.0f) / pow(exp(sum * -1.0f) + 1.0f, 2.0f)); }
    
};

#endif

