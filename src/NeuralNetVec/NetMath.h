//  NetMath.h
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

#include <vector>
#include <numeric>
#include <math.h>

#include "../NetBase.h"

Vector CalculateDotSigmoid(const Matrix& weights, const Vector& values, const Vector& bias, const size_t matRows) {
    const size_t matColumns = values.size();
    Vector result(matRows);
    
    for (size_t r = 0; r < matRows; r++) {
        for (size_t c = 0; c < matColumns; c++) {
            // Get the dot product (sum of multiplications)
            // of the weight-matix and value-vector for each row
            result[r] += (weights[r * matColumns + c] * values[c]);
        }
        // Calculate the sigmoid function f(x) = 1/(1 + e^-x) based on the sum and added bias
        result[r] = 1 / (1 + exp(-(result[r] + bias[r])));
    }
    
    return result;
}


Vector CalculateDotSigmoidPrime(const Matrix& weights, const Vector& values, const Vector& bias, const size_t matRows) {
    const size_t matColumns = values.size();
    Vector result(matRows);
    float tmp = 0.0f;
    
    for (size_t r = 0; r < matRows; r++) {
        for (size_t c = 0; c < matColumns; c++) {
            // Get the dot product (sum of multiplications)
            // of the weight-matix and value-vector for each row
            result[r] += (weights[r * matColumns + c] * values[c]);
        }
        // Calculate the sigmoid prime function based on the sum and added bias
        tmp = (result[r] + bias[r]);
        result[r] = exp(-tmp) / (pow(1 + exp(-tmp), 2));
    }
    
    return result;
}


// Calculate the delta between the nets output and the expected output and multiply by the sigmoid prime values
Vector CalculateLastBiasDelta(const Vector& netOutput, const Vector& expectedOutput, const Vector& dotSigmoidPrime) {
    Vector result(netOutput.size());
    
    for (size_t i = 0; i < netOutput.size(); i++) {
        result[i] = (netOutput[i] - expectedOutput[i]) * dotSigmoidPrime[i];
    }
    
    return result;
}


//dEdB[i] = dEdB[i+1] .dot( W[i+1].transpose()).  multiply  (H[i].dot(W[i]).add(B[i]).applyFunction(sigmoidePrime));
Vector CalculateBiasDelta(const Vector& nextBiasDelta, const Matrix& nextWeights, const size_t rows, const size_t columns, const Vector& dotSigmoidPrime) {
    Vector result(dotSigmoidPrime.size());
    
    // Transpose weight mat
    Matrix transposedWeights(nextWeights.size());
    for (size_t n = 0; n != nextWeights.size(); n++) {
        transposedWeights[n] = nextWeights[rows * (n % columns) + (n / columns)];
    }
    
    // Calculate nextBiasDelta DOT transposedWeights (column count is now row count)
    const size_t matColumns = nextBiasDelta.size();
    const size_t matRows = columns;
    Vector dotResult(matRows);
    
    for (size_t r = 0; r < matRows; r++) {
        for (size_t c = 0; c < matColumns; c++) {
            // Get the dot product (sum of multiplications)
            dotResult[r] += (transposedWeights[r * matColumns + c] * nextBiasDelta[c]);
        }
    }
    
    // dotResult * dotSigmoidPrime
    for (size_t i = 0; i < dotSigmoidPrime.size(); i++) {
        result[i] = dotResult[i] * dotSigmoidPrime[i];
    }
    
    return result;
}


Matrix CalculateWeightDelta(const Vector& neurons, const Vector& biasDelta) {
    Matrix result(neurons.size() * biasDelta.size());
    
    // No transpose needed ... just dot the vecs
    for (int row = 0; row < neurons.size(); row++) {
        for (int col = 0; col < biasDelta.size(); col++) {
            result[row * biasDelta.size() + col] = (neurons[row] * biasDelta[col]);
        }
    }
    
    return result;
}


// W[i].subtract(dEdW[i].multiply(learningRate))
Matrix UpdateWeight(const Matrix& weight, const Matrix& weightDelta, const float learnRate) {
    Matrix result(weight.size());
    for (size_t i = 0; i < weight.size(); i++) {
        result[i] = (weight[i] - (weightDelta[i] * learnRate));
    }
    return result;
}


// B[i].subtract(dEdB[i].multiply(learningRate))
Vector UpdateBias(const Vector& bias, const Vector& biasDelta, const float learnRate) {
    Vector result(bias.size());
    for (size_t i = 0; i < bias.size(); i++) {
        result[i] = (bias[i] - (biasDelta[i] * learnRate));
    }
    return result;
}
