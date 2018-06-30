//  basicApp.cpp
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

// CLANG / GCC compiler flags:		-std=c++14 -Os
// Visual C++ compiler flags:		/Ox

#include "NeuralNetOOP/NeuralNetOOP.h"
#include "NeuralNetVec/NeuralNetVec.h"

using namespace std;
using namespace chrono;

int main() {
    // Net Interface OOP/VEC
    
    auto netOOP = NeuralNetOOP(LAYER_NEURON_TOPOLOGY);
    auto netVec = NeuralNetVec(LAYER_NEURON_TOPOLOGY, ETA);
	
    // Get the MNIST data
    MNIST mnist = MNIST(PATH_IN);
    std::vector<Vector> mnistInput (mnist.trainingData.size());
    std::vector<Vector> mnistOutput (mnist.trainingData.size());
    std::vector<Vector> mnistInput_test (mnist.testData.size());
    std::vector<Vector> mnistOutput_test (mnist.testData.size());
    for(ulong i = 0; i < mnist.trainingData.size(); i++) {
        mnistInput[i] = mnist.trainingData[i].pixelData;
        mnistOutput[i] = mnist.trainingData[i].output;
    }
    for(ulong i = 0; i < mnist.testData.size(); i++) {
        mnistInput_test[i] = mnist.testData[i].pixelData;
        mnistOutput_test[i] = mnist.testData[i].output;
    }
    
    const auto t1 = steady_clock::now();
    netOOP.train(mnist);
    const auto t2 = steady_clock::now();
    
    const auto t3 = steady_clock::now();
    netVec.Train(TRAINING_ITER, mnistInput, mnistOutput);
    const auto t4 = steady_clock::now();
    
    netOOP.test(mnist, std::string(PATH_OUT) + "OOP.txt");
    netVec.test(mnist, std::string(PATH_OUT) + "VEC.txt");
    
    cout << "NeuralNet OOP training time:\t" <<duration_cast<seconds>(t2 - t1).count() <<" sec." <<endl;
    cout << "NeuralNet VEC training time:\t" <<duration_cast<seconds>(t4 - t3).count() <<" sec." <<endl;

	// keep the Windows Console on screen
	if (WINDOWS) { system("pause"); }

    return 0;
}
