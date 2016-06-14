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

// CLANG / GCC compiler flags: -std=c++14 -Os

#include "NeuralNet.h"
#include "MNIST.h"

using namespace std;
using namespace chrono;


int main() {
    
    const auto timePoint1 = steady_clock::now();
    
    NeuralNet net = NeuralNet(LAYER_NEURON_TOPOLOGY);
    MNIST mnist = MNIST(PATH);
    
    const auto timePoint2 = steady_clock::now();
    
    // train the neural net
    for(const auto& t : mnist.trainingData) {
        net.feedForward(t.pixelData);
        net.backPropagate(t.output);
    }
    
    const auto timePoint3 = steady_clock::now();
    
    
    if(DEBUG_OUTPUT) {
        const auto timePoint4 = steady_clock::now();
        
        double errSum = 0.0f;
        for(const auto& t : mnist.testData) {
            net.feedForward(t.pixelData);
            errSum += net.getNetError();
        }
        std::cout <<"Overall Error: " <<(errSum / mnist.testData.size()) <<std::endl;
        // TEST THE NET WITH THREE RANDOM DIGITS FROM THE MNIST TEST SET
        std::cout <<"\n\nVisualise result, by testing 3 random digits: " <<std::endl;
        std::vector<MNISTchar> test {mnist.testData[2347], mnist.testData[6345], mnist.testData[8754]};
        for(const MNISTchar& in : test) {
            net.feedForward(in.pixelData);
            mnist.testPrintout(in);
            std::cout <<"Expected Values:\tOutput Values:" <<std::endl;
            const auto result = net.getResults();
            for(int i = 0; i < in.output.size(); i++) {
                std::cout <<in.output[i] <<"\t\t\t\t\t";
                printf("%.5f\n", result[i]);
            }
            // GENERATE SOME EASY OUTPUT
            std::string tmp = "The digit is";
            ulong num = 100, count = 0;
            for(ulong i = 0; i < result.size(); i++) {
                if(result[i] >= 0.9f) { num = i; }
                if(result[i] <= 0.1f) { count++; }
            }
            tmp += (count >= 9) ? " definitely a: " : " very likely a: ";
            tmp += "\t" + std::to_string(num);
            std::cout <<tmp <<std::endl;
        }
        
        const auto timePoint5 = steady_clock::now();
        
        cout << "\n\nMNIST parsing time:\t\t\t" <<duration_cast<milliseconds>(timePoint2 - timePoint1).count() <<" ms" <<endl;
        cout << "NeuralNet training time:\t" <<duration_cast<milliseconds>(timePoint3 - timePoint2).count() <<" ms" <<endl;
        cout << "NeuralNet testing time:\t\t" <<duration_cast<milliseconds>(timePoint5 - timePoint4).count() <<" ms" <<endl;
        cout << "Total:\t\t\t\t\t\t" <<duration_cast<seconds>(timePoint5 - timePoint1).count() <<" sec\n" <<endl;
    }
    
    return 0;
}
