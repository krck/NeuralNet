//  basicApp.cpp
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

// CLANG / GCC compiler flags: -std=c++14 -Os

#include "NeuralNet.h"

using namespace std;
using namespace chrono;

int main() {
    
    const auto timePoint1 = steady_clock::now();
    
    NeuralNet net = NeuralNet({784, 100, 10});
    MNIST mnist = MNIST(PATH);
    
    const auto timePoint2 = steady_clock::now();
    
    // train the neural net
    for(const auto& t : mnist.trainingData) {
        net.feedForward(t.pixelData);
        net.backPropagate(t.output);
    }
    
    const auto timePoint3 = steady_clock::now();

    
    
    if(DEBUG_OUTPUT) {
//        cout <<"The first 10 MNIST characters are: " <<endl;
//        mnist.testPrintout(0, 10);

        double errSum = 0.0f;
        for(const auto& t : mnist.testData) {
            net.feedForward(t.pixelData);
            errSum += net.getNetError();
        }
        std::cout <<"Overall Error: " <<(errSum / mnist.testData.size()) <<std::endl;
        
        const auto timePoint4 = steady_clock::now();
        
        // try three random digits
        std::cout <<"\n\nVisualise result, by testing 3 random digits: " <<std::endl;
        std::vector<MNISTchar> test {mnist.testData[rand()%10000], mnist.testData[rand()%10000], mnist.testData[rand()%10000]};
        // Feed all possible AND combinations and print the results
        for(const MNISTchar& in : test) {
            net.feedForward(in.pixelData);
            std::cout <<"\nThe Number is: " <<in.label <<std::endl;
            std::cout <<"Expected Values:\tOutput Values:" <<std::endl;
            const auto result = net.getResults();
            for(int i = 0; i < in.output.size(); i++) {
                std::cout <<in.output[i] <<"\t\t\t\t\t" <<result[i] <<std::endl;
            }
        }
        
        const auto timePoint5 = steady_clock::now();
        
        cout << "\n\nMNIST parsing time:\t\t\t" <<duration_cast<milliseconds>(timePoint2 - timePoint1).count() <<" ms" <<endl;
        cout << "NeuralNet training time:\t" <<duration_cast<milliseconds>(timePoint3 - timePoint2).count() <<" ms" <<endl;
        cout << "NeuralNet testing time:\t\t" <<duration_cast<milliseconds>(timePoint5 - timePoint4).count() <<" ms" <<endl;
        cout << "Total:\t\t\t\t\t\t" <<duration_cast<seconds>(timePoint5 - timePoint1).count() <<" sec\n" <<endl;
    }
    
    return 0;
}
