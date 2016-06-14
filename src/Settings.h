//  Settings.h
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
#ifndef Settings_h
#define Settings_h

#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <fstream>
#include <cstdlib>

#include <math.h>

#define PATH                        "/Users/peter/Documents/github/C++/NeuralNet_MNIST/MNIST_DATA/"

//  784N Input Layer / 1x 100N Hidden Layer / 10N Output Layer
//  (Actual Net: Every Layer has one additional Bias Neuron)
#define LAYER_NEURON_TOPOLOGY       {784, 100, 10}
#define ETA                         0.2                         // Overall net learning rate (0.0 = slow learner - 1.0 fast learner)
#define ALPHA                       0.8                         // Momentum (Multiplier of the delta weights) optimal range: 0.0 - 1.0
#define DATASETS                    10000                       // Ammount of Trainingdata
#define RAND_0to1                   (rand()/(double)RAND_MAX)   // Generate Random Number between 0.0f and 1.0f
#define SMOOTHING_FACTOR            100                         // Number of training samples to average over
#define DEBUG_OUTPUT                true                        // Display some Debug output

// BIG-Endian to LITTLE-Endian byte swap
#define swap16(n) (((n&0xFF00)>>8)|((n&0x00FF)<<8))
#define swap32(n) ((swap16((n&0xFFFF0000)>>16))|((swap16(n&0x0000FFFF))<<16))

class Neuron;

typedef unsigned long               ulong;
typedef unsigned char               byte;
typedef std::vector<byte>           row;
typedef std::vector<Neuron>         Layer;
typedef std::vector<ulong>          Topology;

#endif
