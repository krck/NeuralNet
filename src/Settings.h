//  Settings.h
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
