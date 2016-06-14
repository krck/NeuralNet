//  MNIST..h
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
#ifndef MNIST_h
#define MNIST_h

#include "Settings.h"


struct MNISTchar {
    std::vector<double> pixelData;          // Store the 784 (28x28) pixel color values (0-255) of the digit-image
    std::vector<double> output;             // Store the expected output (e.g: label 5 / output 0,0,0,0,0,1,0,0,0,0)
    int label;                              // Store the handwritten digit in number form
    MNISTchar() : pixelData(std::vector<double>()), output(std::vector<double>(10)), label(0) {}
};



class MNIST {
public:
    const std::vector<MNISTchar> trainingData;  // Set of 60.000 handwritten digits to train the net
    const std::vector<MNISTchar> testData;      // Set of 10.000 different handwritten digits to test the net
    
    
    MNIST(const std::string& path)
        :   trainingData(getMNISTdata(path + "train-images.idx3-ubyte", path + "train-labels.idx1-ubyte")),
            testData(getMNISTdata(path + "t10k-images.idx3-ubyte", path + "t10k-labels.idx1-ubyte")) {
                if(!this->trainingData.size()) { std::cout <<"ERROR: parsing training data" <<std::endl; }
                if(!this->testData.size()) { std::cout <<"ERROR: parsing testing data" <<std::endl; }
            }
    
    
private:
    std::vector<MNISTchar> getMNISTdata(const std::string& imagepath, const std::string& labelpath) {
        std::vector<MNISTchar> tmpdata = std::vector<MNISTchar>();
        std::fstream file (imagepath, std::ifstream::in | std::ifstream::binary);
        int magicNum_images = 0, magicNum_labels = 0;
        int itemCount_images = 0, itemCount_labels = 0;
        // READ THE IMAGE FILE DATA
        if(file.is_open()) {
            int row_count = 0, col_count = 0;
            // FILE HEADER INFO is stored as 4 Byte Integers
            file.read((char*)&magicNum_images, 4);
            file.read((char*)&itemCount_images, 4);
            file.read((char*)&row_count, 4);
            file.read((char*)&col_count, 4);
            // Transform Byte values from big to little endian
            magicNum_images = swap32(magicNum_images);
            itemCount_images = swap32(itemCount_images);
            row_count = swap32(row_count);
            col_count= swap32(col_count);
            // Loop throug all the items and store every pixel of every row
            for (int i = 0; i < itemCount_images; i++) {
                MNISTchar tmpchar = MNISTchar();
                for(int r = 0; r < (row_count * col_count); r++) {
                    byte pixel = 0;
                    // read one byte (0-255 color value of the pixel)
                    file.read((char*)&pixel, 1);
                    tmpchar.pixelData.push_back((double)pixel / 255);
                }
                tmpdata.push_back(tmpchar);
            }
        }
        file.close();
        // READ THE LABEL FILE DATA
        file.open(labelpath, std::ifstream::in | std::ifstream::binary);
        if (file.is_open()) {
            file.read((char*)&magicNum_labels, 4);
            file.read((char*)&itemCount_labels, 4);
            magicNum_labels = swap32(magicNum_labels);
            itemCount_labels = swap32(itemCount_labels);
            if(itemCount_images == itemCount_labels) {
                // read all the labels and store them in theire MNISTchars
                for(MNISTchar& m : tmpdata) {
                    file.read((char*)&m.label, 1);
                    m.output[m.label] = 1.0f;
                }
            }
        }
        file.close();
        return tmpdata;
    }
    
    
public:
    void testPrintout(const MNISTchar& mchar) const {
        std::cout <<"------------------------------" <<std::endl;
        int count = 0;
        for (const double& r : mchar.pixelData) {
            if(count < 27) {
                if(r < 0.25) std::cout <<" ";
                else if(r < 0.5) std::cout <<"-";
                else if(r < 0.75) std::cout <<"+";
                else if(r <= 1.0) std::cout <<"#";
                ++count;
            } else {
                std::cout <<std::endl;
                count = 0;
            }
        }
        std::cout <<"\t\tThis is a: " <<mchar.label  <<std::endl;
        std::cout <<"------------------------------" <<std::endl;
    }
    
    
};

#endif
