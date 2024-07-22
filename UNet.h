#ifndef UNET_H
#define UNET_H

#include <vector>
#include <iostream>
#include <cmath>
#include "ConvolutionalLayer.h"
#include "DeconvLayer.h"
#include "ReLULayer.h"
#include "MaxPooling.h"
#include "ConcatLayer.h"

double sigmoid(double x);

class UNet {
public:
    UNet();
    std::vector<std::vector<std::vector<double>>> forward(const std::vector<std::vector<std::vector<double>>>& input);

private:
    ConvLayer conv1, conv2, conv3, conv4, conv5;
    DeconvLayer deconv1, deconv2, deconv3, deconv4;
    ConvLayer output_conv;
    ReLU relu;
    MaxPooling maxpool;
    ConcatLayer concat;
};

#endif // UNET_H
