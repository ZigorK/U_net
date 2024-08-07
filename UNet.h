#ifndef UNET_H
#define UNET_H

#include <vector>
#include <iostream>
#include <cmath>
#include <memory>
#include "Layer.h"
#include "ConvolutionalLayer.h"
#include "DeconvLayer.h"
#include "ReLULayer.h"
#include "MaxPooling.h"
#include "ConcatLayer.h"
#include "Optimizer.h"
#include "SGDOptimizer.h"

double sigmoid(double x);

class UNet {
public:
    UNet();
    
    std::vector<std::vector<std::vector<double>>> forward(const std::vector<std::vector<std::vector<double>>>& input);
    void backward(const std::vector<std::vector<std::vector<double>>>& grad_output);
    void updateWeights(Optimizer& optimizer);

    std::vector<std::vector<std::vector<double>>> getGradients() const;
    void setGradients(const std::vector<std::vector<std::vector<double>>>& gradients);

    std::vector<std::shared_ptr<Layer>> getLayers() const;

private:
    std::vector<std::vector<std::vector<double>>> x1, x3, x5, x7;

    ConvLayer conv1, conv2, conv3, conv4, conv5;
    DeconvLayer deconv1, deconv2, deconv3, deconv4, deconv5;
    ConvLayer output_conv;
    ReLU relu;
    MaxPooling maxpool;
    ConcatLayer concat;

    std::vector<std::vector<std::vector<double>>> weights;
    std::vector<std::vector<std::vector<double>>> gradients;
};

#endif // UNET_H
