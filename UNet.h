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
#include "Optimizer.h"
#include "SGDOptimizer.h"

double sigmoid(double x);

class UNet {
public:
    UNet();
    
    // Прямой проход
    std::vector<std::vector<std::vector<double>>> forward(const std::vector<std::vector<std::vector<double>>>& input);

    // Обратное распространение ошибки
    void backward(const std::vector<std::vector<std::vector<double>>>& grad_output);

    // Обновление весов
    void updateWeights(Optimizer& optimizer);

    // Методы для работы с градиентами
    std::vector<std::vector<std::vector<double>>> getGradients() const;
    void setGradients(const std::vector<std::vector<std::vector<double>>>& gradients);

private:
    // Сохранение промежуточных значений
    std::vector<std::vector<std::vector<double>>> x1, x3, x5, x7;

    // Слои модели
    ConvLayer conv1, conv2, conv3, conv4, conv5;
    DeconvLayer deconv1, deconv2, deconv3, deconv4;
    ConvLayer output_conv;
    ReLU relu;
    MaxPooling maxpool;
    ConcatLayer concat;

    // Хранение весов и градиентов
    std::vector<std::vector<std::vector<double>>> weights;
    std::vector<std::vector<std::vector<double>>> gradients;
};

#endif // UNET_H
