#ifndef DECONVLAYER_H
#define DECONVLAYER_H

#include <vector>
#include<iostream>

class DeconvLayer {
public:
    DeconvLayer(int in_channels, int out_channels, int kernel_size, int stride = 1, int padding = 0);

    std::vector<std::vector<std::vector<double>>> forward(const std::vector<std::vector<std::vector<double>>>& input);
    std::vector<std::vector<std::vector<double>>> backward(const std::vector<std::vector<std::vector<double>>>& grad_output);
    void updateWeights(double learning_rate);

    // Измененные методы
    std::vector<std::vector<std::vector<std::vector<double>>>>& getWeights() {
        return weights;
    }

    std::vector<std::vector<std::vector<std::vector<double>>>>& getGradients() {
        return grad_weights;
    }

private:
    int in_channels;
    int out_channels;
    int kernel_size;
    int stride;
    int padding;
    std::vector<std::vector<std::vector<std::vector<double>>>> weights;
    std::vector<std::vector<std::vector<double>>> input;
    std::vector<std::vector<std::vector<std::vector<double>>>> grad_weights;

    void initializeWeights();
};

#endif // DECONVLAYER_H
