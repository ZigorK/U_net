#ifndef DECONVLAYER_H
#define DECONVLAYER_H

#include <iostream>
#include <vector>

class DeconvLayer {
public:
    DeconvLayer(int in_channels, int out_channels, int kernel_size, int stride = 1, int padding = 0);

    std::vector<std::vector<std::vector<double>>> forward(const std::vector<std::vector<std::vector<double>>>& input);

private:
    int in_channels;
    int out_channels;
    int kernel_size;
    int stride;
    int padding;
    std::vector<std::vector<std::vector<std::vector<double>>>> weights;

    void initializeWeights();
};

#endif // DECONVLAYER_H
