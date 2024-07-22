#ifndef CONVLAYER_H
#define CONVLAYER_H

#include <vector>

class ConvLayer {
public:
    ConvLayer(int in_channels, int out_channels, int kernel_size, int stride = 1, int padding = 1);
    std::vector<std::vector<std::vector<double>>> forward(const std::vector<std::vector<std::vector<double>>>& input);

private:
    int in_channels;
    int out_channels;
    int kernel_size;
    int stride;
    int padding;
    std::vector<std::vector<std::vector<std::vector<double>>>> weights;
    std::vector<std::vector<std::vector<double>>> input;

    void initializeWeights();
};

#endif // CONVLAYER_H
