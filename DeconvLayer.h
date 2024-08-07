#ifndef DECONVLAYER_H
#define DECONVLAYER_H

#include "Layer.h"
#include <vector>
#include <fstream>

class DeconvLayer : public Layer {
public:
    DeconvLayer(int in_channels, int out_channels, int kernel_size, int stride = 1, int padding = 1);
    std::vector<std::vector<std::vector<double>>> forward(const std::vector<std::vector<std::vector<double>>>& input);
    std::vector<std::vector<std::vector<double>>> backward(const std::vector<std::vector<std::vector<double>>>& grad_output);
    void updateWeights(double learning_rate);

    std::vector<std::vector<std::vector<std::vector<double>>>>& getWeights();
    std::vector<std::vector<std::vector<std::vector<double>>>>& getGradients();

    void save(std::ofstream& file) const;
    void load(std::ifstream& file);

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
