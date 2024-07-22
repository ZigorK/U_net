#ifndef RELU_H
#define RELU_H

#include <vector>

class ReLU {
public:
    std::vector<std::vector<std::vector<double>>> forward(const std::vector<std::vector<std::vector<double>>>& input);
    std::vector<std::vector<std::vector<double>>> backward(const std::vector<std::vector<std::vector<double>>>& grad_output);

private:
    std::vector<std::vector<std::vector<double>>> input;
};

#endif // RELU_H
