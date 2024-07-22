#ifndef MAXPOOLING_H
#define MAXPOOLING_H

#include <vector>

class MaxPooling {
public:
    MaxPooling(int kernel_size, int stride);
    std::vector<std::vector<std::vector<double>>> forward(const std::vector<std::vector<std::vector<double>>>& input);
    std::vector<std::vector<std::vector<double>>> backward(const std::vector<std::vector<std::vector<double>>>& grad_output);

private:
    int kernel_size;
    int stride;
    std::vector<std::vector<std::vector<double>>> input;
    std::vector<std::vector<std::vector<int>>> mask;
};

#endif // MAXPOOLING_H
