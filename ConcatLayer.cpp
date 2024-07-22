#include "ConcatLayer.h"

std::vector<std::vector<std::vector<double>>> ConcatLayer::forward(const std::vector<std::vector<std::vector<double>>>& input1, const std::vector<std::vector<std::vector<double>>>& input2) {
    std::vector<std::vector<std::vector<double>>> output = input1;
    output.insert(output.end(), input2.begin(), input2.end());
    return output;
}
