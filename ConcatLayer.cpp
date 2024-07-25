#include "ConcatLayer.h"

// Прямое распространение
std::vector<std::vector<std::vector<double>>> ConcatLayer::forward(const std::vector<std::vector<std::vector<double>>>& input1, const std::vector<std::vector<std::vector<double>>>& input2) {
    std::vector<std::vector<std::vector<double>>> output = input1;
    output.insert(output.end(), input2.begin(), input2.end());
    return output;
}

// Обратное распространение
std::pair<std::vector<std::vector<std::vector<double>>>, std::vector<std::vector<std::vector<double>>>> ConcatLayer::backward(const std::vector<std::vector<std::vector<double>>>& grad_output) {
    size_t size1 = grad_output.size() / 2;
    std::vector<std::vector<std::vector<double>>> grad_input1(grad_output.begin(), grad_output.begin() + size1);
    std::vector<std::vector<std::vector<double>>> grad_input2(grad_output.begin() + size1, grad_output.end());

    return std::make_pair(grad_input1, grad_input2);
}
