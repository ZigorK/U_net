#include "ReLULayer.h"
#include <iostream>

std::vector<std::vector<std::vector<double>>> ReLU::forward(const std::vector<std::vector<std::vector<double>>>& input) {
    this->input = input;
    
    std::cout << "Начало операции ReLU." << std::endl;
    std::cout << "Размеры входа: [" << input.size() << ", " << (input.empty() ? 0 : input[0].size()) << ", " << (input.empty() || input[0].empty() ? 0 : input[0][0].size()) << "]" << std::endl;

    std::vector<std::vector<std::vector<double>>> output = input;
    for (auto& channel : output) {
        for (auto& row : channel) {
            for (auto& value : row) {
                if (value < 0) value = 0;
            }
        }
    }
    
    std::cout << "Размеры выхода: [" << output.size() << ", " << (output.empty() ? 0 : output[0].size()) << ", " << (output.empty() || output[0].empty() ? 0 : output[0][0].size()) << "]" << std::endl;
    std::cout << "Завершение операции ReLU." << std::endl;

    return output;
}

std::vector<std::vector<std::vector<double>>> ReLU::backward(const std::vector<std::vector<std::vector<double>>>& grad_output) {
    std::cout << "Начало обратного прохода ReLU." << std::endl;
    std::cout << "Размеры градиента выхода: [" << grad_output.size() << ", " << (grad_output.empty() ? 0 : grad_output[0].size()) << ", " << (grad_output.empty() || grad_output[0].empty() ? 0 : grad_output[0][0].size()) << "]" << std::endl;

    std::vector<std::vector<std::vector<double>>> grad_input = grad_output;
    for (size_t i = 0; i < input.size(); ++i) {
        for (size_t j = 0; j < input[i].size(); ++j) {
            for (size_t k = 0; k < input[i][j].size(); ++k) {
                if (input[i][j][k] <= 0) grad_input[i][j][k] = 0;
            }
        }
    }
    
    std::cout << "Размеры градиента входа: [" << grad_input.size() << ", " << (grad_input.empty() ? 0 : grad_input[0].size()) << ", " << (grad_input.empty() || grad_input[0].empty() ? 0 : grad_input[0][0].size()) << "]" << std::endl;
    std::cout << "Завершение обратного прохода ReLU." << std::endl;

    return grad_input;
}
