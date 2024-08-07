#include "ReLULayer.h"
#include <iostream>

std::vector<std::vector<std::vector<double>>> ReLU::forward(const std::vector<std::vector<std::vector<double>>>& input) {
    this->input = input;

    // Проверка на пустоту входного тензора
    if (input.empty() || input[0].empty() || input[0][0].empty()) {
        std::cerr << "Ошибка: Входной тензор пуст." << std::endl;
        return {};
    }

    std::vector<std::vector<std::vector<double>>> output = input;
    for (size_t c = 0; c < output.size(); ++c) {
        for (size_t h = 0; h < output[c].size(); ++h) {
            for (size_t w = 0; w < output[c][h].size(); ++w) {
                if (output[c][h][w] < 0) {
                    output[c][h][w] = 0;
                }
            }
        }
    }

    return output;
}

std::vector<std::vector<std::vector<double>>> ReLU::backward(const std::vector<std::vector<std::vector<double>>>& grad_output) {
    std::cout << "Начало обратного прохода ReLU." << std::endl;
    std::cout << "Размеры градиента выхода: [" << grad_output.size() << ", " << (grad_output.empty() ? 0 : grad_output[0].size()) << ", " << (grad_output.empty() || grad_output[0].empty() ? 0 : grad_output[0][0].size()) << "]" << std::endl;

    std::vector<std::vector<std::vector<double>>> grad_input = grad_output;
    for (size_t c = 0; c < grad_input.size(); ++c) {
        for (size_t h = 0; h < grad_input[c].size(); ++h) {
            for (size_t w = 0; w < grad_input[c][h].size(); ++w) {
                if (input[c][h][w] <= 0) {
                    grad_input[c][h][w] = 0;
                }
            }
        }
    }
    
    std::cout << "Размеры градиента входа: [" << grad_input.size() << ", " << (grad_input.empty() ? 0 : grad_input[0].size()) << ", " << (grad_input.empty() || grad_input[0].empty() ? 0 : grad_input[0][0].size()) << "]" << std::endl;
    std::cout << "Завершение обратного прохода ReLU." << std::endl;

    return grad_input;
}

void ReLU::save(std::ofstream& file) const {
    size_t channels = input.size();
    size_t height = channels ? input[0].size() : 0;
    size_t width = height ? input[0][0].size() : 0;

    file.write(reinterpret_cast<const char*>(&channels), sizeof(channels));
    file.write(reinterpret_cast<const char*>(&height), sizeof(height));
    file.write(reinterpret_cast<const char*>(&width), sizeof(width));

    for (const auto& channel : input) {
        for (const auto& row : channel) {
            file.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(double));
        }
    }
}

void ReLU::load(std::ifstream& file) {
    size_t channels, height, width;

    file.read(reinterpret_cast<char*>(&channels), sizeof(channels));
    file.read(reinterpret_cast<char*>(&height), sizeof(height));
    file.read(reinterpret_cast<char*>(&width), sizeof(width));

    input.resize(channels, std::vector<std::vector<double>>(height, std::vector<double>(width)));

    for (auto& channel : input) {
        for (auto& row : channel) {
            file.read(reinterpret_cast<char*>(row.data()), row.size() * sizeof(double));
        }
    }
}
