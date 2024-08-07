#include "ConvolutionalLayer.h"
#include <iostream>
#include <cmath>
#include <cstdlib>

ConvLayer::ConvLayer(int in_channels, int out_channels, int kernel_size, int stride, int padding)
    : in_channels(in_channels), out_channels(out_channels), kernel_size(kernel_size), stride(stride), padding(padding) {
    initializeWeights();
}

void ConvLayer::initializeWeights() {
    weights.resize(out_channels, std::vector<std::vector<std::vector<double>>>(in_channels, std::vector<std::vector<double>>(kernel_size, std::vector<double>(kernel_size))));
    for (int i = 0; i < out_channels; ++i) {
        for (int j = 0; j < in_channels; ++j) {
            for (int k = 0; k < kernel_size; ++k) {
                for (int l = 0; l < kernel_size; ++l) {
                    weights[i][j][k][l] = ((double)rand() / (RAND_MAX));
                }
            }
        }
    }
    grad_weights = weights; // Инициализируем градиенты весов такими же размерами, как и веса
}

std::vector<std::vector<std::vector<double>>> ConvLayer::forward(const std::vector<std::vector<std::vector<double>>>& input) {
    this->input = input;
    int height = input[0].size();
    int width = input[0][0].size();

    int output_height = (height - kernel_size + 2 * padding) / stride + 1;
    int output_width = (width - kernel_size + 2 * padding) / stride + 1;

    if (output_height <= 0 || output_width <= 0) {
        return {};
    }

    std::vector<std::vector<std::vector<double>>> output(out_channels, std::vector<std::vector<double>>(output_height, std::vector<double>(output_width, 0.0)));

    for (int oc = 0; oc < out_channels; ++oc) {
        for (int ic = 0; ic < in_channels; ++ic) {
            for (int i = 0; i < output_height; ++i) {
                for (int j = 0; j < output_width; ++j) {
                    for (int ki = 0; ki < kernel_size; ++ki) {
                        for (int kj = 0; kj < kernel_size; ++kj) {
                            int input_row = i * stride + ki - padding;
                            int input_col = j * stride + kj - padding;
                            if (input_row >= 0 && input_row < height && input_col >= 0 && input_col < width) {
                                output[oc][i][j] += input[ic][input_row][input_col] * weights[oc][ic][ki][kj];
                            }
                        }
                    }
                }
            }
        }
    }

    return output;
}

std::vector<std::vector<std::vector<double>>> ConvLayer::backward(const std::vector<std::vector<std::vector<double>>>& grad_output) {
    // Проверка и вывод размеров grad_output
    if (grad_output.empty() || grad_output[0].empty() || grad_output[0][0].empty()) {
        std::cerr << "grad_output is empty or has incorrect dimensions." << std::endl;
        std::cerr << "grad_output size: " << grad_output.size() << std::endl;
        return {};
    }

    // Проверка и вывод размеров input
    if (input.empty() || input[0].empty() || input[0][0].empty()) {
        std::cerr << "Input is empty or has incorrect dimensions." << std::endl;
        std::cerr << "Input size: " << input.size() << std::endl;
        return {};
    }

    // Проверка и вывод размеров weights
    if (weights.empty() || weights[0].empty() || weights[0][0].empty() || weights[0][0][0].empty()) {
        std::cerr << "Weights are empty or have incorrect dimensions." << std::endl;
        std::cerr << "Weights size: " << weights.size() << std::endl;
        return {};
    }

    // Вывод значений in_channels и out_channels
    std::cerr << "in_channels: " << in_channels << std::endl;
    std::cerr << "out_channels: " << out_channels << std::endl;

    int height = input[0].size();
    int width = input[0][0].size();
    int output_height = grad_output[0].size();
    int output_width = grad_output[0][0].size();

    std::cerr << "Input dimensions: (" << input.size() << ", " << height << ", " << width << ")" << std::endl;
    std::cerr << "grad_output dimensions: (" << grad_output.size() << ", " << output_height << ", " << output_width << ")" << std::endl;
    std::cerr << "Weights dimensions: (" << weights.size() << ", " << weights[0].size() << ", " << weights[0][0].size() << ", " << weights[0][0][0].size() << ")" << std::endl;

    // Проверка соответствия размеров input и grad_output
    if (in_channels != input.size()) {
        std::cerr << "Mismatch in_channels: expected " << in_channels << ", but got " << input.size() << std::endl;
        return {};
    }

    if (out_channels != grad_output.size()) {
        std::cerr << "Mismatch out_channels: expected " << out_channels << ", but got " << grad_output.size() << std::endl;
        return {};
    }

    // Инициализация grad_input и grad_weights
    std::vector<std::vector<std::vector<double>>> grad_input(in_channels, std::vector<std::vector<double>>(height, std::vector<double>(width, 0.0)));
    grad_weights = weights; // Обнуляем градиенты весов

    for (int oc = 0; oc < out_channels; ++oc) {
        for (int ic = 0; ic < in_channels; ++ic) {
            for (int i = 0; i < output_height; ++i) {
                for (int j = 0; j < output_width; ++j) {
                    for (int ki = 0; ki < kernel_size; ++ki) {
                        for (int kj = 0; kj < kernel_size; ++kj) {
                            int input_row = i * stride + ki - padding;
                            int input_col = j * stride + kj - padding;
                            if (input_row >= 0 && input_row < height && input_col >= 0 && input_col < width) {
                                grad_input[ic][input_row][input_col] += grad_output[oc][i][j] * weights[oc][ic][ki][kj];
                                grad_weights[oc][ic][ki][kj] += grad_output[oc][i][j] * input[ic][input_row][input_col];
                            }
                        }
                    }
                }
            }
        }
    }

    return grad_input;
}

void ConvLayer::updateWeights(double learning_rate) {
    for (int i = 0; i < out_channels; ++i) {
        for (int j = 0; j < in_channels; ++j) {
            for (int k = 0; k < kernel_size; ++k) {
                for (int l = 0; l < kernel_size; ++l) {
                    weights[i][j][k][l] -= learning_rate * grad_weights[i][j][k][l];
                }
            }
        }
    }
}

std::vector<std::vector<std::vector<std::vector<double>>>>& ConvLayer::getWeights() {
    return weights;
}

std::vector<std::vector<std::vector<std::vector<double>>>>& ConvLayer::getGradients() {
    return grad_weights;
}

void ConvLayer::save(std::ofstream& file) const {
    file.write(reinterpret_cast<const char*>(&in_channels), sizeof(in_channels));
    file.write(reinterpret_cast<const char*>(&out_channels), sizeof(out_channels));
    file.write(reinterpret_cast<const char*>(&kernel_size), sizeof(kernel_size));
    file.write(reinterpret_cast<const char*>(&stride), sizeof(stride));
    file.write(reinterpret_cast<const char*>(&padding), sizeof(padding));
    for (const auto& out_channel_weights : weights) {
        for (const auto& in_channel_weights : out_channel_weights) {
            for (const auto& row : in_channel_weights) {
                file.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(double));
            }
        }
    }
}

void ConvLayer::load(std::ifstream& file) {
    file.read(reinterpret_cast<char*>(&in_channels), sizeof(in_channels));
    file.read(reinterpret_cast<char*>(&out_channels), sizeof(out_channels));
    file.read(reinterpret_cast<char*>(&kernel_size), sizeof(kernel_size));
    file.read(reinterpret_cast<char*>(&stride), sizeof(stride));
    file.read(reinterpret_cast<char*>(&padding), sizeof(padding));
    weights.resize(out_channels, std::vector<std::vector<std::vector<double>>>(in_channels, std::vector<std::vector<double>>(kernel_size, std::vector<double>(kernel_size))));
    for (auto& out_channel_weights : weights) {
        for (auto& in_channel_weights : out_channel_weights) {
            for (auto& row : in_channel_weights) {
                file.read(reinterpret_cast<char*>(row.data()), row.size() * sizeof(double));
            }
        }
    }
}
