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
    int height = input[0].size();
    int width = input[0][0].size();
    int output_height = grad_output[0].size();
    int output_width = grad_output[0][0].size();

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
