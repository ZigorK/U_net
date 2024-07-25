#include "DeconvLayer.h"

DeconvLayer::DeconvLayer(int in_channels, int out_channels, int kernel_size, int stride, int padding)
    : in_channels(in_channels), out_channels(out_channels), kernel_size(kernel_size), stride(stride), padding(padding) {
    initializeWeights();
}

void DeconvLayer::initializeWeights() {
    weights.resize(out_channels, std::vector<std::vector<std::vector<double>>>(in_channels, std::vector<std::vector<double>>(kernel_size, std::vector<double>(kernel_size))));
    for (int i = 0; i < out_channels; ++i) {
        for (int j = 0; j < in_channels; ++j) {
            for (int k = 0; k < kernel_size; ++k) {
                for (int l = 0; l < kernel_size; ++l) {
                    weights[i][j][k][l] = static_cast<double>(rand()) / RAND_MAX;
                }
            }
        }
    }
    grad_weights = weights; // Инициализируем градиенты весов такими же размерами, как и веса
}

std::vector<std::vector<std::vector<double>>> DeconvLayer::forward(const std::vector<std::vector<std::vector<double>>>& input) {
    this->input = input; // Сохранение входных данных для backward

    int height = input[0].size();
    int width = input[0][0].size();
    int output_height = (height - 1) * stride - 2 * padding + kernel_size;
    int output_width = (width - 1) * stride - 2 * padding + kernel_size;

    std::vector<std::vector<std::vector<double>>> output(out_channels, std::vector<std::vector<double>>(output_height, std::vector<double>(output_width, 0.0)));

    // Отладочная информация о размере входных данных и весов
    std::cout << "Input size: " << input.size() << "x" << height << "x" << width << std::endl;
    std::cout << "Weights size: " << weights.size() << "x" << weights[0].size() << "x" << weights[0][0].size() << "x" << weights[0][0][0].size() << std::endl;

    for (int oc = 0; oc < out_channels; ++oc) {
        if (oc >= output.size()) {
            std::cerr << "Error: Output channel index out of range: " << oc << std::endl;
            continue;
        }

        for (int ic = 0; ic < in_channels; ++ic) {
            if (ic >= input.size()) {
                std::cerr << "Error: Input channel index out of range: " << ic << std::endl;
                continue;
            }

            for (int i = 0; i < height; ++i) {
                if (i >= input[ic].size()) {
                    std::cerr << "Error: Input height index out of range: " << i << std::endl;
                    continue;
                }

                for (int j = 0; j < width; ++j) {
                    if (j >= input[ic][i].size()) {
                        std::cerr << "Error: Input width index out of range: " << j << std::endl;
                        continue;
                    }

                    for (int ki = 0; ki < kernel_size; ++ki) {
                        for (int kj = 0; kj < kernel_size; ++kj) {
                            int output_row = i * stride + ki - padding;
                            int output_col = j * stride + kj - padding;

                            if (output_row >= 0 && output_row < output_height && output_col >= 0 && output_col < output_width) {
                                if (oc >= weights.size() || ic >= weights[oc].size() || ki >= weights[oc][ic].size() || kj >= weights[oc][ic][ki].size()) {
                                    std::cerr << "Error: Weights index out of range. oc: " << oc << ", ic: " << ic << ", ki: " << ki << ", kj: " << kj << std::endl;
                                    continue;
                                }

                                output[oc][output_row][output_col] += input[ic][i][j] * weights[oc][ic][ki][kj];
                            }
                        }
                    }
                }
            }
        }
    }

    return output;
}





std::vector<std::vector<std::vector<double>>> DeconvLayer::backward(const std::vector<std::vector<std::vector<double>>>& grad_output) {
    int height = input[0].size();
    int width = input[0][0].size();
    int output_height = grad_output[0].size();
    int output_width = grad_output[0][0].size();

    std::vector<std::vector<std::vector<double>>> grad_input(in_channels, std::vector<std::vector<double>>(height, std::vector<double>(width, 0.0)));
    grad_weights = weights; // Обнуляем градиенты весов

    for (int oc = 0; oc < out_channels; ++oc) {
        for (int ic = 0; ic < in_channels; ++ic) {
            for (int i = 0; i < height; ++i) {
                for (int j = 0; j < width; ++j) {
                    for (int ki = 0; ki < kernel_size; ++ki) {
                        for (int kj = 0; kj < kernel_size; ++kj) {
                            int output_row = i * stride + ki - padding;
                            int output_col = j * stride + kj - padding;
                            if (output_row >= 0 && output_row < output_height && output_col >= 0 && output_col < output_width) {
                                grad_input[ic][i][j] += grad_output[oc][output_row][output_col] * weights[oc][ic][ki][kj];
                                grad_weights[oc][ic][ki][kj] += grad_output[oc][output_row][output_col] * input[ic][i][j];
                            }
                        }
                    }
                }
            }
        }
    }

    return grad_input;
}

void DeconvLayer::updateWeights(double learning_rate) {
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
