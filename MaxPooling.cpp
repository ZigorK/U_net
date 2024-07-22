#include "MaxPooling.h"
#include <iostream>

MaxPooling::MaxPooling(int kernel_size, int stride) : kernel_size(kernel_size), stride(stride) {}

std::vector<std::vector<std::vector<double>>> MaxPooling::forward(const std::vector<std::vector<std::vector<double>>>& input) {
    this->input = input;
    
    std::cout << "Начало операции MaxPooling." << std::endl;
    std::cout << "Размеры входа: [" << input.size() << ", " << (input.empty() ? 0 : input[0].size()) << ", " << (input.empty() || input[0].empty() ? 0 : input[0][0].size()) << "]" << std::endl;

    int height = input[0].size();
    int width = input[0][0].size();
    int output_height = (height - kernel_size) / stride + 1;
    int output_width = (width - kernel_size) / stride + 1;

    std::vector<std::vector<std::vector<double>>> output(input.size(), std::vector<std::vector<double>>(output_height, std::vector<double>(output_width, 0.0)));
    mask.resize(input.size(), std::vector<std::vector<int>>(height, std::vector<int>(width, 0)));

    for (size_t c = 0; c < input.size(); ++c) {
        for (int i = 0; i < output_height; ++i) {
            for (int j = 0; j < output_width; ++j) {
                double max_val = input[c][i * stride][j * stride];
                int max_idx = i * stride * width + j * stride;

                for (int ki = 0; ki < kernel_size; ++ki) {
                    for (int kj = 0; kj < kernel_size; ++kj) {
                        int input_row = i * stride + ki;
                        int input_col = j * stride + kj;

                        if (input[c][input_row][input_col] > max_val) {
                            max_val = input[c][input_row][input_col];
                            max_idx = input_row * width + input_col;
                        }
                    }
                }

                output[c][i][j] = max_val;
                mask[c][max_idx / width][max_idx % width] = 1;
            }
        }
    }
    
    std::cout << "Размеры выхода: [" << output.size() << ", " << (output.empty() ? 0 : output[0].size()) << ", " << (output.empty() || output[0].empty() ? 0 : output[0][0].size()) << "]" << std::endl;
    std::cout << "Завершение операции MaxPooling." << std::endl;

    return output;
}

std::vector<std::vector<std::vector<double>>> MaxPooling::backward(const std::vector<std::vector<std::vector<double>>>& grad_output) {
    std::vector<std::vector<std::vector<double>>> grad_input = input;

    for (size_t c = 0; c < grad_output.size(); ++c) {
        for (size_t i = 0; i < grad_output[c].size(); ++i) {
            for (size_t j = 0; j < grad_output[c][i].size(); ++j) {
                for (size_t m = 0; m < input[c].size(); ++m) {
                    for (size_t n = 0; n < input[c][m].size(); ++n) {
                        if (mask[c][m][n] == 1) {
                            grad_input[c][m][n] = grad_output[c][i][j];
                        } else {
                            grad_input[c][m][n] = 0;
                        }
                    }
                }
            }
        }
    }

    return grad_input;
}
