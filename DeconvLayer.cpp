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
}

std::vector<std::vector<std::vector<double>>> DeconvLayer::forward(const std::vector<std::vector<std::vector<double>>>& input) {
    int height = input[0].size();
    int width = input[0][0].size();
    int output_height = (height - 1) * stride - 2 * padding + kernel_size;
    int output_width = (width - 1) * stride - 2 * padding + kernel_size;

    std::cout << "Start of DeconvLayer forward operation" << std::endl;
    std::cout << "Input tensor size: [" << in_channels << ", " << height << ", " << width << "]" << std::endl;
    std::cout << "Kernel size: " << kernel_size << ", Stride: " << stride << ", Padding: " << padding << std::endl;
    std::cout << "Output tensor size: [" << out_channels << ", " << output_height << ", " << output_width << "]" << std::endl;

    std::vector<std::vector<std::vector<double>>> output(out_channels, std::vector<std::vector<double>>(output_height, std::vector<double>(output_width, 0.0)));

    std::cout << "Weight tensor size: [" << out_channels << ", " << in_channels << ", " << kernel_size << ", " << kernel_size << "]" << std::endl;

    bool errorOccurred = false;

    for (int oc = 0; oc < out_channels; ++oc) {
        std::cout << "Processing output channel " << oc + 1 << "/" << out_channels << std::endl;
        for (int ic = 0; ic < in_channels; ++ic) {
            for (int i = 0; i < height; ++i) {
                for (int j = 0; j < width; ++j) {
                    for (int ki = 0; ki < kernel_size; ++ki) {
                        for (int kj = 0; kj < kernel_size; ++kj) {
                            int output_row = i * stride + ki - padding;
                            int output_col = j * stride + kj - padding;
                            if (output_row >= 0 && output_row < output_height && output_col >= 0 && output_col < output_width) {
                                if (weights.size() <= oc || weights[oc].size() <= ic || weights[oc][ic].size() <= ki || weights[oc][ic][ki].size() <= kj) {
                                    if (!errorOccurred) {
                                        std::cerr << "Error: Weights array is not correctly initialized at oc=" << oc << ", ic=" << ic << ", ki=" << ki << ", kj=" << kj << std::endl;
                                        errorOccurred = true;
                                        return output;
                                    }
                                } else {
                                    if (oc < output.size() && output_row < output[oc].size() && output_col < output[oc][output_row].size() &&
                                        ic < input.size() && i < input[ic].size() && j < input[ic][i].size()) {
                                        output[oc][output_row][output_col] += input[ic][i][j] * weights[oc][ic][ki][kj];
                                    } else {
                                        if (!errorOccurred) {
                                            std::cerr << "Error: Index out of bounds. oc=" << oc << ", output_row=" << output_row << ", output_col=" << output_col
                                                      << ", ic=" << ic << ", i=" << i << ", j=" << j << std::endl;
                                            errorOccurred = true;
                                            return output;
                                        }
                                    }
                                }
                            } else {
                                if (!errorOccurred) {
                                    std::cerr << "Warning: Index out of bounds. output_row: " << output_row << ", output_col: " << output_col << std::endl;
                                    errorOccurred = true;
                                    return output;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    std::cout << "DeconvLayer forward operation completed" << std::endl;
    return output;
}
