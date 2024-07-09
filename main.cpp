#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <limits>

// свёрточный слой
class ConvLayer {
public:
    ConvLayer(int in_channels, int out_channels, int kernel_size, int stride = 1, int padding = 0)
        : in_channels(in_channels), out_channels(out_channels), kernel_size(kernel_size), stride(stride), padding(padding) {
        initializeWeights();
    }

    std::vector<std::vector<std::vector<double>>> forward(const std::vector<std::vector<std::vector<double>>>& input) {
        int height = input[0].size();
        int width = input[0][0].size();
        int output_height = (height - kernel_size + 2 * padding) / stride + 1;
        int output_width = (width - kernel_size + 2 * padding) / stride + 1;
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

private:
    int in_channels;
    int out_channels;
    int kernel_size;
    int stride;
    int padding;
    std::vector<std::vector<std::vector<std::vector<double>>>> weights;

    void initializeWeights() {
        weights.resize(out_channels, std::vector<std::vector<std::vector<double>>>(in_channels, std::vector<std::vector<double>>(kernel_size, std::vector<double>(kernel_size))));
        for (int i = 0; i < out_channels; ++i) {
            for (int j = 0; j < in_channels; ++j) {
                for (int k = 0; k < kernel_size; ++k) {
                    for (int l = 0; l < kernel_size; ++l) {
                        weights[i][j][k][l] = ((double) rand() / (RAND_MAX));
                    }
                }
            }
        }
    }
};

// функция ReLU
class ReLU {
public:
    std::vector<std::vector<std::vector<double>>> forward(const std::vector<std::vector<std::vector<double>>>& input) {
        std::vector<std::vector<std::vector<double>>> output = input;
        for (auto& channel : output) {
            for (auto& row : channel) {
                for (auto& value : row) {
                    if (value < 0) value = 0;
                }
            }
        }
        return output;
    }
};

// объединяющий слой MaxPooling
class MaxPooling {
public:
    MaxPooling(int pool_size, int stride)
        : pool_size(pool_size), stride(stride) {}

    std::vector<std::vector<std::vector<double>>> forward(const std::vector<std::vector<std::vector<double>>>& input) {
        int height = input[0].size();
        int width = input[0][0].size();
        int output_height = (height - pool_size) / stride + 1;
        int output_width = (width - pool_size) / stride + 1;
        std::vector<std::vector<std::vector<double>>> output(input.size(), std::vector<std::vector<double>>(output_height, std::vector<double>(output_width, 0.0)));

        for (int ic = 0; ic < input.size(); ++ic) {
            for (int i = 0; i < output_height; ++i) {
                for (int j = 0; j < output_width; ++j) {
                    double max_value = -std::numeric_limits<double>::infinity();
                    for (int ki = 0; ki < pool_size; ++ki) {
                        for (int kj = 0; kj < pool_size; ++kj) {
                            int input_row = i * stride + ki;
                            int input_col = j * stride + kj;
                            if (input_row < height && input_col < width) {
                                max_value = std::max(max_value, input[ic][input_row][input_col]);
                            }
                        }
                    }
                    output[ic][i][j] = max_value;
                }
            }
        }
        return output;
    }

private:
    int pool_size;
    int stride;
};

// обратный свёрточный слой
class DeconvLayer {
public:
    DeconvLayer(int in_channels, int out_channels, int kernel_size, int stride = 1, int padding = 0)
        : in_channels(in_channels), out_channels(out_channels), kernel_size(kernel_size), stride(stride), padding(padding) {
        initializeWeights();
    }

    std::vector<std::vector<std::vector<double>>> forward(const std::vector<std::vector<std::vector<double>>>& input) {
        int height = input[0].size();
        int width = input[0][0].size();
        int output_height = (height - 1) * stride - 2 * padding + kernel_size;
        int output_width = (width - 1) * stride - 2 * padding + kernel_size;
        std::vector<std::vector<std::vector<double>>> output(out_channels, std::vector<std::vector<double>>(output_height, std::vector<double>(output_width, 0.0)));

        for (int oc = 0; oc < out_channels; ++oc) {
            for (int ic = 0; ic < in_channels; ++ic) {
                for (int i = 0; i < height; ++i) {
                    for (int j = 0; j < width; ++j) {
                        for (int ki = 0; ki < kernel_size; ++ki) {
                            for (int kj = 0; kj < kernel_size; ++kj) {
                                int output_row = i * stride + ki - padding;
                                int output_col = j * stride + kj - padding;
                                if (output_row >= 0 && output_row < output_height && output_col >= 0 && output_col < output_width) {
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

private:
    int in_channels;
    int out_channels;
    int kernel_size;
    int stride;
    int padding;
    std::vector<std::vector<std::vector<std::vector<double>>>> weights;

    void initializeWeights() {
        weights.resize(out_channels, std::vector<std::vector<std::vector<double>>>(in_channels, std::vector<std::vector<double>>(kernel_size, std::vector<double>(kernel_size))));
        for (int i = 0; i < out_channels; ++i) {
            for (int j = 0; j < in_channels; ++j) {
                for (int k = 0; k < kernel_size; ++k) {
                    for (int l = 0; l < kernel_size; ++l) {
                        weights[i][j][k][l] = ((double) rand() / (RAND_MAX));
                    }
                }
            }
        }
    }
};

// U-Net архитектура
class UNet {
public:
    UNet() : conv1(3, 64, 3), conv2(64, 128, 3), pool1(2, 2), pool2(2, 2), deconv1(128, 64, 2, 2), deconv2(64, 1, 2, 2) {}

    std::vector<std::vector<std::vector<double>>> forward(const std::vector<std::vector<std::vector<double>>>& input) {
        auto x1 = conv1.forward(input);
        x1 = relu.forward(x1);
        auto x2 = pool1.forward(x1);

        auto x3 = conv2.forward(x2);
        x3 = relu.forward(x3);
        auto x4 = pool2.forward(x3);

        auto x5 = deconv1.forward(x4);
        x5 = relu.forward(x5);

        auto output = deconv2.forward(x5);
        return output;
    }

private:
    ConvLayer conv1, conv2;
    ReLU relu;
    MaxPooling pool1, pool2;
    DeconvLayer deconv1, deconv2;
};

// функция для нормализации изображения
std::vector<std::vector<std::vector<double>>> normalize(const cv::Mat& image) {
    std::vector<std::vector<std::vector<double>>> normalized_image(3, std::vector<std::vector<double>>(image.rows, std::vector<double>(image.cols)));
    for (int c = 0; c < 3; ++c) {
        for (int i = 0; i < image.rows; ++i) {
            for (int j = 0; j < image.cols; ++j) {
                normalized_image[c][i][j] = image.at<cv::Vec3b>(i, j)[c] / 255.0;
            }
        }
    }
    return normalized_image;
}

int main() {
    cv::Mat image = cv::imread("/home/zigork/GitHub/carla_hd/train/images/image_0.png", cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Could not open or find the image!" << std::endl;
        return -1;
    }

    std::vector<std::vector<std::vector<double>>> input_image = normalize(image);

    UNet unet;

    auto output = unet.forward(input_image);

    // Вывод результатов
    for (const auto& channel : output) {
        for (const auto& row : channel) {
            for (const auto& value : row) {
                std::cout << value << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    return 0;
}

