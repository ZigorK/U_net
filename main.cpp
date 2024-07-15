#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <algorithm> 
#include <filesystem>  

namespace fs = std::filesystem;

// Функция для получения всех файлов в директории
std::vector<std::string> getFilesInDirectory(const std::string& directory) {
    std::vector<std::string> files;
    for (const auto& entry : fs::directory_iterator(directory)) {
        files.push_back(entry.path().string());
    }
    return files;
}

// свёрточный слой
class ConvLayer {
public:
    ConvLayer(int in_channels, int out_channels, int kernel_size, int stride = 1, int padding = 0)
        : in_channels(in_channels), out_channels(out_channels), kernel_size(kernel_size), stride(stride), padding(padding) {
        initializeWeights();
    }

    std::vector<std::vector<std::vector<double>>> forward(const std::vector<std::vector<std::vector<double>>>& input) {
        this->input = input;
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

    void backward(const std::vector<std::vector<std::vector<double>>>& grad_output, double learning_rate) {
        int height = input[0].size();
        int width = input[0][0].size();
        int output_height = grad_output[0].size();
        int output_width = grad_output[0][0].size();

        std::vector<std::vector<std::vector<std::vector<double>>>> grad_weights(out_channels, std::vector<std::vector<std::vector<double>>>(in_channels, std::vector<std::vector<double>>(kernel_size, std::vector<double>(kernel_size, 0.0))));
        std::vector<std::vector<std::vector<double>>> grad_input(in_channels, std::vector<std::vector<double>>(height, std::vector<double>(width, 0.0)));

        for (int oc = 0; oc < out_channels; ++oc) {
            for (int ic = 0; ic < in_channels; ++ic) {
                for (int i = 0; i < output_height; ++i) {
                    for (int j = 0; j < output_width; ++j) {
                        for (int ki = 0; ki < kernel_size; ++ki) {
                            for (int kj = 0; kj < kernel_size; ++kj) {
                                int input_row = i * stride + ki - padding;
                                int input_col = j * stride + kj - padding;
                                if (input_row >= 0 && input_row < height && input_col >= 0 && input_col < width) {
                                    grad_weights[oc][ic][ki][kj] += grad_output[oc][i][j] * input[ic][input_row][input_col];
                                    grad_input[ic][input_row][input_col] += grad_output[oc][i][j] * weights[oc][ic][ki][kj];
                                }
                            }
                        }
                    }
                }
            }
        }

        // Обновление весов
        for (int oc = 0; oc < out_channels; ++oc) {
            for (int ic = 0; ic < in_channels; ++ic) {
                for (int ki = 0; ki < kernel_size; ++ki) {
                    for (int kj = 0; kj < kernel_size; ++kj) {
                        weights[oc][ic][ki][kj] -= learning_rate * grad_weights[oc][ic][ki][kj];
                    }
                }
            }
        }
    }

private:
    int in_channels;
    int out_channels;
    int kernel_size;
    int stride;
    int padding;
    std::vector<std::vector<std::vector<std::vector<double>>>> weights;
    std::vector<std::vector<std::vector<double>>> input;

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
        this->input = input;
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

    std::vector<std::vector<std::vector<double>>> backward(const std::vector<std::vector<std::vector<double>>>& grad_output) {
        std::vector<std::vector<std::vector<double>>> grad_input = grad_output;
        for (size_t i = 0; i < input.size(); ++i) {
            for (size_t j = 0; j < input[i].size(); ++j) {
                for (size_t k = 0; k < input[i][j].size(); ++k) {
                    if (input[i][j][k] <= 0) {
                        grad_input[i][j][k] = 0;
                    }
                }
            }
        }
        return grad_input;
    }

private:
    std::vector<std::vector<std::vector<double>>> input;
};

// объединяющий слой MaxPooling
class MaxPooling {
public:
    MaxPooling(int pool_size, int stride)
        : pool_size(pool_size), stride(stride) {}

    std::vector<std::vector<std::vector<double>>> forward(const std::vector<std::vector<std::vector<double>>>& input) {
        this->input = input;
        int height = input[0].size();
        int width = input[0][0].size();
        int output_height = (height - pool_size) / stride + 1;
        int output_width = (width - pool_size) / stride + 1;
        std::vector<std::vector<std::vector<double>>> output(input.size(), std::vector<std::vector<double>>(output_height, std::vector<double>(output_width, 0.0)));

        max_indices.resize(input.size(), std::vector<std::vector<std::pair<int, int>>>(output_height, std::vector<std::pair<int, int>>(output_width)));

        for (int ic = 0; ic < input.size(); ++ic) {
            for (int i = 0; i < output_height; ++i) {
                for (int j = 0; j < output_width; ++j) {
                    double max_value = -std::numeric_limits<double>::infinity();
                    for (int ki = 0; ki < pool_size; ++ki) {
                        for (int kj = 0; kj < pool_size; ++kj) {
                            int input_row = i * stride + ki;
                            int input_col = j * stride + kj;
                            if (input_row < height && input_col < width) {
                                if (input[ic][input_row][input_col] > max_value) {
                                    max_value = input[ic][input_row][input_col];
                                    max_indices[ic][i][j] = {input_row, input_col};
                                }
                            }
                        }
                    }
                    output[ic][i][j] = max_value;
                }
            }
        }
        return output;
    }

    std::vector<std::vector<std::vector<double>>> backward(const std::vector<std::vector<std::vector<double>>>& grad_output) {
        std::vector<std::vector<std::vector<double>>> grad_input(input.size(), std::vector<std::vector<double>>(input[0].size(), std::vector<double>(input[0][0].size(), 0.0)));

        for (int ic = 0; ic < grad_output.size(); ++ic) {
            for (int i = 0; i < grad_output[0].size(); ++i) {
                for (int j = 0; j < grad_output[0][0].size(); ++j) {
                    auto [input_row, input_col] = max_indices[ic][i][j];
                    grad_input[ic][input_row][input_col] = grad_output[ic][i][j];
                }
            }
        }
        return grad_input;
    }

private:
    int pool_size;
    int stride;
    std::vector<std::vector<std::vector<double>>> input;
    std::vector<std::vector<std::vector<std::pair<int, int>>>> max_indices;
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

    void backward(const std::vector<std::vector<std::vector<double>>>& input, const std::vector<std::vector<std::vector<double>>>& grad_output, double learning_rate) {
        // Рассчитываем градиенты и обновляем веса
        // Реализуем обратный проход для сверточного слоя
        // Обновляем веса на основе градиентов и скорости обучения
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

// слой объединения (конкатенации)
class ConcatLayer {
public:
    std::vector<std::vector<std::vector<double>>> forward(const std::vector<std::vector<std::vector<double>>>& input1, const std::vector<std::vector<std::vector<double>>>& input2) {
        std::vector<std::vector<std::vector<double>>> output = input1;
        output.insert(output.end(), input2.begin(), input2.end());
        return output;
    }

    std::vector<std::vector<std::vector<std::vector<double>>>> backward(const std::vector<std::vector<std::vector<double>>>& grad_output, size_t split_index) {
        std::vector<std::vector<std::vector<double>>> grad_input1(grad_output.begin(), grad_output.begin() + split_index);
        std::vector<std::vector<std::vector<double>>> grad_input2(grad_output.begin() + split_index, grad_output.end());
        return {grad_input1, grad_input2};
    }
};

// определение модели UNet
class UNet {
public:
    UNet()
        : conv1(3, 64, 3), conv2(64, 128, 3), conv3(128, 256, 3), conv4(256, 512, 3), conv5(512, 1024, 3),
          deconv1(1024, 512, 2, 2), deconv2(1024, 256, 2, 2), deconv3(512, 128, 2, 2), deconv4(256, 64, 2, 2),
          output_conv(128, 1, 1),
          relu(), maxpool(2, 2), concat() {}

    std::vector<std::vector<std::vector<double>>> forward(const std::vector<std::vector<std::vector<double>>>& input) {
        auto x1 = relu.forward(conv1.forward(input));
        auto x2 = maxpool.forward(x1);
        auto x3 = relu.forward(conv2.forward(x2));
        auto x4 = maxpool.forward(x3);
        auto x5 = relu.forward(conv3.forward(x4));
        auto x6 = maxpool.forward(x5);
        auto x7 = relu.forward(conv4.forward(x6));
        auto x8 = maxpool.forward(x7);
        auto x9 = relu.forward(conv5.forward(x8));
        auto x10 = relu.forward(deconv1.forward(x9));
        auto x11 = concat.forward(x10, x7);
        auto x12 = relu.forward(conv4.forward(x11));
        auto x13 = relu.forward(deconv2.forward(x12));
        auto x14 = concat.forward(x13, x5);
        auto x15 = relu.forward(conv3.forward(x14));
        auto x16 = relu.forward(deconv3.forward(x15));
        auto x17 = concat.forward(x16, x3);
        auto x18 = relu.forward(conv2.forward(x17));
        auto x19 = relu.forward(deconv4.forward(x18));
        auto x20 = concat.forward(x19, x1);
        return output_conv.forward(x20);
    }

private:
    ConvLayer conv1, conv2, conv3, conv4, conv5;
    DeconvLayer deconv1, deconv2, deconv3, deconv4;
    ConvLayer output_conv;
    ReLU relu;
    MaxPooling maxpool;
    ConcatLayer concat;
};

// функция загрузки изображений
std::vector<std::vector<std::vector<double>>> loadImage(const std::string& filename) {
    cv::Mat image = cv::imread(filename);
    image.convertTo(image, CV_64FC3, 1.0 / 255.0);
    std::vector<std::vector<std::vector<double>>> data(3, std::vector<std::vector<double>>(image.rows, std::vector<double>(image.cols)));
    for (int i = 0; i < image.rows; ++i) {
        for (int j = 0; j < image.cols; ++j) {
            data[0][i][j] = image.at<cv::Vec3d>(i, j)[0];
            data[1][i][j] = image.at<cv::Vec3d>(i, j)[1];
            data[2][i][j] = image.at<cv::Vec3d>(i, j)[2];
        }
    }
    return data;
}

// функция загрузки масок
std::vector<std::vector<std::vector<double>>> loadMask(const std::string& filename) {
    cv::Mat image = cv::imread(filename, cv::IMREAD_COLOR); // Загружаем как цветное изображение
    image.convertTo(image, CV_64FC3, 1.0 / 255.0);
    std::vector<std::vector<std::vector<double>>> data(3, std::vector<std::vector<double>>(image.rows, std::vector<double>(image.cols)));
    for (int i = 0; i < image.rows; ++i) {
        for (int j = 0; j < image.cols; ++j) {
            data[0][i][j] = image.at<cv::Vec3d>(i, j)[0]; // Канал Blue
            data[1][i][j] = image.at<cv::Vec3d>(i, j)[1]; // Канал Green
            data[2][i][j] = image.at<cv::Vec3d>(i, j)[2]; // Канал Red
        }
    }
    return data;
}


// Функция бинарной кросс-энтропии с выводами для отладки
double binaryCrossEntropy(const std::vector<std::vector<std::vector<double>>>& prediction, const std::vector<std::vector<std::vector<double>>>& target) {
    double loss = 0.0;
    for (size_t i = 0; i < prediction.size(); ++i) {
        for (size_t j = 0; j < prediction[i].size(); ++j) {
            for (size_t k = 0; k < prediction[i][j].size(); ++k) {
                double pred = std::min(std::max(prediction[i][j][k], 1e-15), 1.0 - 1e-15);
                double term1 = target[i][j][k] * std::log(pred);
                double term2 = (1 - target[i][j][k]) * std::log(1 - pred);
                loss -= term1 + term2;

                // Вывод для отладки
                std::cout << "Prediction[" << i << "][" << j << "][" << k << "]: " << pred << std::endl;
                std::cout << "Target[" << i << "][" << j << "][" << k << "]: " << target[i][j][k] << std::endl;
                std::cout << "Term1: " << term1 << std::endl;
                std::cout << "Term2: " << term2 << std::endl;
            }
        }
    }
    double total_loss = loss / (prediction.size() * prediction[0].size() * prediction[0][0].size());
    std::cout << "Total Loss: " << total_loss << std::endl;

    return total_loss;
}


// обучение модели
void trainUNet(UNet& model, const std::vector<std::string>& image_files, const std::vector<std::string>& mask_files, int epochs, double learning_rate) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        double epoch_loss = 0.0;
        std::cout << "Эпоха [" << (epoch + 1) << "/" << epochs << "]" << std::endl;
        for (size_t i = 0; i < image_files.size(); ++i) {
            std::cout << "Обработка изображения " << i + 1 << "/" << image_files.size() << std::endl;
            auto input = loadImage(image_files[i]);
            auto target = loadMask(mask_files[i]);
            auto output = model.forward(input);
            double loss = binaryCrossEntropy(output, target);
            epoch_loss += loss;
            std::cout << "Изображение " << i + 1 << " обработано. Потери: " << loss << std::endl;

            // Обратное распространение ошибки и обновление весов
            // TODO: Реализовать обратный проход и обновление весов
        }
        std::cout << "Эпоха [" << (epoch + 1) << "/" << epochs << "], Потери: " << epoch_loss / image_files.size() << std::endl;
    }
}



int main() {
    std::string images_path = "/home/zigork/GitHub/cifar10_images/0";
    std::string masks_path = "/home/zigork/GitHub/cifar10_images/0";

    std::vector<std::string> image_files;
    std::vector<std::string> mask_files;

    for (const auto& entry : fs::directory_iterator(images_path)) {
        image_files.push_back(entry.path().string());
    }
    for (const auto& entry : fs::directory_iterator(masks_path)) {
        mask_files.push_back(entry.path().string());
    }

    // Ограничим количество обрабатываемых изображений
    size_t num_images = std::min(image_files.size(), mask_files.size());
    num_images = std::min(num_images, static_cast<size_t>(10)); // Ограничим до 10 изображений для отладки

    UNet unet;

    std::cout << "Начало обучения модели UNet..." << std::endl;
    
    for (size_t epoch = 0; epoch < 10; ++epoch) {
        double total_loss = 0.0;
        std::cout << "Epoch [" << epoch + 1 << "/10]" << std::endl;
        
        for (size_t i = 0; i < num_images; ++i) {
            std::cout << "Processing image " << i + 1 << "/" << num_images << std::endl;

            // Загрузка изображения
            cv::Mat image = cv::imread(image_files[i], cv::IMREAD_COLOR);
            if (image.empty()) {
                std::cerr << "Failed to load image: " << image_files[i] << std::endl;
                continue;
            }
            image.convertTo(image, CV_64FC3, 1.0 / 255.0);
            std::vector<std::vector<std::vector<double>>> image_data(3, std::vector<std::vector<double>>(image.rows, std::vector<double>(image.cols)));
            for (int r = 0; r < image.rows; ++r) {
                for (int c = 0; c < image.cols; ++c) {
                    for (int ch = 0; ch < 3; ++ch) {
                        image_data[ch][r][c] = image.at<cv::Vec3d>(r, c)[ch];
                    }
                }
            }

            // Загрузка маски
            cv::Mat mask = cv::imread(mask_files[i], cv::IMREAD_GRAYSCALE);
            if (mask.empty()) {
                std::cerr << "Failed to load mask: " << mask_files[i] << std::endl;
                continue;
            }
            mask.convertTo(mask, CV_64FC1, 1.0 / 255.0);
            std::vector<std::vector<std::vector<double>>> mask_data(1, std::vector<std::vector<double>>(mask.rows, std::vector<double>(mask.cols)));
            for (int r = 0; r < mask.rows; ++r) {
                for (int c = 0; c < mask.cols; ++c) {
                    mask_data[0][r][c] = mask.at<double>(r, c);
                }
            }

            // Прямое распространение
            auto prediction = unet.forward(image_data);

            // Вычисление ошибки
            double loss = binaryCrossEntropy(prediction, mask_data);
            total_loss += loss;

            std::cout << "Image " << i + 1 << " processed. Loss: " << loss << std::endl;
        }
        std::cout << "Epoch [" << epoch + 1 << "/10], Loss: " << total_loss / num_images << std::endl;
    }

    std::cout << "Обучение завершено." << std::endl;

    return 0;
}
