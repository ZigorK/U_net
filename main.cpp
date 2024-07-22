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
    ConvLayer(int in_channels, int out_channels, int kernel_size, int stride = 1, int padding = 1)
        : in_channels(in_channels), out_channels(out_channels), kernel_size(kernel_size), stride(stride), padding(padding) {
        initializeWeights();
    }

    std::vector<std::vector<std::vector<double>>> forward(const std::vector<std::vector<std::vector<double>>>& input) {
        this->input = input;
        int height = input[0].size();
        int width = input[0][0].size();

        int output_height = (height - kernel_size + 2 * padding) / stride + 1;
        int output_width = (width - kernel_size + 2 * padding) / stride + 1;

        if (output_height <= 0 || output_width <= 0) {
            std::cerr << "Ошибка: Размеры выходного тензора равны нулю или отрицательны. Проверьте параметры свертки." << std::endl;
            std::cerr << "Параметры: height = " << height << ", width = " << width << ", kernel_size = " << kernel_size << ", stride = " << stride << ", padding = " << padding << std::endl;
            return {};
        }

        std::vector<std::vector<std::vector<double>>> output(out_channels, std::vector<std::vector<double>>(output_height, std::vector<double>(output_width, 0.0)));

        std::cout << "Начало операции свертки." << std::endl;
        std::cout << "Размеры входа: [" << in_channels << ", " << height << ", " << width << "]" << std::endl;
        std::cout << "Размеры выхода: [" << out_channels << ", " << output_height << ", " << output_width << "]" << std::endl;

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
            std::cout << "Обработка выходного канала " << oc + 1 << "/" << out_channels << " завершена." << std::endl;
        }
        std::cout << "Завершение операции свертки." << std::endl;

        std::cout << "Проверка корректности выходных данных:" << std::endl;
        for (const auto& oc : output) {
            for (const auto& row : oc) {
                for (double val : row) {
                    if (std::isnan(val) || std::isinf(val)) {
                        std::cerr << "Обнаружено некорректное значение в выходном тензоре." << std::endl;
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
    std::vector<std::vector<std::vector<double>>> input;

    void initializeWeights() {
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
    }
};




// функция ReLU
class ReLU { 
public: 
    std::vector<std::vector<std::vector<double>>> forward(const std::vector<std::vector<std::vector<double>>>& input) { 
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
 
    std::vector<std::vector<std::vector<double>>> backward(const std::vector<std::vector<std::vector<double>>>& grad_output) { 
        std::cout << "Начало обратного прохода ReLU." << std::endl; 
        std::cout << "Размеры градиента выхода: [" << grad_output.size() << ", " << (grad_output.empty() ? 0 : grad_output[0].size()) << ", " << (grad_output.empty() || grad_output[0].empty() ? 0 : grad_output[0][0].size()) << "]" << std::endl; 
 
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
         
        std::cout << "Размеры градиента входа: [" << grad_input.size() << ", " << (grad_input.empty() ? 0 : grad_input[0].size()) << ", " << (grad_input.empty() || grad_input[0].empty() ? 0 : grad_input[0][0].size()) << "]" << std::endl; 
        std::cout << "Завершение обратного прохода ReLU." << std::endl; 
 
        return grad_input; 
    } 
 
private: 
    std::vector<std::vector<std::vector<double>>> input; 
};

// объединяющий слой MaxPooling
class MaxPooling {
public:
    MaxPooling(int kernel_size, int stride)
        : kernel_size(kernel_size), stride(stride) {}

    std::vector<std::vector<std::vector<double>>> forward(const std::vector<std::vector<std::vector<double>>>& input) {
        int num_channels = input.size();
        int input_height = input[0].size();
        int input_width = input[0][0].size();

        std::cout << "Начало операции MaxPooling." << std::endl;
        std::cout << "Размеры входа: [" << num_channels << ", " << input_height << ", " << input_width << "]" << std::endl;

        int output_height = (input_height - kernel_size) / stride + 1;
        int output_width = (input_width - kernel_size) / stride + 1;

        // Проверка корректности размеров
        if (output_height <= 0 || output_width <= 0) {
            throw std::runtime_error("Некорректные размеры выходного тензора в MaxPooling слое");
        }

        std::vector<std::vector<std::vector<double>>> output(num_channels, std::vector<std::vector<double>>(output_height, std::vector<double>(output_width, 0.0)));

        for (int channel = 0; channel < num_channels; ++channel) {
            for (int i = 0; i < output_height; ++i) {
                for (int j = 0; j < output_width; ++j) {
                    double max_val = -std::numeric_limits<double>::infinity();
                    for (int ki = 0; ki < kernel_size; ++ki) {
                        for (int kj = 0; kj < kernel_size; ++kj) {
                            int input_row = i * stride + ki;
                            int input_col = j * stride + kj;
                            if (input_row < input_height && input_col < input_width) {
                                max_val = std::max(max_val, input[channel][input_row][input_col]);
                            }
                        }
                    }
                    output[channel][i][j] = max_val;
                }
            }
        }

        std::cout << "Размеры выхода: [" << num_channels << ", " << output_height << ", " << output_width << "]" << std::endl;
        std::cout << "Завершение операции MaxPooling." << std::endl;
        return output;
    }

private:
    int kernel_size;
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
                        weights[i][j][k][l] = static_cast<double>(rand()) / RAND_MAX;
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
        std::cout << "Начало операции объединения (конкатенации)." << std::endl; 
        std::cout << "Размеры входа 1: [" << input1.size() << ", " << (input1.empty() ? 0 : input1[0].size()) << ", " << (input1.empty() || input1[0].empty() ? 0 : input1[0][0].size()) << "]" << std::endl; 
        std::cout << "Размеры входа 2: [" << input2.size() << ", " << (input2.empty() ? 0 : input2[0].size()) << ", " << (input2.empty() || input2[0].empty() ? 0 : input2[0][0].size()) << "]" << std::endl; 
 
        std::vector<std::vector<std::vector<double>>> output = input1; 
        output.insert(output.end(), input2.begin(), input2.end()); 
 
        std::cout << "Размеры выхода: [" << output.size() << ", " << (output.empty() ? 0 : output[0].size()) << ", " << (output.empty() || output[0].empty() ? 0 : output[0][0].size()) << "]" << std::endl; 
        std::cout << "Завершение операции объединения." << std::endl; 
 
        return output; 
    } 
 
    std::vector<std::vector<std::vector<std::vector<double>>>> backward(const std::vector<std::vector<std::vector<double>>>& grad_output, size_t split_index) { 
        std::vector<std::vector<std::vector<double>>> grad_input1(grad_output.begin(), grad_output.begin() + split_index); 
        std::vector<std::vector<std::vector<double>>> grad_input2(grad_output.begin() + split_index, grad_output.end()); 
        return {grad_input1, grad_input2}; 
    } 
};

// Определение сигмоидной функции
double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

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
        std::cout << "Размеры после conv1: [" << x1.size() << ", " << x1[0].size() << ", " << x1[0][0].size() << "]" << std::endl;
        auto x2 = maxpool.forward(x1);
        auto x3 = relu.forward(conv2.forward(x2));
        std::cout << "Размеры после conv2: [" << x3.size() << ", " << x3[0].size() << ", " << x3[0][0].size() << "]" << std::endl;
        auto x4 = maxpool.forward(x3);
        auto x5 = relu.forward(conv3.forward(x4));
        std::cout << "Размеры после conv3: [" << x5.size() << ", " << x5[0].size() << ", " << x5[0][0].size() << "]" << std::endl;
        auto x6 = maxpool.forward(x5);
        auto x7 = relu.forward(conv4.forward(x6));
        std::cout << "Размеры после conv4: [" << x7.size() << ", " << x7[0].size() << ", " << x7[0][0].size() << "]" << std::endl;
        auto x8 = maxpool.forward(x7);
        auto x9 = relu.forward(conv5.forward(x8));
        std::cout << "Размеры после conv5: [" << x9.size() << ", " << x9[0].size() << ", " << x9[0][0].size() << "]" << std::endl;
        auto x10 = relu.forward(deconv1.forward(x9));
        std::cout << "Размеры после deconv1: [" << x10.size() << ", " << x10[0].size() << ", " << x10[0][0].size() << "]" << std::endl;
        auto x11 = concat.forward(x10, x7);
        std::cout << "Размеры после concat1: [" << x11.size() << ", " << x11[0].size() << ", " << x11[0][0].size() << "]" << std::endl;
        auto x12 = relu.forward(conv4.forward(x11));
        std::cout << "Размеры после conv4_1: [" << x12.size() << ", " << x12[0].size() << ", " << x12[0][0].size() << "]" << std::endl;
        auto x13 = relu.forward(deconv2.forward(x12));
        std::cout << "Размеры после deconv2: [" << x13.size() << ", " << x13[0].size() << ", " << x13[0][0].size() << "]" << std::endl;
        auto x14 = concat.forward(x13, x5);
        std::cout << "Размеры после concat2: [" << x14.size() << ", " << x14[0].size() << ", " << x14[0][0].size() << "]" << std::endl;
        auto x15 = relu.forward(conv3.forward(x14));
        std::cout << "Размеры после conv3_1: [" << x15.size() << ", " << x15[0].size() << ", " << x15[0][0].size() << "]" << std::endl;
        auto x16 = relu.forward(deconv3.forward(x15));
        std::cout << "Размеры после deconv3: [" << x16.size() << ", " << x16[0].size() << ", " << x16[0][0].size() << "]" << std::endl;
        auto x17 = concat.forward(x16, x3);
        std::cout << "Размеры после concat3: [" << x17.size() << ", " << x17[0].size() << ", " << x17[0][0].size() << "]" << std::endl;
        auto x18 = relu.forward(conv2.forward(x17));
        std::cout << "Размеры после conv2_1: [" << x18.size() << ", " << x18[0].size() << ", " << x18[0][0].size() << "]" << std::endl;
        auto x19 = relu.forward(deconv4.forward(x18));
        std::cout << "Размеры после deconv4: [" << x19.size() << ", " << x19[0].size() << ", " << x19[0][0].size() << "]" << std::endl;
        auto x20 = concat.forward(x19, x1);
        std::cout << "Размеры после concat4: [" << x20.size() << ", " << x20[0].size() << ", " << x20[0][0].size() << "]" << std::endl;
        auto output = output_conv.forward(x20);
        std::cout << "Размеры выхода: [" << output.size() << ", " << output[0].size() << ", " << output[0][0].size() << "]" << std::endl;

        // Применение сигмоидной функции
        for (size_t i = 0; i < output.size(); ++i) {
            for (size_t j = 0; j < output[i].size(); ++j) {
                for (size_t k = 0; k < output[i][j].size(); ++k) {
                    output[i][j][k] = sigmoid(output[i][j][k]);
                }
            }
        }
        
        return output;
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
    if (prediction.size() != target.size() || 
        prediction[0].size() != target[0].size() || 
        prediction[0][0].size() != target[0][0].size()) {
        std::cerr << "Ошибка: Размеры предсказаний и целевых данных не совпадают." << std::endl;
        std::cerr << "Размеры предсказаний: [" << prediction.size() << ", " << prediction[0].size() << ", " << prediction[0][0].size() << "]" << std::endl;
        std::cerr << "Размеры целевых данных: [" << target.size() << ", " << target[0].size() << ", " << target[0][0].size() << "]" << std::endl;
        return -1.0;
    }

    double loss = 0.0;
    for (size_t i = 0; i < prediction.size(); ++i) {
        for (size_t j = 0; j < prediction[i].size(); ++j) {
            for (size_t k = 0; k < prediction[i][j].size(); ++k) {
                double pred = std::min(std::max(prediction[i][j][k], 1e-15), 1.0 - 1e-15);
                double target_val = std::min(std::max(target[i][j][k], 0.0), 1.0); // Убедитесь, что целевые данные в пределах [0, 1]
                double term1 = target_val * std::log(pred);
                double term2 = (1 - target_val) * std::log(1 - pred);
                loss -= term1 + term2;

                // Вывод для отладки
                std::cout << "Prediction[" << i << "][" << j << "][" << k << "]: " << pred << std::endl;
                std::cout << "Target[" << i << "][" << j << "][" << k << "]: " << target_val << std::endl;
                std::cout << "Term1: " << term1 << std::endl;
                std::cout << "Term2: " << term2 << std::endl;
            }
        }
    }
    double total_loss = loss / (prediction.size() * prediction[0].size() * prediction[0][0].size());
    std::cout << "Total Loss: " << total_loss << std::endl;

    return total_loss;
}


std::vector<std::vector<std::vector<double>>> trimToMatchSize(
    const std::vector<std::vector<std::vector<double>>>& input,
    const std::vector<std::vector<std::vector<double>>>& target) {

    size_t depth = std::min(input.size(), target.size());
    size_t height = std::min(input[0].size(), target[0].size());
    size_t width = std::min(input[0][0].size(), target[0][0].size());

    std::vector<std::vector<std::vector<double>>> trimmed(depth, std::vector<std::vector<double>>(height, std::vector<double>(width, 0.0)));
    
    for (size_t i = 0; i < depth; ++i) {
        for (size_t j = 0; j < height; ++j) {
            for (size_t k = 0; k < width; ++k) {
                trimmed[i][j][k] = input[i][j][k];
            }
        }
    }
    
    return trimmed;
}

std::vector<std::vector<std::vector<double>>> convertToSingleChannel(
    const std::vector<std::vector<std::vector<double>>>& multiChannelData) {

    size_t height = multiChannelData[0].size();
    size_t width = multiChannelData[0][0].size();

    std::vector<std::vector<std::vector<double>>> singleChannelData(1, std::vector<std::vector<double>>(height, std::vector<double>(width, 0.0)));

    for (size_t i = 0; i < height; ++i) {
        for (size_t j = 0; j < width; ++j) {
            singleChannelData[0][i][j] = multiChannelData[0][i][j];
        }
    }

    return singleChannelData;
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
    std::string masks_path = "/home/zigork/GitHub/cifar10_images/1"; 
 
    std::vector<std::string> image_files; 
    std::vector<std::string> mask_files; 
 
    for (const auto& entry : fs::directory_iterator(images_path)) { 
        image_files.push_back(entry.path().string()); 
    } 
    for (const auto& entry : fs::directory_iterator(masks_path)) { 
        mask_files.push_back(entry.path().string()); 
    } 
 
    size_t num_images = std::min(image_files.size(), mask_files.size()); 
    num_images = std::min(num_images, static_cast<size_t>(10)); 
 
    UNet unet; 
 
    std::cout << "Начало обучения модели UNet..." << std::endl; 
 
    for (size_t epoch = 0; epoch < 10; ++epoch) { 
        double total_loss = 0.0; 
        std::cout << "Epoch [" << epoch + 1 << "/10]" << std::endl; 
 
        for (size_t i = 0; i < num_images; ++i) { 
            std::cout << "Processing image " << i + 1 << "/" << num_images << std::endl; 
 
            auto input = loadImage(image_files[i]);
            auto target = loadMask(mask_files[i]);

            // Преобразуем маски в один канал, если необходимо
            if (target.size() > 1) {
                target = convertToSingleChannel(target);
            }

            // Проверка размеров перед вызовом forward 
            std::cout << "Input tensor size: [" << input.size() << ", " << input[0].size() << ", " << input[0][0].size() << "]" << std::endl; 
            std::cout << "Mask tensor size: [" << target.size() << ", " << target[0].size() << ", " << target[0][0].size() << "]" << std::endl; 
 
            auto prediction = unet.forward(input);
 
            // Проверка размеров после forward 
            std::cout << "Prediction tensor size: [" << prediction.size() << ", " << prediction[0].size() << ", " << prediction[0][0].size() << "]" << std::endl;

            if (prediction.size() != target.size() || 
                prediction[0].size() != target[0].size() || 
                prediction[0][0].size() != target[0][0].size()) {
                std::cerr << "Ошибка: Размеры предсказаний и целевых данных не совпадают." << std::endl;
                std::cerr << "Размеры предсказаний: [" << prediction.size() << ", " << prediction[0].size() << ", " << prediction[0][0].size() << "]" << std::endl;
                std::cerr << "Размеры целевых данных: [" << target.size() << ", " << target[0].size() << ", " << target[0][0].size() << "]" << std::endl;

                prediction = trimToMatchSize(prediction, target);
                std::cout << "Размеры предсказаний после обрезки: [" << prediction.size() << ", " << prediction[0].size() << ", " << prediction[0][0].size() << "]" << std::endl;
            }
 
            double loss = binaryCrossEntropy(prediction, target); 
            if (loss < 0) {
                std::cerr << "Произошла ошибка при вычислении binaryCrossEntropy." << std::endl;
                }
            total_loss += loss; 
 
            std::cout << "Image " << i + 1 << " processed. Loss: " << loss << std::endl; 
        } 
        std::cout << "Epoch [" << epoch + 1 << "/10], Loss: " << total_loss / num_images << std::endl; 
    } 
 
    std::cout << "Обучение завершено." << std::endl; 
 
    return 0; 
}

