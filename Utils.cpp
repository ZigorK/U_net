#include "Utils.h"
#include "UNet.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <fstream>

// Загрузка изображения и преобразование в формат double
std::vector<std::vector<std::vector<double>>> loadImage(const std::string& filename) {
    cv::Mat image = cv::imread(filename);
    image.convertTo(image, CV_64FC3, 1.0 / 255.0);
    std::vector<std::vector<std::vector<double>>> data(3, std::vector<std::vector<double>>(image.rows, std::vector<double>(image.cols)));
    for (int i = 0; i < image.rows; ++i) {
        for (int j = 0; j < image.cols; ++j) {
            cv::Vec3d intensity = image.at<cv::Vec3d>(i, j);
            data[0][i][j] = intensity[0];
            data[1][i][j] = intensity[1];
            data[2][i][j] = intensity[2];
        }
    }
    return data;
}

// Загрузка маски и преобразование в формат double
std::vector<std::vector<std::vector<double>>> loadMask(const std::string& filename) {
    cv::Mat mask = cv::imread(filename, cv::IMREAD_GRAYSCALE);
    mask.convertTo(mask, CV_64F, 1.0 / 255.0);
    std::vector<std::vector<std::vector<double>>> data(1, std::vector<std::vector<double>>(mask.rows, std::vector<double>(mask.cols)));
    for (int i = 0; i < mask.rows; ++i) {
        for (int j = 0; j < mask.cols; ++j) {
            double intensity = mask.at<double>(i, j);
            data[0][i][j] = intensity;
        }
    }
    return data;
}

// Функция вычисления потерь (Binary Cross Entropy) с дополнительными проверками
double binaryCrossEntropy(const std::vector<std::vector<std::vector<double>>>& prediction, const std::vector<std::vector<std::vector<double>>>& target) {
    double loss = 0.0;
    const double epsilon = 1e-8; // Малое значение для избежания логарифма нуля
    for (size_t i = 0; i < prediction.size(); ++i) {
        for (size_t j = 0; j < prediction[i].size(); ++j) {
            for (size_t k = 0; k < prediction[i][j].size(); ++k) {
                double pred = std::clamp(prediction[i][j][k], epsilon, 1.0 - epsilon); // Ограничение значений предсказаний
                double targ = target[i][j][k];
                loss -= targ * std::log(pred) + (1 - targ) * std::log(1 - pred);
            }
        }
    }
    return loss / (prediction[0].size() * prediction[0][0].size());
}

// Функция обрезки предсказаний до размера целевых данных
std::vector<std::vector<std::vector<double>>> trimToMatchSize(const std::vector<std::vector<std::vector<double>>>& input, const std::vector<std::vector<std::vector<double>>>& target) {
    int inputDepth = input.size();
    int inputHeight = input[0].size();
    int inputWidth = input[0][0].size();

    int targetHeight = target[0].size();
    int targetWidth = target[0][0].size();
    
    // Размеры для обрезки
    int trimHeight = std::min(inputHeight, targetHeight);
    int trimWidth = std::min(inputWidth, targetWidth);

    std::vector<std::vector<std::vector<double>>> trimmed(inputDepth, std::vector<std::vector<double>>(trimHeight, std::vector<double>(trimWidth)));
    for (int c = 0; c < inputDepth; ++c) {
        for (int i = 0; i < trimHeight; ++i) {
            for (int j = 0; j < trimWidth; ++j) {
                trimmed[c][i][j] = input[c][i][j];
            }
        }
    }

    return trimmed;
}


// Функция преобразования данных с несколькими каналами в одноканальные
std::vector<std::vector<std::vector<double>>> convertToSingleChannel(const std::vector<std::vector<std::vector<double>>>& multiChannelData) {
    int height = multiChannelData[0].size();
    int width = multiChannelData[0][0].size();
    std::vector<std::vector<std::vector<double>>> singleChannelData(1, std::vector<std::vector<double>>(height, std::vector<double>(width, 0.0)));
    for (int c = 0; c < multiChannelData.size(); ++c) {
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                singleChannelData[0][i][j] += multiChannelData[c][i][j] / multiChannelData.size();
            }
        }
    }
    return singleChannelData;
}

// Функция обучения модели U-Net
void trainUNet(UNet& model, const std::vector<std::string>& image_files, const std::vector<std::string>& mask_files, int epochs, SGDOptimizer& optimizer) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        double total_loss = 0.0;
        for (size_t i = 0; i < image_files.size(); ++i) {
            auto input = loadImage(image_files[i]);
            auto target = loadMask(mask_files[i]);

            // Преобразуем маски в один канал, если необходимо
            if (target.size() > 1) {
                target = convertToSingleChannel(target);
            }

            auto output = model.forward(input);
            output = trimToMatchSize(output, target);
            double loss = binaryCrossEntropy(output, target);
            total_loss += loss;
            std::cout << "Эпоха: " << epoch + 1 << ", Пример: " << i + 1 << ", Потери: " << loss << std::endl;

            // Обратное распространение и обновление весов
            model.backward(target);
            model.updateWeights(optimizer);
        }
        std::cout << "Эпоха: " << epoch + 1 << ", Средние потери: " << total_loss / image_files.size() << std::endl;

        // Сохранение модели после каждой эпохи
        saveModel(model, "unet_model_epoch_" + std::to_string(epoch + 1) + ".bin");
    }
}

// Функция сохранения модели
void saveModel(const UNet& model, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (file.is_open()) {
        // Пример сохранения параметров модели
        for (const auto& layer : model.getLayers()) {
            layer->save(file);  // Реализуйте метод save() в каждом слое
        }
        file.close();
    } else {
        std::cerr << "Не удалось открыть файл для сохранения модели: " << filename << std::endl;
    }
}

// Функция загрузки модели
void loadModel(UNet& model, const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (file.is_open()) {
        // Пример загрузки параметров модели
        for (auto& layer : model.getLayers()) {
            layer->load(file);  // Реализуйте метод load() в каждом слое
        }
        file.close();
    } else {
        std::cerr << "Не удалось открыть файл для загрузки модели: " << filename << std::endl;
    }
}
