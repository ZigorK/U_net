#include <iostream>
#include <vector>
#include <filesystem>  // Для работы с файловой системой
#include "ConvolutionalLayer.h"
#include "ReLULayer.h"
#include "MaxPooling.h"
#include "DeconvLayer.h"
#include "ConcatLayer.h"
#include "UNet.h"
#include "Utils.h"
#include "SGDOptimizer.h"  // Подключаем заголовочный файл SGDOptimizer

namespace fs = std::filesystem;  // Удобный алиас для namespace filesystem

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
    SGDOptimizer optimizer(0.01);  // Создаем объект SGDOptimizer с выбранной скоростью обучения

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

            // Вычисление градиентов
            unet.backward(target);
            
            // Обновление весов
            unet.updateWeights(optimizer);
        } 
        std::cout << "Epoch [" << epoch + 1 << "/10], Loss: " << total_loss / num_images << std::endl; 
    } 
 
    std::cout << "Обучение завершено." << std::endl; 
 
    return 0; 
}
