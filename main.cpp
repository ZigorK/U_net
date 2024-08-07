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

    // Вызов функции trainUNet вместо ручного цикла обучения
    trainUNet(unet, image_files, mask_files, 10, optimizer);
 
    std::cout << "Обучение завершено." << std::endl; 
 
    return 0; 
}
