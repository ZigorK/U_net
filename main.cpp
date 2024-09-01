#include <iostream>
#include <vector>
#include <filesystem>  
#include "ConvolutionalLayer.h"
#include "ReLULayer.h"
#include "MaxPooling.h"
#include "DeconvLayer.h"
#include "ConcatLayer.h"
#include "UNet.h"
#include "Utils.h"
#include "SGDOptimizer.h" 

namespace fs = std::filesystem;  

int main() { 
    std::string images_path = "//home/ziigork/Downloads/carla_images/train_rgb"; 
    std::string masks_path = "/home/ziigork/Downloads/carla_images/train_semantic"; 
 
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

    
    trainUNet(unet, image_files, mask_files, 10, optimizer);
 
    std::cout << "Обучение завершено." << std::endl; 
 
    return 0; 
}
