#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "UNet.h" 

std::vector<std::vector<std::vector<double>>> loadImage(const std::string& filename);
std::vector<std::vector<std::vector<double>>> loadMask(const std::string& filename);
double binaryCrossEntropy(const std::vector<std::vector<std::vector<double>>>& prediction, const std::vector<std::vector<std::vector<double>>>& target);
std::vector<std::vector<std::vector<double>>> trimToMatchSize(const std::vector<std::vector<std::vector<double>>>& input, const std::vector<std::vector<std::vector<double>>>& target);
std::vector<std::vector<std::vector<double>>> convertToSingleChannel(const std::vector<std::vector<std::vector<double>>>& multiChannelData);

void trainUNet(UNet& model, const std::vector<std::string>& image_files, const std::vector<std::string>& mask_files, int epochs, double learning_rate);

#endif // UTILS_H
