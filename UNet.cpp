#include "UNet.h"

double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

UNet::UNet()
    : conv1(3, 64, 3), conv2(64, 128, 3), conv3(128, 256, 3), conv4(256, 512, 3), conv5(512, 1024, 3),
      deconv1(1024, 512, 2, 2), deconv2(1024, 256, 2, 2), deconv3(512, 128, 2, 2), deconv4(256, 64, 2, 2),
      output_conv(128, 1, 1), relu(), maxpool(2, 2), concat() {}

std::vector<std::vector<std::vector<double>>> UNet::forward(const std::vector<std::vector<std::vector<double>>>& input) {
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

    for (size_t i = 0; i < output.size(); ++i) {
        for (size_t j = 0; j < output[i].size(); ++j) {
            for (size_t k = 0; k < output[i][j].size(); ++k) {
                output[i][j][k] = sigmoid(output[i][j][k]);
            }
        }
    }
    
    return output;
}
