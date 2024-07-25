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
    auto output = output_conv.forward(x20);

    for (size_t i = 0; i < output.size(); ++i) {
        for (size_t j = 0; j < output[i].size(); ++j) {
            for (size_t k = 0; k < output[i][j].size(); ++k) {
                output[i][j][k] = sigmoid(output[i][j][k]);
            }
        }
    }

    return output;
}

void UNet::backward(const std::vector<std::vector<std::vector<double>>>& grad_output) {
    auto grad_output_conv = output_conv.backward(grad_output); 

    auto grad_deconv4 = deconv4.backward(grad_output_conv); 
    auto [grad_x19, grad_x1] = concat.backward(grad_deconv4); 
    auto grad_conv2_deconv3 = conv2.backward(grad_x19); 
    auto grad_deconv3 = deconv3.backward(grad_conv2_deconv3); 
    auto [grad_x18, grad_x3] = concat.backward(grad_deconv3); 
    auto grad_conv3_deconv2 = conv3.backward(grad_x18); 
    auto grad_deconv2 = deconv2.backward(grad_conv3_deconv2); 
    auto [grad_x17, grad_x5] = concat.backward(grad_deconv2); 
    auto grad_conv4_deconv1 = conv4.backward(grad_x17); 
    auto grad_deconv1 = deconv1.backward(grad_conv4_deconv1); 
    auto [grad_x16, grad_x7] = concat.backward(grad_deconv1); 
    auto grad_conv5 = conv5.backward(grad_x16); 
    auto grad_maxpool4 = maxpool.backward(grad_conv5); 
    auto grad_conv4 = conv4.backward(grad_maxpool4);
    auto grad_maxpool3 = maxpool.backward(grad_conv4);
    auto grad_conv3 = conv3.backward(grad_maxpool3);
    auto grad_maxpool2 = maxpool.backward(grad_conv3);
    auto grad_conv2 = conv2.backward(grad_maxpool2);
    auto grad_maxpool1 = maxpool.backward(grad_conv2);
    auto grad_conv1 = conv1.backward(grad_maxpool1);

    setGradients(grad_conv1); 
}


void UNet::updateWeights(Optimizer& optimizer) {
    optimizer.updateWeights(conv1.getWeights(), conv1.getGradients());
    optimizer.updateWeights(conv2.getWeights(), conv2.getGradients());
    optimizer.updateWeights(conv3.getWeights(), conv3.getGradients());
    optimizer.updateWeights(conv4.getWeights(), conv4.getGradients());
    optimizer.updateWeights(conv5.getWeights(), conv5.getGradients());
    optimizer.updateWeights(deconv1.getWeights(), deconv1.getGradients());
    optimizer.updateWeights(deconv2.getWeights(), deconv2.getGradients());
    optimizer.updateWeights(deconv3.getWeights(), deconv3.getGradients());
    optimizer.updateWeights(deconv4.getWeights(), deconv4.getGradients());
    optimizer.updateWeights(output_conv.getWeights(), output_conv.getGradients());
}



std::vector<std::vector<std::vector<double>>> UNet::getGradients() const {
    return gradients;
}

void UNet::setGradients(const std::vector<std::vector<std::vector<double>>>& gradients) {
    this->gradients = gradients;
}
