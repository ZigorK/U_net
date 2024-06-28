#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>

using namespace cv;
using namespace cv::dnn;
using namespace std;

Net createUNet() {
    Net net = readNetFromONNX("unet_model.onnx");
    return net;
}

Mat preprocess(const Mat& img) {
    Mat blob;
    blobFromImage(img, blob, 1.0/255.0, Size(256, 256), Scalar(), true, false);
    return blob;
}

Mat postprocess(const Mat& output) {
    Mat result;
    // Convert the output to a format suitable for visualization
    normalize(output, result, 0, 255, NORM_MINMAX);
    result.convertTo(result, CV_8U);
    return result;
}

int main() {
    // Load the U-Net model
    Net net = createUNet();

    // Load an image
    Mat img = imread("image.jpg");
    if (img.empty()) {
        cout << "Could not read the image" << endl;
        return 1;
    }

    // Preprocess the image
    Mat inputBlob = preprocess(img);

    // Set the network input
    net.setInput(inputBlob);

    // Perform forward pass
    Mat outputBlob = net.forward();

    // Postprocess the output
    Mat result = postprocess(outputBlob);

    // Display the result
    imshow("Input", img);
    imshow("Output", result);
    waitKey(0);

    return 0;
}
