#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
using namespace std;
using namespace cv;
extern "C" {
    int GetSegnetMap(int*** arr, int dim1, int dim2, int dim3)
    {
        // 创建灰度图像
        Mat matt(dim2, dim3, CV_8UC1);

        // 将 arr 中的数据复制到 matt 中
        for (int i = 0; i < dim2; i++) {
            for (int j = 0; j < dim3; j++) {
                matt.at<uchar>(i, j) = static_cast<uchar>(arr[0][i][j]);
            }
        }

        // 缩放图像大小到1920x1080
        Mat resizedMatt;
        resize(matt, resizedMatt, Size(1920, 1080), 0, 0, INTER_NEAREST);

        // 将灰度图像转换为热力图
        Mat heatmap;
        applyColorMap(resizedMatt, heatmap, COLORMAP_JET);

        // 读取背景图像
        Mat background = imread("background.bmp");
        if (background.empty()) {
            cout << "Failed to open background image." << endl;
            return -1;
        }

        // 缩放背景图像大小
        resize(background, background, Size(1920, 1080));

        // 叠加热力图和背景图像
        double alpha = 0.7;
        Mat blended;
        addWeighted(background, alpha, heatmap, 1 - alpha, 0, blended);

        // 逆时针旋转90度
        Mat rotated;
        rotate(blended, rotated, ROTATE_90_CLOCKWISE);

        // 保存合成后的图像为image.bmp
        bool success = imwrite("image.bmp", rotated);
        if (success) {
            cout << "Successfully saved image.bmp" << endl;
            return 0;
        }
        else {
            cout << "Failed to save image.bmp" << endl;
            return -1;
        }
    }
}
