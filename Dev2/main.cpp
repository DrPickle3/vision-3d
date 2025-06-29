#include <iostream>
#include <opencv2/opencv.hpp>

int main() {

    // Caméra de gauche
    cv::Mat R_cam_g = (cv::Mat_<double>(3, 3) <<
        0.9962, 0,     -0.0872,
        0,      0.9962, 0,
        0.0872, 0,      0.9962
    );

    cv::Mat T_cam_g = (cv::Mat_<double>(3, 1) << 0, 0, 0);
    cv::Point2d O_cam_g(538.625, 510.471);
    cv::Point2d S_cam_g(0.00155227, 0.00155227);
    double zprime_cam_g = 1.0;

    // Caméra de droite
    cv::Mat R_cam_d = (cv::Mat_<double>(3, 3) <<
        0.9962, 0,     0.0872,
        0,      0.9962, 0,
        -0.0872, 0,    0.9962
    );

    cv::Mat T_cam_d = (cv::Mat_<double>(3, 1) << 5, 0, 0);
    cv::Point2d O_cam_d(765.134, 510.599);
    cv::Point2d S_cam_d(0.00155227, 0.00155227);
    double zprime_cam_d = 1.0;

    std::cout << "Hello World!" << std::endl;
    return 0;
}