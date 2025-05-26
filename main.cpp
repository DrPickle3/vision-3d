#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;
using namespace cv;

int main()
{
	Mat image = imread("images/right.png");

    if (image.empty()) {
        cout << "Could not open or find the image!" << endl;
        return -1;
    }

    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY); // Convert to grayscale

    vector<Point2f> corners;
	int maxCorners = 750;
	double qualityLevel = 0.01;
	double minDistance = 50;  
    goodFeaturesToTrack(gray, corners, maxCorners, qualityLevel, minDistance);

    // Draw the detected corners as green circles
    for (const Point2f& pt : corners) {
        circle(image, pt, 5, Scalar(0, 255, 0), -1); // Green circle
    }

    imwrite("corners.png", image);
    waitKey(0);
	cout << corners.size() << endl;
    return 0;
}