#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;
using namespace cv;

int main()
{
	Mat image = imread("images/left_cam.png");

    if (image.empty()) {
        cout << "Could not open or find the image!" << endl;
        return -1;
    }

    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY);  //gray to detect corners

    //GOAT parametres = 750 max pooints, 0.03 quality, 70 min distance pour les photos *_cam.png

    vector<Point2f> corners;
	int maxCorners = 750;
	double qualityLevel = 0.03;
	double minDistance = 70;  
    goodFeaturesToTrack(gray, corners, maxCorners, qualityLevel, minDistance);

    for (const Point2f& pt : corners) {
        circle(image, pt, 5, Scalar(0, 255, 0), -1);
    }

    imwrite("corners.png", image);
    waitKey(0);
	cout << corners.size() << endl;
    return 0;
}