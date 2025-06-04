#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <fstream>

using namespace std;
using namespace cv;

int main()
{
    const double PIXEL_SIZE = 0.00122;
	Mat image = imread("images/left.png");

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
      // 1) Trier les points par x, puis par y
    std::sort(corners.begin(), corners.end(),
        [](const Point2f &a, const Point2f &b) {
            if (std::fabs(a.x - b.x) > 1e-3)
                return a.x < b.x;
            return a.y < b.y;
        });

    // 2) Écrire dans corners.txt une fois trié
    ofstream file("corners.txt");
    if (!file.is_open()) {
        cout << "Could not open the file!" << endl;
        return -1;
    }
    for (const Point2f& pt : corners) {
        file << pt.x << "," << pt.y << "," << "0" << endl;
    }
    file.close();


    return 0;
}