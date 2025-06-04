#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <fstream>

using namespace std;
using namespace cv;

vector<Point2f> filterCorners(vector<Point2f> corners)
{
    vector<Point2f> filtered;
    double distanceThreshold = 180.0;
    int minNeighbors = 3;
    int minSpreadX = 50;
    int minSpreadY = 50;

    for (size_t i = 0; i < corners.size(); ++i)
    {
        vector<Point2f> neighbors;

        for (size_t j = 0; j < corners.size(); ++j)
        {
            if (i == j) //compare the corner with itself
                continue;
            if (norm(corners[i] - corners[j]) < distanceThreshold)  //if the corner is near (~180 px)
                neighbors.push_back(corners[j]);
        }

        if (neighbors.size() >= minNeighbors)   //min 3 because its a square
        {
            float minX = neighbors[0].x, maxX = neighbors[0].x;
            float minY = neighbors[0].y, maxY = neighbors[0].y;

            for (const auto &p : neighbors)
            {
                minX = min(minX, p.x);
                maxX = max(maxX, p.x);
                minY = min(minY, p.y);
                maxY = max(maxY, p.y);  
            }

            float spreadX = maxX - minX;    //bounding boxes maximums
            float spreadY = maxY - minY;

            if (spreadX >= minSpreadX && spreadY >= minSpreadY) //only keep square neighbors
                filtered.push_back(corners[i]);
        }
    }
    return filtered;
}

int main()
{
    const double PIXEL_SIZE = 0.00122; //https://www.samsung.com/uk/support/mobile-devices/check-out-the-new-camera-functions-of-the-galaxy-s22-series/

    Mat image = imread("images/front.png");

    if (image.empty())
    {
        cout << "Could not open or find the image!" << endl;
        return -1;
    }

    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY); // gray to detect corners

    // GOAT parametres = 750 max pooints, 0.03 quality, 70 min distance pour les photos *_cam.png

    vector<Point2f> corners;
    int maxCorners = 750;
    double qualityLevel = 0.03;
    double minDistance = 70;
    goodFeaturesToTrack(gray, corners, maxCorners, qualityLevel, minDistance);  //Corners

    sort(corners.begin(), corners.end(),    //Sort column by column
         [](const Point2f &a, const Point2f &b)
         {
             if (fabs(a.x - b.x) > 1e-3)
                 return a.x < b.x;
             return a.y < b.y;
         });

    vector<Point2f> filtered = filterCorners(corners); //exclude wrong corners

    for (const Point2f &pt : filtered)  //draw points
    {
        circle(image, pt, 5, Scalar(0, 255, 0), -1);
    }
    imwrite("corners.png", image);
    waitKey(0);

    ofstream file("corners.txt");   //write points
    if (!file.is_open())
    {
        cout << "Could not open the file!" << endl;
        return -1;
    }

    for (const Point2f &pt : filtered)
    {
        file << pt.x << "," << pt.y << "," << "0" << endl;
    }
    file.close();

    return 0;
}