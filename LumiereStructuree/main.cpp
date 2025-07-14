#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <fstream>

using namespace std;
using namespace cv;



int main()
{
    // 1) Charger l’image et détecter les coins
    Mat image = imread("images/bit01.ppm");

    if (image.empty())
    {
        cout << "Could not open or find the image!" << endl;
        return -1;
    }

    // chaque pixel trouver sequence binaire => colonne de reference i

    //  x - i

    imwrite("corners.png", image);
    waitKey(0);

    return 0;
}
