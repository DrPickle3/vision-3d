#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <fstream>
#include <string>
#include <iomanip>

using namespace std;
using namespace cv;

int main()
{
    vector<cv::Mat> images;

    string prefix = "images/bit";
    string extension = ".ppm";

    for (int i = 1; i < 10; i++)
    {
        for (const string suffix : {"", "_inv"})
        {
            ostringstream filename;
            filename << prefix << setw(2) << setfill('0') << i << suffix << extension;

            cv::Mat img = cv::imread(filename.str(), IMREAD_GRAYSCALE);
            if (img.empty())
            {
                cerr << "Could not open or find: " << filename.str() << endl;
                continue;
            }
            images.push_back(img);
            // cout << "Loaded: " << filename.str() << endl;
        }
    }

    int rows = images[0].rows;
    int cols = images[0].cols;

    std::vector<std::string> binary(rows * cols, "");
    uchar seuil = 10;

    for (int i = 8; i >= 0; i--)
    {
        cv::Mat normal = images[i * 2];
        cv::Mat inverse = images[i * 2 + 1];

        cv::Mat verif = Mat::zeros(images[0].size(), CV_8UC1);

        for (int y = 0; y < rows; y++)
        {
            for (int x = 0; x < cols; x++)
            {
                uchar normalValue = normal.at<uchar>(y, x); // Grayscale avec opencv c en uchar et non int
                uchar inverseValue = inverse.at<uchar>(y, x);

                if (normalValue - inverseValue > seuil)
                {
                    verif.at<uchar>(y, x) = 255;
                    binary[y * cols + x] += "1";
                }
                else
                {
                    verif.at<uchar>(y, x) = 0;
                    binary[y * cols + x] += "0";
                }
            }
        }
        string filename = "verif" + to_string(i + 1) + ".png";
        imwrite(filename, verif);
    }

    Mat emptyImage = Mat::zeros(images[0].size(), CV_8UC1);

    // long sum = 0;
    double maxValue = 511;

    for (int y = 0; y < emptyImage.rows; y++)
    {
        for (int x = 0; x < emptyImage.cols; x++)
        {
            string binaryCol = binary[y * emptyImage.cols + x];
            int value = 0;

            for (int index = binaryCol.size() - 1; index >= 0; index--)
            {
                char bit = binaryCol[index];
                if (bit == '1')
                    value += pow(2, binaryCol.size() - 1 - index);
            }

            emptyImage.at<uchar>(y, x) = abs(x - value) / maxValue * 255;
            // sum += abs(x - value);
        }
    }

    // double sq_sum = 0;
    // int total = emptyImage.rows * emptyImage.cols;
    // double mean = sum / total;

    // for (int y = 0; y < emptyImage.rows; y++)
    // {
    //     for (int x = 0; x < emptyImage.cols; x++)
    //     {
    //         uchar val = emptyImage.at<uchar>(y, x);
    //         double diff = val - mean;
    //         sq_sum += diff * diff;
    //     }
    // }

    // double variance = sq_sum / total;
    // double ecart_type = sqrt(variance);

    // for (int y = 0; y < emptyImage.rows; y++)
    // {
    //     for (int x = 0; x < emptyImage.cols; x++)
    //     {
    //         // emptyImage.at<uchar>(y, x) = min(mean + 2 * ecart_type, static_cast<double>(emptyImage.at<uchar>(y, x)));
    //         // emptyImage.at<uchar>(y, x) = max(mean - 2 * ecart_type, static_cast<double>(emptyImage.at<uchar>(y, x)));
    //     }
    // }

    // for (int y = 0; y < emptyImage.rows; y++)
    // {
    //     for (int x = 0; x < emptyImage.cols; x++)
    //     {
    //         // emptyImage.at<uchar>(y, x) = emptyImage.at<uchar>(y, x) / maxValue * 255;
    //     }
    // }

    // cout << maxValue << endl;

    // for (int i = 0; i < binary.size(); i++)
    // {
    //     if (binary[i] != "000000000")
    //         cout << binary[i] << endl;
    // }

    imwrite("disparity.png", emptyImage);
    return 0;
}
