#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;

int MIN_DISPARITY = 5;
int MAX_DISPARITY = 60;
int WINDOW_SIZE = 5;
int SYMMETRIC_TOLERANCE = 45;

cv::Mat getDisparityMap(const cv::Mat &image1, const cv::Mat &image2)
{
    cv::Mat disparityMap(image1.rows, image1.cols, CV_8U);

    for (int y = 0; y < image1.rows; ++y)
    {
        for (int x = 0; x < image1.cols; ++x)
        {
            int bestDisparityLeft = 0;
            float bestSumLeft = -FLT_MAX;

            for (int d = MIN_DISPARITY; d <= MAX_DISPARITY; ++d) // CONTRAINTE LIMITE DE DISPARITE
            {
                if (x - d < 0)
                    continue;

                float sum = 0;

                for (int k = -WINDOW_SIZE; k <= WINDOW_SIZE; ++k)
                {
                    for (int j = -WINDOW_SIZE; j <= WINDOW_SIZE; ++j)
                    {
                        if (y + j < 0 || y + j >= image1.rows)
                            continue;
                        if (x + k < 0 || x + k >= image1.cols)
                            continue;
                        if (x + k - d < 0 || x + k - d >= image2.cols)
                            continue;

                        int pixelGNeighbor = image1.at<uchar>(y + j, x + k);
                        int pixelDNeighbor = image2.at<uchar>(y + j, x + k - d);

                        sum += -1.0f * (pixelGNeighbor - pixelDNeighbor) * (pixelGNeighbor - pixelDNeighbor);
                    }
                }

                // CONTRAINTE DE SIMILARITE
                if (sum > bestSumLeft)
                {
                    bestDisparityLeft = d;
                    bestSumLeft = sum;
                }
            }

            int bestDisparityRight = 0;
            float bestSumRight = -FLT_MAX;

            for (int d = MIN_DISPARITY; d <= MAX_DISPARITY; ++d) // CONTRAINTE LIMITE DE DISPARITE
            {
                if (x - d < 0)
                    continue;

                float sum = 0;

                for (int k = -WINDOW_SIZE; k <= WINDOW_SIZE; ++k)
                {
                    for (int j = -WINDOW_SIZE; j <= WINDOW_SIZE; ++j)
                    {
                        if (y + j < 0 || y + j >= image2.rows)
                            continue;
                        if (x + k < 0 || x + k >= image2.cols)
                            continue;
                        if (x + k - d < 0 || x + k - d >= image1.cols)
                            continue;

                        int pixelGNeighbor = image1.at<uchar>(y + j, x + k - d);
                        int pixelDNeighbor = image2.at<uchar>(y + j, x + k);

                        sum += -1.0f * (pixelDNeighbor - pixelGNeighbor) * (pixelDNeighbor - pixelGNeighbor);
                    }
                }

                // CONTRAINTE DE SIMILARITE
                if (sum > bestSumRight)
                {
                    bestDisparityRight = d;
                    bestSumRight = sum;
                }
            }
            if (abs(bestDisparityLeft - bestDisparityRight) <= SYMMETRIC_TOLERANCE)
            {
                // CONTRAINTE DE SYMMETRIE
                disparityMap.at<uchar>(y, x) = static_cast<uchar>(bestDisparityLeft);
            }
        }
    }
    return disparityMap;
}

int main()
{

    cv::Mat imageG = cv::imread("images/im0.png", cv::IMREAD_GRAYSCALE);
    cv::Mat imageD = cv::imread("images/im1.png", cv::IMREAD_GRAYSCALE);

    if (imageG.empty() || imageD.empty())
    {
        cerr << "Error loading images!" << endl;
        return -1;
    }

    cv::Mat disparity = getDisparityMap(imageG, imageD);

    // cv::Mat disparityD = getDisparityMap(imageD, imageG);

    // cv::Mat disparitySymmetric(disparityG.rows, disparityG.cols, CV_8U, cv::Scalar(0));

    // // CONTRAINTE DE SYMMETRIE
    // for (int y = 0; y < disparitySymmetric.rows; y++)
    // {
    //     for (int x = 0; x < disparitySymmetric.cols; x++)
    //     {
    //         int d = disparityG.at<uchar>(y, x);
    //         int xr = x - d;

    //         if (xr >= 0 && xr < imageD.cols)
    //         {
    //             int dR = disparityD.at<uchar>(y, xr);
    //             if (abs(d - dR) <= SYMMETRIC_TOLERANCE)
    //             {
    //                 disparitySymmetric.at<uchar>(y, x) = d;
    //             }
    //         }
    //     }
    // }

    normalize(disparity, disparity, 0, 255, cv::NORM_MINMAX);

    cv::imwrite("images/disparityMap.png", disparity);

    return 0;
}