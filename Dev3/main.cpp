#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;

int MIN_DISPARITY = 12;
int MAX_DISPARITY = 55;
int WINDOW_SIZE = 5;
int SYMMETRIC_TOLERANCE = 45;

cv::Mat getRightViewDisparityMap(const cv::Mat &leftImage, const cv::Mat &rightImage)
{
    cv::Mat disparityMap(rightImage.rows, rightImage.cols, CV_8U, cv::Scalar(0));

    for (int y = 0; y < rightImage.rows; ++y)
    {
        int previousBestDisparity = 0;
        int previousX = -1;

        for (int x = 0; x < rightImage.cols; ++x)
        {
            int bestDisparityRight = 0;
            float bestSumRight = -FLT_MAX;

            // Step 1: Compute disparity from right to left
            for (int d = MIN_DISPARITY; d <= MAX_DISPARITY; ++d)
            {
                if (x + d >= leftImage.cols)
                    continue;

                float sum = 0.0f;

                for (int dy = -WINDOW_SIZE; dy <= WINDOW_SIZE; ++dy)
                {
                    for (int dx = -WINDOW_SIZE; dx <= WINDOW_SIZE; ++dx)
                    {
                        int yy = y + dy;
                        int xx = x + dx;
                        int xxShifted = x + dx + d;

                        if (yy < 0 || yy >= rightImage.rows || xx < 0 || xx >= rightImage.cols)
                            continue;
                        if (xxShifted < 0 || xxShifted >= leftImage.cols)
                            continue;

                        int pixelRight = rightImage.at<uchar>(yy, xx);
                        int pixelLeft = leftImage.at<uchar>(yy, xxShifted);

                        float diff = pixelRight - pixelLeft;
                        sum += -1.0f * diff * diff;
                    }
                }

                if (sum > bestSumRight)
                {
                    bestDisparityRight = d;
                    bestSumRight = sum;
                }
            }

            // Step 2: Compute matching disparity from left to right (symmetry check)
            int matchXInLeft = x + bestDisparityRight;
            int bestDisparityLeft = 0;
            float bestSumLeft = -FLT_MAX;

            if (matchXInLeft < leftImage.cols)
            {
                for (int d = MIN_DISPARITY; d <= MAX_DISPARITY; ++d)
                {
                    if (matchXInLeft - d < 0)
                        continue;

                    float sum = 0.0f;

                    for (int dy = -WINDOW_SIZE; dy <= WINDOW_SIZE; ++dy)
                    {
                        for (int dx = -WINDOW_SIZE; dx <= WINDOW_SIZE; ++dx)
                        {
                            int yy = y + dy;
                            int xxL = matchXInLeft + dx;
                            int xxR = matchXInLeft + dx - d;

                            if (yy < 0 || yy >= leftImage.rows || xxL < 0 || xxL >= leftImage.cols || xxR < 0 || xxR >= rightImage.cols)
                                continue;

                            int pixelLeft = leftImage.at<uchar>(yy, xxL);
                            int pixelRight = rightImage.at<uchar>(yy, xxR);

                            float diff = pixelLeft - pixelRight;
                            sum += -1.0f * diff * diff;
                        }
                    }

                    if (sum > bestSumLeft)
                    {
                        bestDisparityLeft = d;
                        bestSumLeft = sum;
                    }
                }
            }

            // Step 3: Order constraint
            bool orderOK = true;
            if (previousX != -1 && x > previousX)
            {
                int mappedPrev = previousX + previousBestDisparity;
                int mappedCurr = x + bestDisparityRight;

                if (mappedCurr <= mappedPrev) {
                    // orderOK = false;
                }
            }

            // Step 4: Symmetry check and final assignment
            if (orderOK && std::abs(bestDisparityRight - bestDisparityLeft) <= SYMMETRIC_TOLERANCE)
            {
                disparityMap.at<uchar>(y, x) = static_cast<uchar>(bestDisparityRight);
                previousBestDisparity = bestDisparityRight;
                previousX = x;
            }
        }
    }

    return disparityMap;
}

cv::Mat getDisparityMap(const cv::Mat &image1, const cv::Mat &image2)
{
    cv::Mat disparityMap(image1.rows, image1.cols, CV_8U);

    for (int y = 0; y < image1.rows; ++y)
    {
        int previousBestDisparity = 0;
        int previousX = -1;
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

            // CONTRAINTE D'ORDRE
            bool orderOk = true;

            if (previousX != -1 && x > previousX)
            {
                int mappedPrev = previousX - previousBestDisparity;
                int mappedCurr = x - bestDisparityLeft;

                if (mappedCurr <= mappedPrev)
                {
                    // orderOk = false;
                }
            }

            // CONTRAINTE DE SYMMETRIE
            if (orderOk && abs(bestDisparityLeft - bestDisparityRight) <= SYMMETRIC_TOLERANCE)
            {
                disparityMap.at<uchar>(y, x) = static_cast<uchar>(bestDisparityLeft);
                previousBestDisparity = bestDisparityLeft;
                previousX = x;
            }
        }
    }
    return disparityMap;
}

int main()
{

    cv::Mat imageG = cv::imread("images/Teddy/im0.png", cv::IMREAD_GRAYSCALE);
    cv::Mat imageD = cv::imread("images/Teddy/im1.png", cv::IMREAD_GRAYSCALE);

    if (imageG.empty() || imageD.empty())
    {
        cerr << "Error loading images!" << endl;
        return -1;
    }

    // cv::Mat disparity = getRightViewDisparityMap(imageG, imageD);

    cv::Mat disparity = getDisparityMap(imageG, imageD);

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