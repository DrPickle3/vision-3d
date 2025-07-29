#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/viz.hpp>
#include <vector>
#include <string>
#include <limits>
#include <fstream>

using namespace std;

// CONTRAINTE DE LIMITE DE DISPARITE
int MIN_DISPARITY = 5;
int MAX_DISPARITY = 75;

// MC par correlation
int WINDOW_SIZE = 5;
int SYMMETRIC_TOLERANCE = 1;
int CONTINUOUS_DISPARITY_TOLERANCE = 20;

cv::Mat getDisparityMap(const cv::Mat &left, const cv::Mat &right)
{
    cv::Mat disparityMap = cv::Mat::zeros(right.rows, right.cols, CV_8U);

    for (int y = WINDOW_SIZE; y < right.rows - WINDOW_SIZE; ++y)
    {
        int previousDisparity = -1;

        for (int x = WINDOW_SIZE; x < right.cols - WINDOW_SIZE; ++x)
        {
            int rightBestDisparity = 0;
            float minRightErr = FLT_MAX;

            // CONTRAINTE DE LIMITE DE DISPARITE
            for (int d = MIN_DISPARITY; d <= MAX_DISPARITY; ++d)
            {
                if (x + d >= left.cols - WINDOW_SIZE)
                    continue;

                float err = 0;

                // CONTRAINTE DE SIMILARITE
                for (int j = -WINDOW_SIZE; j <= WINDOW_SIZE; ++j)
                {
                    for (int k = -WINDOW_SIZE; k <= WINDOW_SIZE; ++k)
                    {
                        int rightPixel = right.at<uchar>(y + j, x + k);
                        int leftPixel = left.at<uchar>(y + j, x + k + d);

                        float diff = rightPixel - leftPixel;
                        err += diff * diff; // Somme des différences au carré
                    }
                }

                // CONTRAINTE DE CONTINUITÉ
                if (previousDisparity != -1)
                {
                    int discontinuity = abs(d - previousDisparity);
                    if (discontinuity > CONTINUOUS_DISPARITY_TOLERANCE)
                    {
                        err += discontinuity * 100;
                    }
                }

                if (err < minRightErr) // On minimise l'erreur
                {
                    minRightErr = err;
                    rightBestDisparity = d;
                }
            }

            // Pour la symétrie, on part du pixel trouvé avec la disparité droite et
            // on cherche la disparité gauche correspondante
            int leftX = x + rightBestDisparity; // Position correspondante dans l'image gauche (du point correspondant trouvé)

            if (leftX >= left.cols - WINDOW_SIZE)
                continue;

            int leftBestDisparity = 0;
            float leftMinErr = FLT_MAX;

            for (int d = MIN_DISPARITY; d <= MAX_DISPARITY; ++d)
            {
                if (leftX - d < WINDOW_SIZE)
                    continue;

                float err = 0;

                for (int j = -WINDOW_SIZE; j <= WINDOW_SIZE; ++j)
                {
                    for (int k = -WINDOW_SIZE; k <= WINDOW_SIZE; ++k)
                    {
                        int leftPixel = left.at<uchar>(y + j, leftX + k);
                        int rightPixel = right.at<uchar>(y + j, leftX + k - d);

                        float diff = leftPixel - rightPixel;
                        err += diff * diff;
                    }
                }

                if (err < leftMinErr)
                {
                    leftMinErr = err;
                    leftBestDisparity = d;
                }
            }

            if (abs(rightBestDisparity - leftBestDisparity) <= SYMMETRIC_TOLERANCE)
            {
                previousDisparity = rightBestDisparity;
                disparityMap.at<uchar>(y, x) = static_cast<uchar>(previousDisparity);
            }
        }
    }

    return disparityMap;
}

void getParameters(ifstream &file, float &zPrime, float &dOx, float &Tx)
{
    string line;
    while (getline(file, line))
    {
        if (line.find("z'=") != std::string::npos)
        {
            size_t pos = line.find('=');
            if (pos != std::string::npos)
            {
                zPrime = std::stof(line.substr(pos + 1)); // convert substring to float
            }
        }
        else if (line.find("dOx=") != std::string::npos)
        {
            size_t pos = line.find('=');
            if (pos != std::string::npos)
            {
                dOx = std::stof(line.substr(pos + 1));
            }
        }
        else if (line.find("Tx=") != std::string::npos)
        {
            size_t pos = line.find('=');
            if (pos != std::string::npos)
            {
                Tx = std::stof(line.substr(pos + 1));
            }
        }
    }
}

cv::Mat getDepthMap(const cv::Mat &disparity, float zPrime, float dOx, float Tx)
{
    cv::Mat depthMap = cv::Mat::zeros(disparity.size(), CV_32F); // Float 32-bit

    std::vector<cv::Vec3f> points;
    for (int y = 0; y < disparity.rows; y++)
    {
        for (int x = 0; x < disparity.cols; x++)
        {
            float d = static_cast<float>(disparity.at<uchar>(y, x));
            if (d != 0)
            {
                float z = (zPrime * Tx) / (d + dOx);

                depthMap.at<float>(y, x) = z;
                // renverse y pcq cv::viz a un repere avec un y axis vers le haut
                // renverse z pour avoir les objets avec un petit z devant
                // et les objets avec un grand z derriere
                points.emplace_back(x, disparity.cols - y, -z);
            }
        }
    }

    cv::Mat cloudMat(points.size(), 1, CV_32FC3, points.data());

    // Visualize
    cv::viz::Viz3d window("Depth Visualization");
    cv::viz::WCloud cloud(cloudMat, cv::viz::Color::green());
    window.showWidget("cloud", cloud);
    // window.spin();
    /*
    *
    * 
    * 
    * DECOMMENTER LA LIGNE AU DESSUS POUR VOIR LA VISUALISATION
    * 
    *     
    * 
    */
    return depthMap;
}

void writeDisparityAndDepthMap(string folderName)
{
    cv::Mat imageLeft = cv::imread("images/" + folderName + "/im0.png", cv::IMREAD_GRAYSCALE);
    cv::Mat imageRight = cv::imread("images/" + folderName + "/im1.png", cv::IMREAD_GRAYSCALE);
    std::ifstream parameters("images/" + folderName + "/calib.txt");

    if (imageLeft.empty() || imageRight.empty() || !parameters.is_open())
    {
        cerr << "Error loading images!" << endl;
        return;
    }

    float zPrime, dOx, Tx;
    getParameters(parameters, zPrime, dOx, Tx);

    cv::Mat disparity = getDisparityMap(imageLeft, imageRight);
    cv::Mat depth = getDepthMap(disparity, zPrime, dOx, Tx);

    cv::Mat normalizedDisparity;
    normalize(disparity, normalizedDisparity, 0, 255, cv::NORM_MINMAX);

    // la depth map reviens a linverse pomal de la disparity map quand est en png 2d.
    // Pour la voir dans un petit visualisateur juste a aller decommenter la ligne dans le code plus haut
    cv::Mat normalizedDepth;
    normalize(depth, normalizedDepth, 0, 255, cv::NORM_MINMAX);

    cv::imwrite("images/" + folderName + "/disparityMap.png", normalizedDisparity);
    cv::imwrite("images/" + folderName + "/depthMap.png", normalizedDepth);
}

int main()
{
    writeDisparityAndDepthMap("Adirondack");
    writeDisparityAndDepthMap("Playroom");
    writeDisparityAndDepthMap("Teddy");
    writeDisparityAndDepthMap("Vintage");
    return 0;
}