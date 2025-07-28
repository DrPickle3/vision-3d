#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

// Corrélation locale simple avec SSD (Sum of Squared Differences)
// tailleFenetre : similitude
// maxDisparite  : limite de la disparité
Mat correlationSSD(const Mat &gauche, const Mat &droite, int tailleFenetre, int maxDisparite)
{
    int numberOfRows = gauche.rows;
    int numberOfColumns = gauche.cols;
    int halfWindow = tailleFenetre;

    // Initialisation image de sortie
    Mat img_disparite(numberOfRows, numberOfColumns, CV_8U, Scalar(0));

    for (int y = halfWindow; y < numberOfRows - halfWindow; ++y)
    {
        for (int x = halfWindow + maxDisparite; x < numberOfColumns - halfWindow; ++x)
        {
            int meilleureDisparite = 0;
            double meilleurSSD = DBL_MAX;

            for (int disp = 0; disp <= maxDisparite; ++disp)
            {
                int xDroiteDecalee = x - disp;
                if (xDroiteDecalee - halfWindow < 0)
                    continue;

                double ssd = 0.0;

                for (int offsetY = -halfWindow; offsetY <= halfWindow; ++offsetY)
                {
                    for (int offsetX = -halfWindow; offsetX <= halfWindow; ++offsetX)
                    {
                        uchar valGauche = gauche.at<uchar>(y + offsetY, x + offsetX);
                        uchar valDroite = droite.at<uchar>(y + offsetY, xDroiteDecalee + offsetX);
                        double diff = static_cast<double>(valGauche) - static_cast<double>(valDroite);
                        ssd += diff * diff;
                    }
                }

                if (ssd < meilleurSSD)
                {
                    meilleurSSD = ssd;
                    meilleureDisparite = disp;
                }
            }

            // Normalisation de la disparité pour l'affichage
            img_disparite.at<uchar>(y, x) = static_cast<uchar>(meilleureDisparite * 255 / maxDisparite);
        }
    }

    return img_disparite;
}

int main()
{
    Mat imageG = imread("images/im0.png", IMREAD_GRAYSCALE);
    Mat imageD = imread("images/im1.png", IMREAD_GRAYSCALE);

    if (imageG.empty() || imageD.empty())
    {
        cout << "Could not open or find the image!" << endl;
        return -1;
    }

    if (imageG.size() != imageD.size())
    {
        cout << "Images must be of the same size!" << endl;
        return -1;
    }

    int tailleFenetre = 5;

    // Seulement contrainte de similitude
    int maxDisparite = 64;
    Mat carteDisparite = correlationSSD(imageG, imageD, tailleFenetre, maxDisparite);
    imwrite("./images/carte_disparite_similitude.png", carteDisparite);
    waitKey(10);

    // pour contrainte de limite de la disparité
    // int tailleFenetre = 5;

    for (int i = 0; i < imageG.rows; i++)
    {
        for (int j = 0; j < imageG.cols; j++)
        {
        }
    }

    cout << "Hello World!" << endl;
    return 0;
}