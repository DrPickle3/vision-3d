#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

// Corrélation locale simple avec SSD (Sum of Squared Differences)
// tailleFenetre : similitude
// maxDisparite  : limite de la disparité
Mat correlationSSD(const Mat& gauche, const Mat& droite, int tailleFenetre, int maxDisparite)
{
    int numberOfRows = gauche.rows;
    int numberOfColumns = gauche.cols;
    int halfWindow = tailleFenetre;

    // Initialisation image de sortie
    Mat img_disparite(numberOfRows, numberOfColumns, CV_8U, Scalar(0));

    for (int y = halfWindow; y < numberOfRows - halfWindow; ++y)
    {
        for (int x = halfWindow; x < numberOfColumns - halfWindow; ++x)
        {
            int meilleureDisparite = 0;
            double meilleurSSD = DBL_MAX;

            // Contrainte ordre
            // On vérifie la disparité précédente pour éviter les sauts importants
            int disparitePrecedente = (x > halfWindow) ? img_disparite.at<uchar>(y, x - 1) : -1;

            for (int disp = 0; disp <= maxDisparite; ++disp)
            {
                // Contrainte d'ordre
                // Appliquer la contrainte d'ordre uniquement si on a une disparité précédente
                if (disparitePrecedente >= 0 && disp > disparitePrecedente + 4) // tolérance de 4
                    continue;

                int xDroiteDecalee = x - disp;
                // Vérification des limites de l'image droite
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
            img_disparite.at<uchar>(y, x) = static_cast<uchar>(meilleureDisparite);
        }
    }

    return img_disparite;
}


// tailleFenetre : similitude
// maxDisparite  : limite de la disparité
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
    int maxDisparite = 64;

    Mat resultat = correlationSSD(imageG, imageD, tailleFenetre, maxDisparite);

    // Seulement contrainte de similitude + limite de la disparité
    Mat carteDisparite;
    resultat.convertTo(carteDisparite, CV_8U, 255.0 / maxDisparite);
    imwrite("./images/carte_disparite_similitude+limite.png", carteDisparite);
    waitKey(10);



    cout << "Hello World!" << endl;
    return 0;
}