#include <opencv2/opencv.hpp>
#include <iostream>
#include <opencv2/viz.hpp>

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
            double meilleurScore = DBL_MAX;

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

                //Contrainte de continuite selon ce que je comprends
                double penaliteContinuite = 0.0;
                // Seuil de continuité
                int epsilon = 3;

                // Comparer avec le pixel de gauche
                if (x > halfWindow)
                {
                    int dispVoisin = img_disparite.at<uchar>(y, x - 1);
                    penaliteContinuite += abs(disp - dispVoisin) > epsilon ? 50.0 : 0.0;
                }
                // Comparer avec le pixel du dessus
                if (y > halfWindow)
                {
                    int dispVoisin = img_disparite.at<uchar>(y - 1, x);
                    penaliteContinuite += abs(disp - dispVoisin) > epsilon ? 50.0 : 0.0;
                }

                double score = ssd + penaliteContinuite;

                if (score < meilleurScore)
                {
                    meilleurScore = score;
                    meilleureDisparite = disp;
                }
            }
            img_disparite.at<uchar>(y, x) = static_cast<uchar>(meilleureDisparite);
        }
    }

    return img_disparite;
}

Mat calculerProfondeur(const Mat& disparite)
{
    double zPrime = 1500.0;
    double Tx = 80.0;
    double dOx = 52.0;

    Mat carteProfondeur(disparite.size(), CV_32F, Scalar(0));

    for (int y = 0; y < disparite.rows; ++y)
    {
        for (int x = 0; x < disparite.cols; ++x)
        {
            int disp = disparite.at<uchar>(y, x);

            if (disp > 0)
            {
                float profondeur = static_cast<float>((zPrime * Tx) / (disp + dOx));
                carteProfondeur.at<float>(y, x) = profondeur;
            }
            else
            {
                carteProfondeur.at<float>(y, x) = 0.0f;
            }
        }
    }

    return carteProfondeur;
}

void afficherCarteProfondeur3D(const Mat& carteProfondeur)
{
    double focale = 1500.0;
    double Ox = carteProfondeur.cols / 2.0;
    double Oy = carteProfondeur.rows / 2.0;

    vector<Vec3f> points;

    // Conversioncarte de profondeur en nuage de points 3D
    for (int y = 0; y < carteProfondeur.rows; ++y)
    {
        for (int x = 0; x < carteProfondeur.cols; ++x)
        {
            float z = carteProfondeur.at<float>(y, x);

            if (z > 0) 
            {
                float X = (x - Ox) * z / focale;
                float Y = (Oy - y) * z / focale; // inversion de Y pour l'affichage
                points.emplace_back(X, Y, -z); // -z pour que les objets proches soient devant
            }
        }
    }

    Mat cloudMat(points.size(), 1, CV_32FC3, points.data());

    viz::Viz3d fenetre("Visualisation 3D de la carte de profondeur");
    viz::WCloud nuage(cloudMat, viz::Color::green());
    fenetre.showWidget("nuage", nuage);
    fenetre.spin();
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
    int maxDisparite = 64;

    Mat disparite = correlationSSD(imageG, imageD, tailleFenetre, maxDisparite);

    Mat carteProfondeur = calculerProfondeur(disparite);
    afficherCarteProfondeur3D(carteProfondeur);

    // Seulement contrainte de similitude + limite de la disparité
    Mat carteDisparite;
    disparite.convertTo(carteDisparite, CV_8U, 255.0 / maxDisparite);
    imwrite("./images/carte_disparite_similitude+limite.png", carteDisparite);
    waitKey(10);

    cout << "Hello World!" << endl;
    return 0;
}