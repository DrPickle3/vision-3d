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
            if (i == j) // compare the corner with itself
                continue;
            if (norm(corners[i] - corners[j]) < distanceThreshold) // if the corner is near (~180 px)
                neighbors.push_back(corners[j]);
        }

        if (neighbors.size() >= minNeighbors) // min 3 because its a square
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

            float spreadX = maxX - minX; // bounding boxes maximums
            float spreadY = maxY - minY;

            if (spreadX >= minSpreadX && spreadY >= minSpreadY) // only keep square neighbors
                filtered.push_back(corners[i]);
        }
    }
    return filtered;
}

int main()
{
    const double PIXEL_SIZE = 0.00122; // https://www.samsung.com/uk/support/mobile-devices/check-out-the-new-camera-functions-of-the-galaxy-s22-series/

    // 1) Charger l’image et détecter les coins
    Mat image = imread("images/front.png");

    if (image.empty())
    {
        cout << "Could not open or find the image!" << endl;
        return -1;
    }
    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY); // gray to detect corners

    // GOAT parametres = 750 max pooints, 0.03 quality, 70 min distance pour les photos *_cam.png

    // Pour faire notre fichier de points correspondant
    // en assumant que points.txt et corners.txt soit trie par colonne
    writeCorrespondingPoints();

    vector<Point2f> corners;
    int maxCorners = 750;
    double qualityLevel = 0.03;
    double minDistance = 70;
    goodFeaturesToTrack(gray, corners, maxCorners, qualityLevel, minDistance); // Corners

    sort(corners.begin(), corners.end(), // Sort column by column
         [](const Point2f &a, const Point2f &b)
         {
             if (fabs(a.x - b.x) > 1e-3)
                 return a.x < b.x;
             return a.y < b.y;
         });

    vector<Point2f> filtered = filterCorners(corners); // exclude wrong corners

    for (const Point2f &pt : filtered) // draw points
    {
        circle(image, pt, 5, Scalar(0, 255, 0), -1);
    }
    imwrite("corners.png", image);
    waitKey(0);

    ofstream file("corners.txt"); // write points
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

    // -------------------------------
    // Étape 0 : passage pixel → repère image « réel »
    // -------------------------------
    Point2f O;
    O.x = image.cols / 2.0f; // Om en pixels
    O.y = image.rows / 2.0f; // On en pixels

    float S = PIXEL_SIZE; // largeur physique d’un pixel (mm)

    vector<Point2f> realCorners;
    realCorners.reserve(corners.size());
    for (const Point2f &pt : corners)
    {
        float dx = pt.x - O.x; // (m_i − O_m)
        float dy = pt.y - O.y; // (n_i − O_n)
        float xdi = dx * S;   // x_{d,i} = (m_i − O_m)·S
        float ydi = dy * S;   // y_{d,i} = (n_i − O_n)·S
        realCorners.emplace_back(xdi, ydi);
    }

    // -------------------------------
    // Préparation des coordonnées 3D (Xs, Ys, Zs=0) de la mire
    // -------------------------------
    //
    int nbCols = 22;    // nombre de colonnes de la mire
    int nbRows = 32;    // nombre de lignes de la mire
    double step = 8.0; // espacement (mm) entre deux coins

    int M = static_cast<int>(realCorners.size());
    if (M != nbCols * nbRows)
    {
        cout << "Erreur : M (" << M << ") != nbCols*nbRows ("
             << nbCols * nbRows << ") !" << endl;
        return -1;
    }

    vector<double> Xs(M), Ys(M);
    for (int i = 0; i < M; ++i)
    {
        int ligne = i / nbCols;   // 0 ≤ ligne < nbRows
        int colonne = i % nbCols; // 0 ≤ colonne < nbCols
        Xs[i] = colonne * step;  // Xs[i] en mm
        Ys[i] = ligne * step;    // Ys[i] en mm
        // Zs[i] est implicitement 0
    }

    // -------------------------------
    // Étape 1 de Tsai : construire A et b pour a1…a5
    // -------------------------------
    Mat A = Mat::zeros(M, 5, CV_64F);
    Mat b = Mat::zeros(M, 1, CV_64F);

    for (int i = 0; i < M; ++i)
    {
        double X = Xs[i];            // X_{s,i} (mm)
        double Y = Ys[i];            // Y_{s,i} (mm)
        double x = realCorners[i].x; // x_{d,i} (mm)
        double y = realCorners[i].y; // y_{d,i} (mm)

        // Remplir la ligne i de A :  y·(a1·X + a2·Y + a3)  –  x·(a4·X + a5·Y + 1)  = 0
        A.at<double>(i, 0) = y * X;  // coeff. devant a1 = R11^c/Ty^c
        A.at<double>(i, 1) = y * Y;  // coeff. devant a2 = R12^c/Ty^c
        A.at<double>(i, 2) = y;      // coeff. devant a3 = T_x^c/Ty^c
        A.at<double>(i, 3) = -x * X; // coeff. devant a4 = R21^c/Ty^c
        A.at<double>(i, 4) = -x * Y; // coeff. devant a5 = R22^c/Ty^c

        // Terme de droite = x_{d,i}
        b.at<double>(i, 0) = x;
    }

    // Résoudre en moindres carrés : A·a_hat = b
    Mat a_hat;
    solve(A, b, a_hat, DECOMP_SVD);

    // Extraire a1…a5
    double a1 = a_hat.at<double>(0, 0); // R11^c/Ty^c
    double a2 = a_hat.at<double>(1, 0); // R12^c/Ty^c
    double a3 = a_hat.at<double>(2, 0); // T_x^c/Ty^c
    double a4 = a_hat.at<double>(3, 0); // R21^c/Ty^c
    double a5 = a_hat.at<double>(4, 0); // R22^c/Ty^c

    // -------------------------------
    // Reconstruction partielle de R^c et T^c
    // -------------------------------
    // 1) Normaliser première ligne avec Ty1
    double Ty1 = 1.0 / sqrt(a1 * a1 + a2 * a2);
    double r11c = a1 * Ty1;
    double r12c = a2 * Ty1;
    double tmp1 = 1.0 - (r11c * r11c + r12c * r12c);
    if (tmp1 < 0 && tmp1 > -1e-8)
        tmp1 = 0;
    if (tmp1 < -1e-8)
    {
        cout << "Erreur critique : r11^2 + r12^2 = "
             << (r11c * r11c + r12c * r12c) << " > 1" << endl;
        return -1;
    }
    double signS1 = +1.0;
    double r13c = signS1 * sqrt(tmp1);

    // 2) Normaliser deuxième ligne avec Ty2 (indépendamment)
    double Ty2 = 1.0 / sqrt(a4 * a4 + a5 * a5);
    double r21c = a4 * Ty2;
    double r22c = a5 * Ty2;
    double tmp2 = 1.0 - (r21c * r21c + r22c * r22c);
    if (tmp2 < 0 && tmp2 > -1e-8)
        tmp2 = 0;
    if (tmp2 < -1e-8)
    {
        cout << "Erreur critique : r21^2 + r22^2 = "
             << (r21c * r21c + r22c * r22c) << " > 1" << endl;
        return -1;
    }
    double signS2 = +1.0;
    double r23c = signS2 * sqrt(tmp2);

    // 3) Troisième ligne par produit vectoriel
    Vec3d row1(r11c, r12c, r13c);
    Vec3d row2(r21c, r22c, r23c);
    Vec3d row3_cross = row1.cross(row2);
    double r31c = row3_cross[0];
    double r32c = row3_cross[1];
    double r33c = row3_cross[2];

    // 4) Choisir T_y^c : ici Ty1
    double Tyc = Ty1;

    // 5) Calculer T_x^c
    double txc = a3 * Tyc;

    // 6) Affichage final de R^c et T^c partiels
    cout << "Estimation après Étape 1 :\n";
    cout << " R^c = \n"
         << "   [" << r11c << ", " << r12c << ", " << r13c << "]\n"
         << "   [" << r21c << ", " << r22c << ", " << r23c << "]\n"
         << "   [" << r31c << ", " << r32c << ", " << r33c << "]\n";
    cout << " T^c = [" << txc << ", " << Tyc << ", (inconnu) ]\n";
    cout << "→ Passez à l’Étape 2 pour z' et Tz^c, puis l’Étape 3 pour k₁.\n";

    return 0;
}
