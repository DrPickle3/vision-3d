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

vector<Point3f> getPoints()
{
    vector<Point3f> points3D;
    string line;

    ifstream filePoints("points.txt");
    if (!filePoints.is_open())
    {
        cerr << "Impossible d'ouvrir points.txt" << endl;
        return points3D;
    }

    while (getline(filePoints, line))
    {
        replace(line.begin(), line.end(), ',', ' ');
        istringstream iss(line);
        float x, y, z;
        if (iss >> x >> y >> z)
        {
            points3D.emplace_back(x, y, z);
        }
    }
    filePoints.close();
    return points3D;
}

static int writeCorrespondingPoints()
{
    vector<Point3f> points3D = getPoints();

    ifstream fileCorners("corners.txt");
    if (!fileCorners.is_open())
    {
        cerr << "Impossible d'ouvrir corners.txt" << endl;
        return -1;
    }

    vector<Point3f> corners;
    string line;

    while (getline(fileCorners, line))
    {
        replace(line.begin(), line.end(), ',', ' ');
        istringstream iss(line);
        float u, v, w;
        if (iss >> u >> v >> w)
        {
            corners.emplace_back(u, v, w);
        }
    }
    fileCorners.close();

    // Verifier que points.txt et corners.txt ont le meme nbre de points
    if (points3D.size() != corners.size())
    {
        cerr << "Erreur : nombre de points 3D (" << points3D.size()
             << ") et de coins 3D (" << corners.size() << ") différent." << endl;
        return -1;
    }

    // Ecrire dans le fichier correspondingPoints.txt
    ofstream fileCorrespondingPoints("correspondingPoints.txt");
    if (!fileCorrespondingPoints.is_open())
    {
        cerr << "Impossible d'ouvrir correspondingPoints.txt pour écriture." << endl;
        return -1;
    }

    for (size_t i = 0; i < points3D.size(); ++i)
    {
        fileCorrespondingPoints << points3D[i].x << " "
                                << points3D[i].y << " "
                                << points3D[i].z << " , "
                                << corners[i].x << " "
                                << corners[i].y << " "
                                << corners[i].z << endl;
    }

    cout << "Fichier correspondingPoints.txt généré avec succès." << endl;
    cout << "Nombre de mises en correspondances : " << points3D.size() << endl;

    return 0;
}

double getTy(double r11, double r12, double r21, double r22)
{
    double condition = r11 * r22 - r12 * r21;
    double condition2 = r11 * r11 + r21 * r21;
    double condition3 = r12 * r12 + r22 * r22;
    double condition4 = r21 * r21 + r22 * r22;
    double condition5 = r11 * r11 + r12 * r12;

    if (condition != 0)
    {
        double S = r11 * r11 + r12 * r12 + r21 * r21 + r22 * r22;
        return sqrt((S - sqrt(pow(S, 2) - 4 * pow(condition, 2))) / (2 * pow(condition, 2)));
    }
    else if (condition2 != 0)
    {
        return 1 / sqrt(condition2);
    }
    else if (condition3 != 0)
    {
        return 1 / sqrt(condition3);
    }
    else if (condition4 != 0)
    {
        return 1 / sqrt(condition4);
    }
    else if (condition5 != 0)
    {
        return 1 / sqrt(condition5);
    }
    return 0;
}

int main()
{
    const double PIXEL_SIZE = 0.00122; // https://www.samsung.com/uk/support/mobile-devices/check-out-the-new-camera-functions-of-the-galaxy-s22-series/

    // 1) Charger l’image et détecter les coins
    Mat image = imread("images/left.png");

    if (image.empty())
    {
        cout << "Could not open or find the image!" << endl;
        return -1;
    }
    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY); // gray to detect corners

    // GOAT parametres = 750 max pooints, 0.03 quality, 70 min distance pour les photos *.png

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

    // Pour faire notre fichier de points correspondant
    // en assumant que points.txt et corners.txt soit trie par colonne
    writeCorrespondingPoints();

    // -------------------------------
    // Étape 0 : passage pixel → repère image « réel »
    // -------------------------------
    Point2f O;
    O.x = image.cols / 2.0f; // Om en pixels
    O.y = image.rows / 2.0f; // On en pixels

    float S = PIXEL_SIZE; // largeur physique d’un pixel (mm)

    vector<Point2f> realCorners; // points dans le repere image reelle
    realCorners.reserve(filtered.size());
    for (const Point2f &pt : filtered)
    {
        float xdi = (pt.x - O.x) * S;
        float ydi = (pt.y - O.y) * S;
        realCorners.emplace_back(xdi, ydi);
    }

    int M = static_cast<int>(realCorners.size());
    vector<Point3f> points3D = getPoints();

    // -------------------------------
    // Étape 1 de Tsai : construire A et b pour a1…a5
    // -------------------------------
    Mat A = Mat::zeros(M, 5, CV_64F);
    Mat b = Mat::zeros(M, 1, CV_64F);

    for (int i = 0; i < M; ++i)
    {
        double X = points3D[i].x;    // X_{s,i} (mm)
        double Y = points3D[i].y;    // Y_{s,i} (mm)
        double x = realCorners[i].x; // x_{d,i} (mm)
        double y = realCorners[i].y; // y_{d,i} (mm)

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

    // Trouver les Rc/Tcy, Tcx/Tcy, etc.
    double r11 = a_hat.at<double>(0, 0); // R11^c/Ty^c
    double r12 = a_hat.at<double>(1, 0); // R12^c/Ty^c
    double tx = a_hat.at<double>(2, 0);  // T_x^c/Ty^c
    double r21 = a_hat.at<double>(3, 0); // R21^c/Ty^c
    double r22 = a_hat.at<double>(4, 0); // R22^c/Ty^c

    // -------------------------------
    // Reconstruction partielle de R^c et T^c
    // -------------------------------
    // 1) Normaliser première ligne avec Ty1

    double Tyc = getTy(r11, r12, r21, r22);

    double R11 = r11 * Tyc;
    double R12 = r12 * Tyc;
    double R21 = r21 * Tyc;
    double R22 = r22 * Tyc;
    double Txc = tx * Tyc;

    double testX = R11 * points3D[0].x + R12 * points3D[0].y + Txc;
    double testY = R21 * points3D[0].x + R22 * points3D[0].y + Tyc;

    if (testX * realCorners[0].x < 0 || testY * realCorners[0].y < 0)
    {
        Tyc *= -1;
        R11 *= -1;
        R12 *= -1;
        R21 *= -1;
        R22 *= -1;
        Txc *= -1;
    }

    double S_bizarre = 1;
    if (R11 * R21 + R12 * R22 < 0)
        S_bizarre *= -1;

    double R13 = sqrt(1 - pow(R11, 2) - pow(R12, 2));
    double R23 = S_bizarre * sqrt(1 - pow(R21, 2) - pow(R22, 2));
    double R31 = (1 - pow(R11, 2) - R12 * R21) / R13;
    double R32 = (1 - R21 * R12 - pow(R22, 2)) / R23;
    double R33 = sqrt(1 - R31 * R13 - R32 * R23);

    // trouver z prime et Tzc
    int n = points3D.size();
    Mat C(n, 2, CV_64F);
    Mat D(n, 1, CV_64F);

    for (int i = 0; i < points3D.size(); i++)
    {
        double X = points3D[i].x;
        double Y = points3D[i].y;
        double x = realCorners[i].x;
        double y = realCorners[i].y;

        double Yi = R21 * X + R22 * Y + Tyc;
        double Wi = R31 * X + R32 * Y;

        C.at<double>(i, 0) = Yi - y;
        C.at<double>(i, 1) = 1.0;
        D.at<double>(i, 0) = Wi * y;
    }

    Mat solution;
    solve(C, D, solution, DECOMP_SVD);

    double zPrime = solution.at<double>(0, 0);
    double Tzc = solution.at<double>(1, 0);

    if (zPrime < 0)
    {
        zPrime *= -1;
        Tzc *= -1;
        R13 *= -1;
        R23 *= -1;
        R31 *= -1;
        R32 *= -1;
    }

    for (int i = 0; i < points3D.size(); i++)
    {
        double x = zPrime * ((R11 * points3D[i].x + R12 * points3D[i].y + Txc) / (R31 * points3D[i].x + R32 * points3D[i].y + Tzc));
        double y = zPrime * ((R21 * points3D[i].x + R22 * points3D[i].y + Tyc) / (R31 * points3D[i].x + R32 * points3D[i].y + Tzc));

        int x_pixel = static_cast<int>(x / PIXEL_SIZE + O.x);
        int y_pixel = static_cast<int>(y / PIXEL_SIZE + O.y);
        Point2f pt(x_pixel, y_pixel);
        circle(image, pt, 5, Scalar(0, 255, 0), -1);
    }

    double norm_row1 = std::sqrt(R11 * R11 + R12 * R12 + R13 * R13);
    double norm_row2 = std::sqrt(R21 * R21 + R22 * R22 + R23 * R23);
    double norm_row3 = std::sqrt(R31 * R31 + R32 * R32 + R33 * R33);

    cout << "row 1: " << R11 << " " << R12 << " " << R13 << " " << endl;
    cout << "row 2: " << R21 << " " << R22 << " " << R23 << " " << endl;
    cout << "row 3: " << R31 << " " << R32 << " " << R33 << " " << endl;

    imwrite("corners.png", image);
    waitKey(0);

    return 0;
}
