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

double getTy(double r11, double r12, double r21, double r22)
{
    double condition1 = r11 * r22 - r12 * r21;
    double condition2 = pow(r11, 2) + pow(r21, 2);
    double condition3 = pow(r12, 2) + pow(r22, 2);
    double condition4 = pow(r21, 2) + pow(r22, 2);
    double condition5 = pow(r11, 2) + pow(r12, 2);

    if (condition1 != 0)
    {
        double S = pow(r11, 2) + pow(r12, 2) + pow(r21, 2) + pow(r22, 2);
        return sqrt((S - sqrt(pow(S, 2) - 4 * pow(condition1, 2))) / (2 * pow(condition1, 2)));
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
    Mat image = imread("images/front.png");

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

    vector<Point2f> cornersPixelCoordinates = filterCorners(corners); // exclude wrong corners

    sort(cornersPixelCoordinates.begin(), cornersPixelCoordinates.end(), // Sort column by column
         [](const Point2f &a, const Point2f &b)
         {
             if (fabs(a.y - b.y) > 30)
                 return a.y < b.y;
             return a.x < b.x;
         });

    ofstream file("corners.txt"); // write points
    if (!file.is_open())
    {
        cout << "Could not open the file!" << endl;
        return -1;
    }

    for (const Point2f &pt : cornersPixelCoordinates)
    {
        file << pt.x << "," << pt.y << "," << "0" << endl;
    }
    file.close();

    Point2f O;
    O.x = image.cols / 2.0f; // Om en pixels
    O.y = image.rows / 2.0f; // On en pixels

    float S = PIXEL_SIZE; // largeur physique d’un pixel (mm)

    vector<Point3f> points3D = getPoints();

    vector<Point2f> realCorners; // points dans le repere image reelle
    realCorners.reserve(cornersPixelCoordinates.size());
    for (const Point2f &pt : cornersPixelCoordinates)
    {
        float xdi = (pt.x - O.x) * S;
        float ydi = (pt.y - O.y) * S;
        realCorners.emplace_back(xdi, ydi);
    }

    // -------------------------------
    // Étape 1 de Tsai : construire A et b pour a1…a5
    // -------------------------------
    Mat A = Mat::zeros(cornersPixelCoordinates.size(), 5, CV_64F);
    Mat b = Mat::zeros(cornersPixelCoordinates.size(), 1, CV_64F);

    for (int i = 0; i < cornersPixelCoordinates.size(); ++i)
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

    double signe = 1;
    if (R11 * R21 + R12 * R22 >= 0)
        signe *= -1;

    double R13 = sqrt(1 - pow(R11, 2) - pow(R12, 2));
    double R23 = signe * sqrt(1 - pow(R21, 2) - pow(R22, 2));
    double R31 = (1 - pow(R11, 2) - R12 * R21) / R13;
    double R32 = (1 - R21 * R12 - pow(R22, 2)) / R23;
    double R33 = sqrt(1 - R31 * R13 - R32 * R23);

    // trouver Tzc
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

        C.at<double>(i, 0) = Yi;
        C.at<double>(i, 1) = -y;
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

    double k1 = 0.0;        // on part de k1 = 0
    double zprime = zPrime; // valeur de l’étape 2.i (devrait être positive)
    double Tz_curr = Tzc;   // valeur de l’étape 2.i

    // Paramètres LM
    double tau = 1e-3;   // on choisit τ=10^-3
    double eps1 = 1e-8;  // tolérance sur ‖g‖∞
    double eps2 = 1e-12; // tolérance sur ‖Δp‖
    int maxIterLM = 200; // nombre max d’itérations
    double lambdaLM;     // mu
    double nu = 2.0;     // facteur d’ajustement

    // Construction du vecteur initial des résidus et Jacobien
    cv::Mat rVec; // (M x 1)
    cv::Mat Jmat; // (M x 3)

    // Calcul initial du Jacobien et du gradient pour p^(0)
    // On remplit rVec et Jmat via une fonction lambda :
    auto computeResidualsAndJacobian = [&](
                                           const double &k1_loc,
                                           const double &z_loc,
                                           const double &Tz_loc,
                                           cv::Mat &rOut, // (M x 1) résidus
                                           cv::Mat &JOut  // (M x 3) Jacobien
                                       )
    {
        // Préallouer la taille
        if (rOut.rows != points3D.size() || rOut.cols != 1)
            rOut = cv::Mat::zeros(points3D.size(), 1, CV_64F);
        if (JOut.rows != points3D.size() || JOut.cols != 3)
            JOut = cv::Mat::zeros(points3D.size(), 3, CV_64F);

        for (int i = 0; i < points3D.size(); ++i)
        {
            // 1) Coordonnées 3D sur la mire (Zs=0)
            double Xi = points3D[i].x;
            double Yi = points3D[i].y;
            double Zi = 0.0;

            // 2) Coordonnées « réelles » détectées (en mm)
            double xdi = static_cast<double>(realCorners[i].x);
            double ydi = static_cast<double>(realCorners[i].y);

            // 3) r_i^2
            double r2 = xdi * xdi + ydi * ydi;

            // 4) Num_i et Den_i
            double Num_i = R11 * Xi + R12 * Yi + Txc;    // = R^c_11 X + R^c_12 Y + T^c_x
            double Den_i = R31 * Xi + R32 * Yi + Tz_loc; // = R^c_31 X + R^c_32 Y + T^c_z

            // Éviter division par zéro
            if (fabs(Den_i) < 1e-12)
                Den_i = (Den_i >= 0 ? 1e-12 : -1e-12);

            // 5) résidu e_i(p) = xdi*(1 + k1_loc*r2)*Den_i  -  Num_i * z_loc
            double resid_i = xdi * (1.0 + k1_loc * r2) * Den_i - (Num_i * z_loc);
            rOut.at<double>(i, 0) = resid_i;

            // 6) calcul des dérivées partielles
            double d_e_d_k1 = xdi * r2 * Den_i;
            double d_e_d_z = -Num_i;
            double d_e_d_Tz = xdi * (1.0 + k1_loc * r2);

            JOut.at<double>(i, 0) = d_e_d_k1;
            JOut.at<double>(i, 1) = d_e_d_z;
            JOut.at<double>(i, 2) = d_e_d_Tz;
        }
    };

    // Premier calcul de rVec et Jmat en p^(0)
    computeResidualsAndJacobian(k1, zprime, Tz_curr, rVec, Jmat);

    // A = J^T J  (3×3),  g = J^T r  (3×1)
    cv::Mat A_lsq = Jmat.t() * Jmat; // Hessienne approchée (3×3)
    cv::Mat g_lsq = Jmat.t() * rVec; // gradient (3×1)

    // Initialisation de lambdaLM (mu)
    {
        double maxDiag = std::max({fabs(A_lsq.at<double>(0, 0)),
                                   fabs(A_lsq.at<double>(1, 1)),
                                   fabs(A_lsq.at<double>(2, 2))});
        lambdaLM = tau * maxDiag;
    }

    // Calcul initial du coût E_old = ‖e(p)‖²
    double E_old = 0.0;
    for (int i = 0; i < points3D.size(); ++i)
    {
        double rv = rVec.at<double>(i, 0);
        E_old += rv * rv;
    }

    bool found = false;
    int iter = 0;

    // Levenberg–Marquardt
    while (!found && iter < maxIterLM)
    {
        ++iter;

        // 1) résolution (A + lambda·I)·Δp = -g
        cv::Mat H_lm = A_lsq.clone();
        H_lm.at<double>(0, 0) += lambdaLM;
        H_lm.at<double>(1, 1) += lambdaLM;
        H_lm.at<double>(2, 2) += lambdaLM;

        cv::Mat minus_g = -1.0 * g_lsq; // (3×1)
        cv::Mat deltaP;                 // (3×1)

        bool solved = cv::solve(H_lm, minus_g, deltaP, cv::DECOMP_CHOLESKY);
        if (!solved)
        {
            cv::solve(H_lm, minus_g, deltaP, cv::DECOMP_SVD);
        }

        // 2) si le pas est trop petit, on arrête
        double norm_dp = sqrt(
            deltaP.at<double>(0, 0) * deltaP.at<double>(0, 0) +
            deltaP.at<double>(1, 0) * deltaP.at<double>(1, 0) +
            deltaP.at<double>(2, 0) * deltaP.at<double>(2, 0));
        double norm_p = sqrt(k1 * k1 + zprime * zprime + Tz_curr * Tz_curr);
        if (norm_dp <= eps2 * norm_p)
        {
            found = true;
            break;
        }

        // 3) proposer p_new = p + Δp
        double k1_new = k1 + deltaP.at<double>(0, 0);
        double zprime_new = zprime + deltaP.at<double>(1, 0);
        double Tz_new = Tz_curr + deltaP.at<double>(2, 0);

        // 4) calculer le nouveau coût E_new
        cv::Mat rVec_new, Jmat_dummy;
        computeResidualsAndJacobian(k1_new, zprime_new, Tz_new, rVec_new, Jmat_dummy);
        double E_new = 0.0;
        for (int i = 0; i < points3D.size(); ++i)
        {
            double rv = rVec_new.at<double>(i, 0);
            E_new += rv * rv;
        }

        // ρ = (E_old – E_new) / [ Δpᵀ (λ·Δp – g) ]
        cv::Mat temp = (lambdaLM * deltaP) - g_lsq; // (3×1)
        double denom_rho = deltaP.at<double>(0, 0) * temp.at<double>(0, 0) + deltaP.at<double>(1, 0) * temp.at<double>(1, 0) + deltaP.at<double>(2, 0) * temp.at<double>(2, 0);
        double rho = 0.0;
        if (fabs(denom_rho) > 1e-15)
            rho = (E_old - E_new) / denom_rho;
        else
            rho = -1.0; // pas valide

        // 5) décision : accepter ou rejeter le pas
        if (rho > 0.0)
        {
            // acceptation du pas
            k1 = k1_new;
            zprime = zprime_new;
            Tz_curr = Tz_new;

            // mise à jour de λ
            double factor = 1.0 - pow(2.0 * rho - 1.0, 3);
            factor = max(1.0 / 3.0, factor);
            lambdaLM = lambdaLM * factor;
            nu = 2.0;

            // mise à jour de E_old
            E_old = E_new;

            // recalculer rVec et Jmat, A_lsq, g_lsq
            computeResidualsAndJacobian(k1, zprime, Tz_curr, rVec, Jmat);
            A_lsq = Jmat.t() * Jmat;
            g_lsq = Jmat.t() * rVec;
        }
        else
        {
            // rejet du pas : on reste sur p
            lambdaLM = lambdaLM * nu;
            nu = nu * 2.0;
        }

        // 6) critère d’arrêt sur le gradient
        double maxAbs_g = max({fabs(g_lsq.at<double>(0, 0)),
                               fabs(g_lsq.at<double>(1, 0)),
                               fabs(g_lsq.at<double>(2, 0))});
        if (maxAbs_g < eps1)
        {
            found = true;
            break;
        }
    }

    double reprojectionErrorSum = 0.0;
    for (int i = 0; i < points3D.size(); ++i)
    {
        double Xi = points3D[i].x;
        double Yi = points3D[i].y;
        double Zi = 0.0;

        // projection linéaire “idéale”
        double Den_i = R31 * Xi + R32 * Yi + Tz_curr;
        if (fabs(Den_i) < 1e-12)
            Den_i = (Den_i >= 0 ? 1e-12 : -1e-12);

        double x_c = ((R11 * Xi + R12 * Yi + Txc) / Den_i) * zprime;
        double y_c = ((R21 * Xi + R22 * Yi + Tyc) / Den_i) * zprime;

        // on applique la distorsion radiale
        double r2_c = x_c * x_c + y_c * y_c;
        double x_d_pred = x_c * (1.0 + k1 * r2_c);
        double y_d_pred = y_c * (1.0 + k1 * r2_c);

        // retour en pixels
        double u = x_d_pred * (1.0 / S) + O.x;
        double v = y_d_pred * (1.0 / S) + O.y;

        int ui = static_cast<int>(round(u));
        int vi = static_cast<int>(round(v));
        circle(image, Point(ui, vi), 7, Scalar(0, 255, 0), -1);
        double dx = cornersPixelCoordinates[i].x - u;
        double dy = cornersPixelCoordinates[i].y - v;
        reprojectionErrorSum += dx * dx + dy * dy;
    }

    imwrite("corners.png", image);
    waitKey(0);

    return 0;
}
