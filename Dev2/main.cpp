#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

// Caméra de gauche
cv::Mat R_cam_g = (cv::Mat_<double>(3, 3) << 0.9962, 0, -0.0872,
    0, 0.9962, 0,
    0.0872, 0, 0.9962);

cv::Mat T_cam_g = (cv::Mat_<double>(3, 1) << 0, 0, 0);
cv::Point2d O_cam_g(538.625, 510.471);
cv::Point2d S_cam_g(0.00155227, 0.00155227);
double zprime_cam_g = 1.0;

// Caméra de droite
cv::Mat R_cam_d = (cv::Mat_<double>(3, 3) << 0.9962, 0, 0.0872,
    0, 0.9962, 0,
    -0.0872, 0, 0.9962);

cv::Mat T_cam_d = (cv::Mat_<double>(3, 1) << 5, 0, 0);
cv::Point2d O_cam_d(765.134, 510.599);
cv::Point2d S_cam_d(0.00155227, 0.00155227);
double zprime_cam_d = 1.0;

// Je t'ai volé ça Cam car flemme not gonna lie
Mat findRg(Mat T)
{
    // e1 = T/|T|
    double norm = cv::norm(T);
    double e11 = T.at<double>(0, 0) / norm;
    double e12 = T.at<double>(1, 0) / norm;
    double e13 = T.at<double>(2, 0) / norm;

    // e2 = (-Ty, Tx, 0) / |(-Ty, Tx, 0)|        (en dehors de l'ecran)
    double norm2 = sqrt(pow(T.at<double>(1, 0), 2) + pow(T.at<double>(0, 0), 2));
    double e21 = -T.at<double>(1, 0) / norm2;
    double e22 = T.at<double>(0, 0) / norm2;
    double e23 = 0;

    // e3 = e1 X e2
    double e31 = e12 * e23 - e22 * e13;
    double e32 = -(e11 * e23 - e21 * e13);
    double e33 = e11 * e22 - e21 * e12;
    return (Mat_<double>(3, 3) << e11, e12, e13, e21, e22, e23, e31, e32, e33);
}

Vec3b interpolationBilineaire(const Mat& image, double x, double y)
{
    if (x < 0 || x >= image.cols - 1 || y < 0 || y >= image.rows - 1)
        return Vec3b(0, 0, 0); // noir si hors image

    int x1 = floor(x), y1 = floor(y);
    int x2 = x1 + 1, y2 = y1 + 1;
    double dx = x - x1, dy = y - y1;

    Vec3b I00 = image.at<Vec3b>(y1, x1);
    Vec3b I10 = image.at<Vec3b>(y1, x2);
    Vec3b I01 = image.at<Vec3b>(y2, x1);
    Vec3b I11 = image.at<Vec3b>(y2, x2);

    Vec3b result;
    for (int c = 0; c < 3; c++)
        result[c] = static_cast<uchar>(
            (1 - dx) * (1 - dy) * I00[c] +
            dx * (1 - dy) * I10[c] +
            (1 - dx) * dy * I01[c] +
            dx * dy * I11[c]
            );

    return result;
}


int main()
{
    /*
    R et T theorique

    Nous voulons Pd = R(Pg - T)
    Nous avons Pg = RgPs + Tg et Pd = RdPs + Td

    Isolons Ps de g:
        Ps = Rg**-1 * Pg - Tg           (Rg**-1 = Rg**t)
    Remplacons dans la 2e equation:
        Pd = RdRg**t (Pg - Tg) + Td
    Posons R = RdRg**t:
        Pd = R(Pg - Tg) + Td
     => Pd = R(Pg - Tg) + RR**-1 * Td
     => Pd = R(Pg - Tg + R**t * Td)
    Posons T = Tg - R**t*Td:
        Pd = R(Pg - T)      ou R = RdRg**t et T = Tg - R**t*Td
    */

    // Calcul R = Rd * Rg^T
    Mat R = R_cam_d * R_cam_g.t();

    // Calcul T = Tg - R^T * Td
    Mat T = T_cam_g - R.t() * T_cam_d;

    // Test Affichage des paramètres extrinsèques
    cout << "===== Paramètres extrinsèques du système de stéréo =====" << endl;
    cout << "Matrice de rotation R :" << endl
        << R << endl
        << endl;
    cout << "Vecteur de translation T :" << endl
        << T << endl;

    // b

    // Construire Rg
    /*
    double Tx = T.at<double>(0, 0);
    double Ty = T.at<double>(1, 0);

    Mat num_e2 = (Mat_<double>(3, 1) << Tx, Ty, 0);
    double denom_e2 = sqrt(Tx * Tx + Ty * Ty);

    Mat e1 = T / norm(T);
    Mat e2 = num_e2 / denom_e2;
    Mat e3 = e1.cross(e2);

    Mat Rg_r;
    hconcat(vector<Mat>{e1, e2, e3}, Rg_r);

    // Construire Rd
    Mat Rd_r = R.t() * Rg_r;
    */
    Mat Rg_r = findRg(T_cam_d);
    Mat Rd_r = R * Rg_r;


    // Lecture images
    Mat Ig = imread("images/AloeG.png", IMREAD_COLOR);
    Mat Id = imread("images/AloeD.png", IMREAD_COLOR);

    if (Ig.empty() || Id.empty())
    {
        cerr << "Erreur de chargement des images." << endl;
        return -1;
    }

    // Création images rectifiées
    Mat Ig_r = Mat::zeros(Ig.size(), Ig.type());
    Mat Id_r = Mat::zeros(Id.size(), Id.type());

    // Image de gauche Ig
    // Boucle pour chaque pixel de l'image rectifié de gauche
    for (int m_r = 0; m_r < Ig.rows; m_r++)
    {
        for (int n_r = 0; n_r < Ig.cols; n_r++)
        {
            // qg point dans image de gauche rectifiée
            double xg_r = (n_r - O_cam_g.x) * S_cam_g.x;
            double yg_r = (m_r - O_cam_g.y) * S_cam_g.y;
            double zg_r = zprime_cam_g;

            Mat qg = (Mat_<double>(3, 1) << xg_r, yg_r, zg_r);

            // rotation inverse dans R3
            Mat Qg = Rg_r.t() * qg;

            // reprojection sur le plan original R3
            Mat pg = (zg_r / Qg.at<double>(2, 0)) * Qg;

            // retrouver pixel correspondant (R2) - coord. non arrondies
            double m = (pg.at<double>(0, 0) / S_cam_g.x) + O_cam_g.x;
            double n = (pg.at<double>(1, 0) / S_cam_g.y) + O_cam_g.y;

            // interpolation bilinéaire valeur Ig (m, n) \ partir de ses voisins
            // remplir image rectifiée
            Ig_r.at<Vec3b>(m_r, n_r) = interpolationBilineaire(Ig, n, m);

        }
    }

    // Image de droite Id
    // Boucle pour chaque pixel de l'image rectifié de droite
    for (int m_r = 0; m_r < Id.rows; m_r++)
    {
        for (int n_r = 0; n_r < Id.cols; n_r++)
        {
            double xd_r = (n_r - O_cam_d.x) * S_cam_d.x;
            double yd_r = (m_r - O_cam_d.y) * S_cam_d.y;
            double zd_r = zprime_cam_d;

            Mat qd = (Mat_<double>(3, 1) << xd_r, yd_r, zd_r);
            Mat Qd = Rd_r.t() * qd;
            Mat pd = (zd_r / Qd.at<double>(2, 0)) * Qd;

            double m = (pd.at<double>(0, 0) / S_cam_d.x) + O_cam_d.x;
            double n = (pd.at<double>(1, 0) / S_cam_d.y) + O_cam_d.y;

            Id_r.at<Vec3b>(m_r, n_r) = interpolationBilineaire(Id, n, m);
        }
    }

    imwrite("images/AloeG_rectifiee.png", Ig_r);
    imwrite("images/AloeD_rectifiee.png", Id_r);
    cout << "Images sauvegardées avec succès !" << endl;

    return 0;
}