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

    // Lecture images
    Mat Ig = imread("AloeG.png", IMREAD_COLOR);
    Mat Id = imread("AloeD.png", IMREAD_COLOR);

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
            double xg_r = (m_r - O_cam_g.x) * S_cam_g.x;
            double yg_r = (n_r - O_cam_g.y) * S_cam_g.y;
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
        
        }
    }

    // Image de droite Id
    // Boucle pour chaque pixel de l'image rectifié de droite

    return 0;
}