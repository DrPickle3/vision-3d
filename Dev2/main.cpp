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



    return 0;
}