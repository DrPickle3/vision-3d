#include <iostream>
#include <opencv2/opencv.hpp>

// Interpolation bilinéaire pour images couleur
cv::Vec3b interpolationBilineaire(const cv::Mat& img, double x, double y) {
    // Coordonnées des pixels voisins
    int x1 = static_cast<int>(std::floor(x));
    int y1 = static_cast<int>(std::floor(y));
    int x2 = x1 + 1;
    int y2 = y1 + 1;

    // Si on dépasse les bords de l’image, on renvoie noir
    if (x1 < 0 || x2 >= img.cols || y1 < 0 || y2 >= img.rows)
        return cv::Vec3b(0, 0, 0);

    // Distances entre la position réelle et x1,y1
    double dx = x - x1;
    double dy = y - y1;

    // quatre pixels encadrant (x,y)
    cv::Vec3b Q11 = img.at<cv::Vec3b>(y1, x1); // coin supérieur gauche
    cv::Vec3b Q21 = img.at<cv::Vec3b>(y1, x2); // coin supérieur droit
    cv::Vec3b Q12 = img.at<cv::Vec3b>(y2, x1); // coin inférieur gauche
    cv::Vec3b Q22 = img.at<cv::Vec3b>(y2, x2); // coin inférieur droit

    // Calculer de l’interpolation 
    cv::Vec3b resultat;
    for (int c = 0; c < 3; ++c) {
        double val = (1 - dx) * (1 - dy) * Q11[c]
                   + dx       * (1 - dy) * Q21[c]
                   + (1 - dx) * dy       * Q12[c]
                   + dx       * dy       * Q22[c];

        resultat[c] = static_cast<uchar>(cv::saturate_cast<uchar>(val)); // saturate_cast<uchar> s'assure que la valeur reste dans l’intervalle [0,255].
    }
    return resultat;
}


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
    // ouvre les deux images  Aloe
    cv::Mat img_g = cv::imread("images/Aloeg.png", cv::IMREAD_COLOR);
    cv::Mat img_d = cv::imread("images/AloeD.png", cv::IMREAD_COLOR);

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
    // Calcul de R et T pour la caméra gauche
    cv::Mat R = R_cam_d * R_cam_g.t();
    cv::Mat T = T_cam_g - R.t() * T_cam_d;

    // log pour voir les résultats
    std::cout << "Partie a) Affichage de R et T"<< std::endl;
    std::cout << "R:\n" << R << std::endl;
    std::cout << "T:\n" << T << std::endl;


    // Déclaration et initialisation de l'image rectifiée gauche

    cv::Mat img_rect_g = cv::Mat::zeros(img_g.size(), img_g.type());
    // pour tous pixel mr,nr de Ird
    for (int mr = 0; mr < img_g.rows; ++mr) {  // pour tous les pixels (m,n) de l'image droite
       for (int nr = 0; nr < img_g.cols; ++nr) {
            // a. Trouver la position dans ℝ3 du point rectifié
            //qg= (xgr, ygr,zʹ) xgr= (mr −Omr )⋅ Sxygr= (nr −Onr)⋅ Sy
            double xgr = (mr - O_cam_g.x) * S_cam_g.x;
            double ygr = (nr - O_cam_g.y) * S_cam_g.y;
            double zgr = zprime_cam_g;
            cv::Mat qg = (cv::Mat_<double>(3, 1) << xgr, ygr, zgr);
            // b. Rotation inverse (dans ℝ3)
            //Qg= Rgtqg= (Xgr,Ygr,Zgr)
            cv::Mat Qg = R_cam_g.t() * qg;
            // c. Reprojection sur le plan original (ℝ3)
            //pg=(z'd/Zgr)Qg=(xg,yg,zg′)
            cv::Mat pg = (1.0 / Qg.at<double>(2)) * Qg;  // pg = (z' / Zgr) * Qg
            // d. retrouver le pixel correspondant (dans R2): les coordonnées ne sont pas arrondies
            //m = (xg / Sx) + Om
            //n = (yg / Sy) + On
            double m = (pg.at<double>(0) / S_cam_g.x) + O_cam_g.x;
            double n = (pg.at<double>(1) / S_cam_g.y) + O_cam_g.y;
            //e. Interpoler (bilinéaire) la valeur Ig (m, n) à partir de ses voisins
            //f. Report de la valeur interpolée dans l'image rectifiée
            cv::Vec3b color = bilinear_interpolate(img_g, m, n);
            img_rect_g.at<cv::Vec3b>(mr, nr) = color;

       }
    }
    
    // Déclaration et initialisation de l'image rectifiée droite
    cv::Mat img_rect_d = cv::Mat::zeros(img_d.size(), img_d.type());
    // pour tous pixel mr,nr de Ird
    for (int mr = 0; mr < img_d.rows; ++mr) {
        for (int nr = 0; nr < img_d.cols; ++nr) {
            // a. Trouver la position dans ℝ3 du point rectifié
            //qd= (xdr, ydr,zʹ) xdr= (mr −Odr )⋅ Sxdr= (nr −Odr)⋅ Sy
            double xdr = (mr - O_cam_d.x) * S_cam_d.x;
            double ydr = (nr - O_cam_d.y) * S_cam_d.y;
            double zdr = zprime_cam_d;
            cv::Mat qd = (cv::Mat_<double>(3, 1) << xdr, ydr, zdr);
            // b. Rotation inverse (dans ℝ3)
            //Qd= Rdtqd= (Xdr,Ydr,Zdr)
            cv::Mat Qd = R_cam_d.t() * qd;
            // c. Reprojection sur le plan original (ℝ3)
            //pd=(z'd/Zdr)Qd=(xdr,ydr,z'd)
            cv::Mat pd = (1.0 / Qd.at<double>(2)) * Qd;  // pd = (z' / Zdr) * Qd
            // d. retrouver le pixel correspondant (dans R2): les coordonnées ne sont pas arrondies
            //m = (xdr / Sx) + Odr
            //n = (ydr / Sy) + Odr
            double m = (pd.at<double>(0) / S_cam_d.x) + O_cam_d.x;
            double n = (pd.at<double>(1) / S_cam_d.y) + O_cam_d.y;
            //e. Interpoler (bilinéaire) la valeur Id (m, n) à partir de ses voisins
            //f. Report de la valeur interpolée dans l'image rectifiée
            cv::Vec3b color = bilinear_interpolate(img_d, m, n);
            img_rect_d.at<cv::Vec3b>(mr, nr) = color;

        }
    }

    // enregistre l'image rectifiée gauche
    cv::imwrite("images/Aloeg_rectifiee.png", img_rect_g);
    // enregistre l'image rectifiée droite
    cv::imwrite("images/AloeD_rectifiee.png", img_rect_d);
    //tourner les images
    cv::Mat img_rect_g_rotated, img_rect_d_rotated;
    cv::rotate(img_rect_g, img_rect_g_rotated, cv::ROTATE_90_CLOCKWISE);
    cv::rotate(img_rect_d, img_rect_d_rotated, cv::ROTATE_90_CLOCKWISE);
    // enregistre l'image rectifiée gauche tournée
    cv::imwrite("images/Aloeg_rectifiee_rotated.png", img_rect_g_rotated);
    // enregistre l'image rectifiée droite tournée
    cv::imwrite("images/AloeD_rectifiee_rotated.png", img_rect_d_rotated);
    


    return 0;
}