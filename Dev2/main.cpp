#include <iostream>
#include <opencv2/opencv.hpp>

// Interpolation bilinéaire pour images couleur
cv::Vec3b interpolationBilineaire(const cv::Mat &image, double oldX, double oldY)
{
    if (oldX < -1 || oldX > image.cols || 
        oldY < -1 || oldY > image.rows)
    {
        // on est trop en dehors : on renvoie noir
        return cv::Vec3b(0, 0, 0);
    }

    // coins entiers autour de (oldX, oldY)
    int x1 = static_cast<int>(std::floor(oldX));
    int y1 = static_cast<int>(std::floor(oldY));
    int x2 = static_cast<int>(std::ceil(oldX));
    int y2 = static_cast<int>(std::ceil(oldY));

    double dx = oldX - x1;
    double dy = oldY - y1;

    cv::Vec3d acc(0,0,0);
    double sumW = 0.0;

    // Q11 : coin supérieur gauche
    if (x1 >= 0 && y1 >= 0) {
        double w = (1 - dx) * (1 - dy);
        acc += w * cv::Vec3d(image.at<cv::Vec3b>(y1, x1));
        sumW += w;
    }
    // Q21 : coin supérieur droit
    if (x2 < image.cols && y1 >= 0) {
        double w = dx * (1 - dy);
        acc += w * cv::Vec3d(image.at<cv::Vec3b>(y1, x2));
        sumW += w;
    }
    // Q12 : coin inférieur gauche
    if (x1 >= 0 && y2 < image.rows) {
        double w = (1 - dx) * dy;
        acc += w * cv::Vec3d(image.at<cv::Vec3b>(y2, x1));
        sumW += w;
    }
    // Q22 : coin inférieur droit
    if (x2 < image.cols && y2 < image.rows) {
        double w = dx * dy;
        acc += w * cv::Vec3d(image.at<cv::Vec3b>(y2, x2));
        sumW += w;
    }

    acc *= (1.0 / sumW);
    // on renvoie le pixel interpolé
    return cv::Vec3b(
        std::round(acc[0]),
        std::round(acc[1]),
        std::round(acc[2])
    );
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

// Fonction de rectification 
cv::Mat rectification(
    const cv::Mat &img,
    const cv::Mat &Rcam,
    const cv::Point2d &O_cam,
    const cv::Point2d &S_cam,
    double zprime_cam
)
{
    // Déclaration de l'image rectifiée
    cv::Mat img_rect = cv::Mat::zeros(img.size(), img.type());
    cv::Mat Rinv = Rcam.t();

    // pour tous pixel mr,nr de Ir
    for (int mr = 0; mr < img.rows; ++mr)
    {
        for (int nr = 0; nr < img.cols; ++nr)
        {
            // a. Trouver la position dans ℝ3 du point rectifié
            // q = (xr, yr, z')
            // xr = (nr - O_cam.x) ⋅ S_cam.x
            // yr = (mr - O_cam.y) ⋅ S_cam.y
            double xr = (nr - O_cam.x) * S_cam.x;
            double yr = (mr - O_cam.y) * S_cam.y;
            double zr = zprime_cam;
            cv::Mat q = (cv::Mat_<double>(3, 1) << xr, yr, zr);

            // b. Rotation inverse (dans ℝ3)
            cv::Mat Q = Rinv * q;

            // c. Reprojection sur le plan original (ℝ3)
            // p = (z' / Zr) * Q = (x, y, z')
            cv::Mat p = (zprime_cam / Q.at<double>(2)) * Q;

            // d. retrouver le pixel correspondant (dans R2): les coordonnées ne sont pas arrondies
            // m = (x / S_cam.x) + O_cam.x
            // n = (y / S_cam.y) + O_cam.y
            double m = (p.at<double>(0) / S_cam.x) + O_cam.x;
            double n = (p.at<double>(1) / S_cam.y) + O_cam.y;

            // e. Interpoler (bilinéaire) la valeur I(m, n) à partir de ses voisins
            // f. Report de la valeur interpolée dans l'image rectifiée
            img_rect.at<cv::Vec3b>(mr, nr) = interpolationBilineaire(img, m, n);
        }
    }
    return img_rect;
}

int main()
{
    // ouvre les deux images  Aloe
    cv::Mat img_g = cv::imread("images/AloeG.png", cv::IMREAD_COLOR);
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

    // Rectification des images
    cv::Mat img_rect_g = rectification(img_g, R_cam_g, O_cam_g, S_cam_g, zprime_cam_g);
    cv::Mat img_rect_d = rectification(img_d, R_cam_d, O_cam_d, S_cam_d, zprime_cam_d);
  
    // enregistre les images rectifiées
    cv::imwrite("images/AloeG_rectifiee.png", img_rect_g);
    cv::imwrite("images/AloeD_rectifiee.png", img_rect_d);

    // concatenation des images rectifiées pour pouvoir mieux verifier
    // trouver la taille maximale
    int max_rows = std::max(img_rect_g.rows, img_rect_d.rows);
    int total_cols = img_rect_g.cols + img_rect_d.cols;
    // créer une image vide de la taille maximale
    cv::Mat img_concat = cv::Mat::zeros(max_rows, total_cols, img_rect_g.type());
    // placer la gauche
    img_rect_g.copyTo(img_concat(cv::Rect(0, 0,
                                          img_rect_g.cols,
                                          img_rect_g.rows)));
    // placer la droite à droite de la gauche
    img_rect_d.copyTo(img_concat(cv::Rect(img_rect_g.cols, 0,
                                          img_rect_d.cols,
                                          img_rect_d.rows)));
    // enregistre l'image concaténée
    cv::imwrite("images/Aloe_rectifiee_concat.png", img_concat);

    // surtout pour verification personellle, fait aussi manuellement avec des lignes sur Gimp
    // tracer une ligne tous les 40 lignes
    int step = 40;
    for (int y = 0; y < img_concat.rows; y += step)
    {
        cv::line(img_concat,
                 cv::Point(0, y),
                 cv::Point(img_concat.cols, y),
                 cv::Scalar(0, 0, 255),
                 1, cv::LINE_AA);
    }

    cv::imwrite("images/verification.png", img_concat);

    return 0;
}