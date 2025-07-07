#include <iostream>
#include <fstream>
#include <tuple>
#include <opencv2/opencv.hpp>

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

cv::Mat findRg(cv::Mat T)
{
    // e1 = T/|T|
    double norm = cv::norm(T);
    double e11 = T.at<double>(0, 0) / norm;
    double e12 = T.at<double>(1, 0) / norm;
    double e13 = T.at<double>(2, 0) / norm;

    // e2 = (-Ty, Tx, 0) / |(-Ty, Tx, 0)|        (en dehors de l'ecran)
    double norm2 = std::sqrt(std::pow(T.at<double>(1, 0), 2) + std::pow(T.at<double>(0, 0), 2));
    double e21 = -T.at<double>(1, 0) / norm2;
    double e22 = T.at<double>(0, 0) / norm2;
    double e23 = 0;

    // e3 = e1 X e2
    double e31 = e12 * e23 - e22 * e13;
    double e32 = -(e11 * e23 - e21 * e13);
    double e33 = e11 * e22 - e21 * e12;
    return (cv::Mat_<double>(3, 3) << e11, e12, e13, e21, e22, e23, e31, e32, e33);
}

std::tuple<cv::Size, cv::Point2d> getParameters(cv::Mat imageG, cv::Point2d Og, cv::Point2d Sg, cv::Mat Rg, double zprimeG, cv::Mat imageD, cv::Point2d Od, cv::Point2d Sd, cv::Mat Rd, double zprimeD)
{
    std::vector<cv::Point2d> cornersG;

    for (int y = 0; y < 2; y++)
    {
        for (int x = 0; x < 2; x++)
        {
            double m = x * (imageG.cols - 1); // Coins de l'image originale gauche
            double n = y * (imageG.rows - 1);

            double px = (m - Og.x) * Sg.x;
            double py = (n - Og.y) * Sg.y;
            double pz = zprimeG;

            cv::Mat p = (cv::Mat_<double>(3, 1) << px, py, pz); // point image
            cv::Mat P = Rg * p;                                 // point scene
            cv::Mat q = (pz / P.at<double>(2, 0)) * P;

            int newX = q.at<double>(0, 0) / Sg.x + Og.x;
            int newY = q.at<double>(1, 0) / Sg.y + Og.y;

            cornersG.emplace_back(newX, newY); // On conserve les 4 coins projetes
        }
    }

    double minXg = cornersG[0].x;
    double maxXg = cornersG[0].x;
    double minYg = cornersG[0].y;
    double maxYg = cornersG[0].y;

    for (const auto &pt : cornersG) // On prends les min et les max des axes
    {
        if (pt.x < minXg)
            minXg = pt.x;
        if (pt.x > maxXg)
            maxXg = pt.x;
        if (pt.y < minYg)
            minYg = pt.y;
        if (pt.y > maxYg)
            maxYg = pt.y;
    }

    int widthG = static_cast<int>(std::round(maxXg - minXg)); // Width et height du canvas pour que l'image projetee fit parfaitement
    int heightG = static_cast<int>(std::round(maxYg - minYg));

    std::vector<cv::Point2d> cornersD;

    for (int y = 0; y < 2; y++) // Meme chose pour l'image de droite
    {
        for (int x = 0; x < 2; x++)
        {
            double m = x * (imageD.cols - 1); // Coins de l'image originale
            double n = y * (imageD.rows - 1);

            double px = (m - Od.x) * Sd.x;
            double py = (n - Od.y) * Sd.y;
            double pz = zprimeD;

            cv::Mat p = (cv::Mat_<double>(3, 1) << px, py, pz); // point image
            cv::Mat P = Rd * p;                                 // point scene
            cv::Mat q = (pz / P.at<double>(2, 0)) * P;

            int newX = q.at<double>(0, 0) / Sd.x + Od.x;
            int newY = q.at<double>(1, 0) / Sd.y + Od.y;

            cornersD.emplace_back(newX, newY);
        }
    }

    double minXd = cornersD[0].x;
    double maxXd = cornersD[0].x;
    double minYd = cornersD[0].y;
    double maxYd = cornersD[0].y;

    for (const auto &pt : cornersD)
    {
        if (pt.x < minXd)
            minXd = pt.x;
        if (pt.x > maxXd)
            maxXd = pt.x;
        if (pt.y < minYd)
            minYd = pt.y;
        if (pt.y > maxYd)
            maxYd = pt.y;
    }

    int widthD = static_cast<int>(std::round(maxXd - minXd));
    int heightD = static_cast<int>(std::round(maxYd - minYd));

    int height = std::max(heightG, heightD); // height et width commune (on veut le meme nombre de rangee pour faire la correspondance)
    int width = std::max(widthG, widthD);

    double maxX = std::max(maxXg, maxXd); // On prend le min et max de l'image jumelee
    double minX = std::min(minXg, minXd);

    double Ox = (maxX - minX) / 2; // Le nouveau O est au centre de cette image jumelee
    double Oy = height / 2;

    return std::make_tuple(cv::Size(width, height), cv::Point2d(Ox, Oy)); // On retourne les parametres de la camera rectifiee
}

void rectify(cv::Mat image, cv::Mat image_rectified, cv::Point2d O, cv::Point2d S, cv::Mat R, double zprime)
{
    for (int y = 0; y < image.rows; y++) // Rectification avec lignes noires (mauvais)
    {
        for (int x = 0; x < image.cols; x++)
        {
            double px = (x - O.x) * S.x;
            double py = (y - O.y) * S.y;
            double pz = zprime;
            cv::Mat p = (cv::Mat_<double>(3, 1) << px, py, pz);

            cv::Mat P = R * p;

            cv::Mat q = (pz / P.at<double>(2, 0)) * P;

            int newX = q.at<double>(0, 0) / S.x + O.x;
            int newY = q.at<double>(1, 0) / S.y + O.y;

            if (newX >= 0 && newX < image.cols && newY >= 0 && newY < image.rows)
            {
                image_rectified.at<cv::Vec3b>(newY, newX) = image.at<cv::Vec3b>(y, x); // cv::Vec3b pcq c'est une image RGB
            }
        }
    }
}

void invert_rectify(cv::Mat image, cv::Mat image_rectified, cv::Point2d oldO, cv::Point2d newO, cv::Point2d S, cv::Mat R, double zprime)
{
    for (double y = 0; y < image_rectified.rows; y++)
    {
        for (double x = 0; x < image_rectified.cols; x++)
        {
            double qx = (x - newO.x) * S.x;
            double qy = (y - newO.y) * S.y;
            double qz = zprime;
            cv::Mat q = (cv::Mat_<double>(3, 1) << qx, qy, qz);

            cv::Mat Q = R.t() * q;

            cv::Mat p = (qz / Q.at<double>(2, 0)) * Q;

            double oldX = p.at<double>(0, 0) / S.x + oldO.x; // transformation inverse vers image originale
            double oldY = p.at<double>(1, 0) / S.y + oldO.y;

            if (oldX >= -1 && oldX < image.cols + 1 && oldY >= -1 && oldY < image.rows + 1) // Interpolation bilineaire
            {
                cv::Point2d p1(std::floor(oldX), std::floor(oldY));
                cv::Point2d p2(std::floor(oldX), std::ceil(oldY));
                cv::Point2d p3(std::ceil(oldX), std::floor(oldY));
                cv::Point2d p4(std::ceil(oldX), std::ceil(oldY));

                cv::Vec3b value(0, 0, 0);

                double dx = oldX - std::floor(oldX);
                double dy = oldY - std::floor(oldY);

                if (p1.x >= 0 && p1.y >= 0)
                {
                    double k1 = (1 - dx) * (1 - dy);
                    cv::Vec3b c = image.at<cv::Vec3b>(p1.y, p1.x);
                    value += k1 * cv::Vec3d(c);
                }
                if (p2.x >= 0 && p2.y < image.rows)
                {
                    double k2 = (1 - dx) * dy;
                    cv::Vec3b c = image.at<cv::Vec3b>(p2.y, p2.x);
                    value += k2 * cv::Vec3d(c);
                }
                if (p3.x < image.cols && p3.y >= 0)
                {
                    double k3 = dx * (1 - dy);
                    cv::Vec3b c = image.at<cv::Vec3b>(p3.y, p3.x);
                    value += k3 * cv::Vec3d(c);
                }
                if (p4.x < image.cols && p4.y < image.rows)
                {
                    double k4 = dx * dy;
                    cv::Vec3b c = image.at<cv::Vec3b>(p4.y, p4.x);
                    value += k4 * cv::Vec3d(c);
                }

                image_rectified.at<cv::Vec3b>(y, x) = value;
            }
        }
    }
}

int main()
{
    /*
    R et T theorique
        RgPg + Tg = RdPd + Td

    Isolons Pg:
        => RgPg = RdPd + Td - Tg
        => RgtRgPg = Rgt(RdPd + Td - Tg)         (Rg**-1 = Rgt)
        => Pg = Rgt(RdPd + RdRdt(Td - Tg))
        => Pg = RgtRd(Pd + Rdt(Td-Tg))     Posons R = RgtRd et T = Rdt(Td-Tg)
        => Pg = R(Pd + T)
    */

    // R et T selon la preuve au dessus
    cv::Mat R = R_cam_g.t() * R_cam_d;
    cv::Mat T = R_cam_d.t() * (T_cam_d - T_cam_g);

    // Etape 1 de la rectification
    cv::Mat Rg_rectification = findRg(T);
    cv::Mat Rd_rectification = R.t() * Rg_rectification;

    cv::Mat imageG = cv::imread("images/AloeG.png");
    cv::Mat imageD = cv::imread("images/AloeD.png");

    if (imageG.empty() || imageD.empty())
    {
        std::cout << "Could not open or find the image!" << std::endl;
        return -1;
    }

    cv::Mat imageG_blackLined = cv::Mat::zeros(imageG.size(), imageG.type());
    cv::Mat imageD_blackLined = cv::Mat::zeros(imageD.size(), imageD.type());

    rectify(imageG, imageG_blackLined, O_cam_g, S_cam_g, Rg_rectification, zprime_cam_g);
    rectify(imageD, imageD_blackLined, O_cam_d, S_cam_d, Rd_rectification, zprime_cam_d);

    cv::imwrite("images/RectifiedG.png", imageG_blackLined);
    cv::imwrite("images/RectifiedD.png", imageD_blackLined);

    // Etape 2 de la grille (Trouver les parametres de la camera rectifiee)
    auto [size, newO] = getParameters(imageG, O_cam_g, S_cam_g, Rg_rectification, zprime_cam_g, imageD, O_cam_d, S_cam_d, Rd_rectification, zprime_cam_d);

    cv::Mat imageG_rectified = cv::Mat::zeros(size, imageG.type()); // Images vides de la bonne taille
    cv::Mat imageD_rectified = cv::Mat::zeros(size, imageD.type());

    invert_rectify(imageG, imageG_rectified, O_cam_g, newO, S_cam_g, R_cam_g, zprime_cam_g); // Rectification inverse
    invert_rectify(imageD, imageD_rectified, O_cam_d, newO, S_cam_d, R_cam_d, zprime_cam_d);

    cv::imwrite("images/InvertRectifiedG.png", imageG_rectified);
    cv::imwrite("images/InvertRectifiedD.png", imageD_rectified);

    cv::Mat merge;
    cv::hconcat(imageG_rectified, imageD_rectified, merge);
    cv::imwrite("images/Rectified.png", merge); // On met les images cotes a cotes pour mieux les verifier sur Gimp

    return 0;
}