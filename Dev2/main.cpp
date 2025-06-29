#include <iostream>
#include <fstream>
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

cv::Mat findR()
{
    cv::Mat Rgt = R_cam_g.t();
    return R_cam_d * Rgt;
}

cv::Mat findT(cv::Mat R)
{
    cv::Mat Rt = R.t();
    return T_cam_g - Rt * T_cam_d;
}

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
    return (cv::Mat_<double>(3, 3) << e11, e21, e31, e12, e22, e32, e13, e23, e33);
}

void rectify(cv::Mat image, cv::Mat image_rectified, cv::Point2d O, cv::Point2d S, cv::Mat Rg, double zprime)
{
    for (int y = 0; y < image.rows; y++)
    {
        for (int x = 0; x < image.cols; x++)
        {
            double pgx = (x - O.x) * S.x;
            double pgy = (y - O.y) * S.y;
            double pgz = zprime;
            cv::Mat pg = (cv::Mat_<double>(3, 1) << pgx, pgy, pgz);

            cv::Mat Pg = Rg * pg;

            cv::Mat qg = (pgz / Pg.at<double>(2, 0)) * Pg;

            int newX = qg.at<double>(0, 0) / S.x + O.x;
            int newY = qg.at<double>(1, 0) / S.y + O.y;

            if (newX >= 0 && newX < image.cols && newY >= 0 && newY < image.rows)
            {
                image_rectified.at<cv::Vec3b>(newY, newX) = image.at<cv::Vec3b>(y, x); // cv::Vec3b pcq c'est une image RGB
            }
        }
    }
}

void invert_rectify(cv::Mat image, cv::Mat image_rectified, cv::Point2d O, cv::Point2d S, cv::Mat R, double zprime)
{
    for (double y = 0; y < image_rectified.rows; y++)
    {
        for (double x = 0; x < image_rectified.cols; x++)
        {
            double qx = (x - O.x) * S.x;
            double qy = (y - O.y) * S.y;
            double qz = zprime;
            cv::Mat q = (cv::Mat_<double>(3, 1) << qx, qy, qz);

            cv::Mat Q = R.t() * q;

            cv::Mat p = (qz / Q.at<double>(2, 0)) * Q;

            double oldX = p.at<double>(0, 0) / S.x + O.x;
            double oldY = p.at<double>(1, 0) / S.y + O.y;

            if (oldX >= -1 && oldX < image.cols + 1 && oldY >= -1 && oldY < image.rows + 1)
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

    // R et T selon la preuve au dessus
    cv::Mat R = findR();
    cv::Mat T = findT(R);

    // Etape 1 de la rectification
    cv::Mat Rg_rectification = findRg(T);

    // Etape 2
    cv::Mat Rd_rectification = R.t() * Rg_rectification;

    // Etape 3 (pour chaque pixel de l'image de gauche)
    cv::Mat imageG = cv::imread("images/AloeG.png");
    cv::Mat imageD = cv::imread("images/AloeD.png");

    if (imageG.empty() || imageD.empty())
    {
        std::cout << "Could not open or find the image!" << std::endl;
        return -1;
    }

    cv::Mat imageG_rectified = cv::Mat::zeros(imageG.size(), imageG.type());
    cv::Mat imageD_rectified = cv::Mat::zeros(imageD.size(), imageD.type());

    // rectify(imageG, imageG_rectified, O_cam_g, S_cam_g, Rg_rectification, zprime_cam_g);
    // rectify(imageD, imageD_rectified, O_cam_d, S_cam_d, Rd_rectification, zprime_cam_d);
    invert_rectify(imageG, imageG_rectified, O_cam_g, S_cam_g, Rg_rectification, zprime_cam_g);
    invert_rectify(imageD, imageD_rectified, O_cam_d, S_cam_d, Rd_rectification, zprime_cam_d);

    cv::flip(imageG_rectified, imageG_rectified, -1); // Weird mais ma photo est flipped sur les 2 axes
    cv::imwrite("images/RectifiedG.png", imageG_rectified);

    cv::flip(imageD_rectified, imageD_rectified, -1);
    cv::imwrite("images/RectifiedD.png", imageD_rectified);

    return 0;
}