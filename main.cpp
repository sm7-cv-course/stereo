#include "imagepairmatcher.h"
#include "imageshiftestimator.h"

#include <iostream>
#include <opencv/cv.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/stereo.hpp>
#include <opencv2/calib3d.hpp>

using namespace std;

void stereo_depth_builder(const string &path_img1, const string &path_img2, const string &out_map)
{
    cv::Mat imgL = cv::imread(path_img1, 0);
    cv::Mat imgR = cv::imread(path_img2, 0);
    cv::Mat disparity, out;

    cv::imshow("1", imgL);
    cv::imshow("2", imgR);

    auto stereoBM = cv::StereoBM::create();//16, 15);
    stereoBM->compute(imgL, imgR, disparity);

    cv::normalize(disparity, out, 0, 255, cv::NORM_MINMAX);

    cv::imshow("3", out);

    cv::imwrite(out_map, out);

    char key = char (cv::waitKey(0));
    if (key == 27)
        return;
}

vector<float> calcDisparityValues(vector<Point2f> l, vector<Point2f> r)
{
    if (l.size() != r.size())
        return {};

    vector<float> out;

    for (size_t i = 0; i < l.size(); ++i) {
        if (fabs(l[i].y - r[i].y) < 1)
            out.push_back(l[i].x - r[i].x);
        out.push_back(l[i].x - r[i].x);

vector<float> normailize(const vector<float> &vec, float min = 0, float max = 255)
{
    vector<float> out;

    float min_in_val = vec[0];
    float max_in_val = vec[0];

    for (size_t i = 0; i < vec.size(); ++i) {
        if (min_in_val > vec[i]) {
            min_in_val = vec[i];
        }

        if (max_in_val < vec[i]) {
            max_in_val = vec[i];
        }
    }

    for (size_t i = 0; i < vec.size(); ++i) {
        out.push_back((max - min) * (vec[i] - min_in_val) / (max_in_val - min_in_val) + min);
    }

    return out;
}

cv::Mat calcSparseDisparityMap(vector<Point2f> l, vector<Point2f> r, cv::Mat imgL)
{
    cv::Mat out = cv::Mat::zeros(imgL.rows, imgL.cols, imgL.type());

    vector<float> disp_vec = calcDisparityValues(l, r);

    disp_vec = normailize(disp_vec);
    for (int i = 0; i < l.size(); ++i) {
        out.at<uchar>(l[i].y, l[i].x) = disp_vec[i];
    }

    return out;
}

int main()
{
    cout << "Hello World!" << endl;

    // string path_img1 = "view1.png";
    // string path_img2 = "view5.png";
    string path_img1 = "scene1.row3.col1.ppm";
    string path_img2 = "scene1.row3.col2.ppm";
    string out_map = "map.png";
    string out_map_sparse = "map_sparse.png";

    cv::Mat imgL = cv::imread(path_img1, 0);
    cv::Mat imgR = cv::imread(path_img2, 0);

    SURFImagePairMatcher matcher;

    matcher.match(imgL, imgR);

    std::vector<Point2f> pointsL, pointsR;
    cv::KeyPoint::convert(matcher.getPoints("L"), pointsL);
    cv::KeyPoint::convert(matcher.getPoints("R"), pointsR);

    ImageShiftEstimator shiftEstimator;
    shiftEstimator.setPoints(pointsL, pointsR);
    shiftEstimator.run();

    cv::Mat sparse = calcSparseDisparityMap(pointsL, pointsR, imgL);

    cv::imwrite(out_map_sparse, sparse);

    cv::watershed(imgL, sparse);

    /*cv::watershed(imgL, sparse);

    cv::imwrite(out_map_sparse, sparse);*/
    cv::imwrite(out_map_sparse, sparse);

    cout << shiftEstimator.estimatedShift();

    return 0;
}

/*int main()
{
    cout << "Hello World!" << endl;

    // string path_img1 = "view1.png";
    // string path_img2 = "view5.png";
    string path_img1 = "scene1.row3.col1.ppm";
    string path_img2 = "scene1.row3.col2.ppm";
    string out_map = "map.png";

    stereo_depth_builder(path_img1, path_img2, out_map);

    return 0;
}*/
