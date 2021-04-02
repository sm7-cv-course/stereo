#include "imagepairmatcher.h"
#include "imageshiftestimator.h"

#include <iostream>
#include <opencv/cv.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/opencv_modules.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/stereo.hpp>
#include <opencv2/calib3d.hpp>

using namespace std;

int ROI_SIZE = 16;
int ROI_STEP = 1;
int TEMPLATE_SIZE = 32;
int TEMPLATE_STEP = 1;
int Y_ADDITION = 4;

cv::Mat SobelFilter(cv::Mat img, cv::Mat &hor, cv::Mat &vert, cv::Mat &mag, cv::Size w = cv::Size(3,3))
{
    // Convert to float and adapt value range (for simplicity)
    img.convertTo(img, CV_32F, 1.0 / 255); // Value range: 0.0 - 1.0

    // Build data for 3x3 vertical Sobel kernel
    float sobelKernelHorizontalData[3][3] =
    {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };
    // Calculate normalization divisor/factor
    float sobelKernelNormalizationDivisor = 4.f;
    float sobelKernelNormalizationFactor = 1.f / sobelKernelNormalizationDivisor;

    // Generate cv::Mat for vertical filter kernel
    cv::Mat sobelKernelHorizontal =
        cv::Mat(3,3, CV_32F, sobelKernelHorizontalData); // Value range of filter result (if it is used for filtering): 0 - 4*255 or 0.0 - 4.0
    // Apply filter kernel normalization
    sobelKernelHorizontal *= sobelKernelNormalizationFactor; // Value range of filter result (if it is used for filtering): 0 - 255 or 0.0 - 1.0

    // Generate cv::Mat for horizontal filter kernel
    cv::Mat sobelKernelVertical;
    cv::transpose(sobelKernelHorizontal, sobelKernelVertical);

    // Apply two distinct Sobel filtering steps
    cv::Mat imgFilterResultVertical;
    cv::Mat imgFilterResultHorizontal;
    cv::filter2D(img, imgFilterResultVertical, CV_32F, sobelKernelVertical);
    cv::filter2D(img, imgFilterResultHorizontal, CV_32F, sobelKernelHorizontal);

    // Build overall filter result by combining the previous results
    cv::Mat imgFilterResultMagnitude;
    cv::magnitude(imgFilterResultVertical, imgFilterResultHorizontal, imgFilterResultMagnitude);

    // Write images to HDD. Important: convert back to uchar, otherwise we get black images
    cv::Mat imgFilterResultVerticalUchar;
    cv::Mat imgFilterResultHorizontalUchar;
    cv::Mat imgFilterResultMagnitudeUchar;
    imgFilterResultVertical.convertTo(imgFilterResultVerticalUchar, CV_8UC3, 255);
    imgFilterResultHorizontal.convertTo(imgFilterResultHorizontalUchar, CV_8UC3, 255);
    imgFilterResultMagnitude.convertTo(imgFilterResultMagnitudeUchar, CV_8UC3, 255);

    hor = imgFilterResultHorizontal;
    vert = imgFilterResultVertical;
    mag = imgFilterResultMagnitude;

    return imgFilterResultHorizontal;
}

// Very simple and not optimal correlator
int simple_corr(cv::Mat imgL, cv::Mat imgR, vector<Point2f> &l, vector<Point2f> &r, float max_shift)
{
    for (size_t i = 0; i < imgL.rows - Y_ADDITION; i++) {
        // take one row on te left image
        // cv::Rect roiRect = cv::Rect(0, i, imgL.cols, 1);

        for (size_t j = 0; j < imgR.cols - TEMPLATE_SIZE; j += TEMPLATE_STEP) {
            cv::Rect tRect = cv::Rect(j, i, TEMPLATE_SIZE, Y_ADDITION);

            float corr_max = 0;
            int index_max_l = 0, index_max_r = 0;
            cv::Mat corr;
            for (int k = 0; k < imgL.cols - TEMPLATE_SIZE; k += ROI_STEP) {
                cv::Rect roiRect = cv::Rect(k, i, TEMPLATE_SIZE, Y_ADDITION);

                //cv::imwrite("roi.png", imgL(roiRect));
                //cv::imwrite("template.png", imgR(tRect));

                cv::matchTemplate(imgL(roiRect), imgR(tRect), corr, cv::TM_CCOEFF_NORMED);
                if (corr.at<float>(0, 0) > corr_max) {
                    corr_max = corr.at<float>(0, 0);
                    index_max_r = j;
                    index_max_l = k;
                }
            }
            if (fabs(index_max_r - index_max_l) < max_shift * 1.5) {
                l.push_back(Point2f(index_max_l, i));
                r.push_back(Point2f(index_max_r, i));
            }
        }
    }

    return 0;
}

// Very simple and not optimal correlator
int simple_corr2(cv::Mat imgL, cv::Mat imgR, vector<Point2f> &l, vector<Point2f> &r, float max_shift)
{
    for (size_t i = 0; i < imgL.rows - Y_ADDITION; i++) {
        // take one row on te left image
        // cv::Rect roiRect = cv::Rect(0, i, imgL.cols, 1);

        cv::Rect roiRect = cv::Rect(0, i, imgL.cols, Y_ADDITION);
        cv::Mat roiImg = imgL(roiRect);

        for (size_t j = 0; j < imgR.cols - TEMPLATE_SIZE; j += TEMPLATE_STEP) {
            cv::Rect tRect = cv::Rect(j, i, TEMPLATE_SIZE, Y_ADDITION);

            cv::Mat corr;

            //cv::imwrite("roi.png", roiImg);
            //cv::imwrite("template.png", imgR(tRect));

            // Apply template Matching
            cv::matchTemplate(roiImg, imgR(tRect), corr, cv::TM_CCOEFF_NORMED);

            double minVal, maxVal;
            Point2i minLoc, maxLoc;
            cv::minMaxLoc(corr, &minVal, &maxVal, &minLoc, &maxLoc);

            if (fabs(maxLoc.x - j) < max_shift * 1.5) {
                l.push_back(Point2f(maxLoc.x, i));
                r.push_back(Point2f(j, i));
            }
        }
    }

    return 0;
}

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

    // cout << "disp_vec differences = ";

    for (size_t i = 0; i < l.size(); ++i) {
        if (fabs(l[i].y - r[i].y) < 2)
            out.push_back(l[i].x - r[i].x);
        // cout << "(" << (l[i].x - r[i].x) << ", " << (l[i].y - r[i].y) << ")";
    }

    return out;
}

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

    //cout << "disp_vec = ";

    for (int i = 0; i < l.size(); ++i) {
        out.at<uchar>(l[i].y, l[i].x) = disp_vec[i];
        // cout << " " << disp_vec[i];
    }

    return out;
}

cv::Mat applyKneighborSearch(cv::Mat imgL)
{
    cv::Mat out = cv::Mat::zeros(imgL.rows, imgL.cols, imgL.type());

    unsigned int max_neighbours = 1;
    cv::Mat indices, dists; // neither assume type nor size here !
    double radius = 50.0;

    cv::Mat_<float> features(0, 2);

    for (int i = 0; i < imgL.rows; ++i) {
        for (int j = 0; j < imgL.cols; ++j) {
            if (imgL.at<uchar>(i, j) == 0)
                continue;

            cout << " " << static_cast<int>(imgL.at<uchar>(i, j));
            cv::Mat row = (cv::Mat_<float>(1, 2) << i, j);
            features.push_back(row);
        }
    }

    cout << "features:" << features << endl;

    cv::flann::Index flann_index(features, cv::flann::KDTreeIndexParams(1));

    for (int i = 0; i < imgL.rows; ++i) {
        for (int j = 0; j < imgL.cols; ++j) {
            if (imgL.at<uchar>(i, j) != 0)
                continue;

            cv::Mat query = (cv::Mat_<float>(1, 2) << i, j);

            flann_index.radiusSearch(query, indices, dists, radius, max_neighbours,
                cv::flann::SearchParams(32));

            cout << "dists:" << dists << endl;

            float val = 0, nearestIndex = 0, minDist = imgL.cols;
            for (int k = 0; k < indices.cols; ++k) {
                if (minDist > dists.at<float>(0, k)) {
                    minDist = dists.at<float>(0, k);
                    nearestIndex = k;
                    // val = features.at<float>(0, k);
                    val = imgL.at<uchar>(features.at<float>(0, k), features.at<float>(1, k));
                }
            }

            cout << "val = " << val;
            out.at<float>(i, j) = val;
        }
    }

    return out;
}

cv::Mat applyKneighborSearch_simple(cv::Mat imgL)
{
    cv::Mat out = cv::Mat::zeros(imgL.rows, imgL.cols, imgL.type());

    unsigned int max_neighbours = 1;
    cv::Mat indices, dists; // neither assume type nor size here !
    double radius = 50.0;

    vector<Point2f> features;

    for (int i = 0; i < imgL.rows; ++i) {
        for (int j = 0; j < imgL.cols; ++j) {
            if (imgL.at<uchar>(i, j) == 0)
                continue;

            cout << " " << static_cast<int>(imgL.at<uchar>(i, j));
            features.push_back(Point2f(j, i));
        }
    }

    cout << "features:" << features << endl;

    cout << "retrived image values:" << endl;
    for (int k = 0; k < features.size(); ++k) {
        cout << " " << static_cast<int>(imgL.at<uchar>(features[k].y, features[k].x));
    }

    for (int i = 0; i < imgL.rows; ++i) {
        for (int j = 0; j < imgL.cols; ++j) {
            if (imgL.at<uchar>(i, j) != 0)
                out.at<uchar>(i, j) = imgL.at<uchar>(i, j);

            // Point2f query = Point2f(j, i);

            // Find distances from query to each features (brute force, not optimal!)
            float min_dist = std::max(2 * imgL.cols, 2 * imgL.rows); // dist cannot be greater
            int min_index = 0;
            for (int k = 0; k < features.size(); ++k) {
                float dist = sqrtf((features[k].x - j) * (features[k].x - j) + (features[k].y - i) * (features[k].y - i));
                if (dist < min_dist) {
                    min_dist = dist;
                    min_index = k;
                }
            }

            //cout << "dists: " << min_dist << "min_index: " << min_index << endl;

            out.at<uchar>(i, j) = imgL.at<uchar>(features[min_index].y, features[min_index].x);

            /*cout << "dists:" << dists << endl;

            float val = 0, nearestIndex = 0, minDist = imgL.cols;
            for (int k = 0; k < indices.cols; ++k) {
                if (minDist > dists.at<float>(0, k)) {
                    minDist = dists.at<float>(0, k);
                    nearestIndex = k;
                    // val = features.at<float>(0, k);
                    val = imgL.at<uchar>(features.at<float>(0, k), features.at<float>(1, k));
                }
            }*/

            // cout << "val = " << val;
            // out.at<float>(i, j) = val;
        }
    }

    return out;
}

int main()
{
    cout << "Hello World!" << endl;

    string path_img1 = "view1.png";
    string path_img2 = "view5.png";
    // string path_img1 = "scene1.row3.col1.ppm";
    // string path_img2 = "scene1.row3.col2.ppm";
    string out_map = "map.png";
    string out_map_sparse = "map_sparse.png";
    string out_map_sparse_interp = "map_sparse_interp.png";

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

    pointsL.clear();
    pointsR.clear();

    cv::Mat imgL_grad_v, imgL_grad_h, imgL_grad_mag;
    cv::Mat imgR_grad_v, imgR_grad_h, imgR_grad_mag;

    cv::Mat imgL_grad = SobelFilter(imgL, imgL_grad_v, imgL_grad_h, imgL_grad_mag);
    cv::Mat imgR_grad = SobelFilter(imgR, imgR_grad_v, imgR_grad_h, imgR_grad_mag);


    Y_ADDITION = 16;
    simple_corr2(imgL_grad_mag, imgR_grad_mag, pointsL, pointsR, shiftEstimator.getMaxShift().x);
    Y_ADDITION = 4;
    simple_corr2(imgL_grad_h, imgR_grad_h, pointsL, pointsR, shiftEstimator.getMaxShift().x);

    cv::Mat sparse = calcSparseDisparityMap(pointsL, pointsR, imgL);

    // cout << "sparse matrix" << sparse;

    cv::imwrite(out_map + "_T" + std::to_string(TEMPLATE_SIZE) + "_YA" + std::to_string(Y_ADDITION) + ".png", sparse);

    //cv::KDTree::KDTree tree;
    //auto kn = cv::ml::KNearest::create();

    /*cv::watershed(imgL, sparse);

    cv::imwrite(out_map_sparse, sparse);*/

    //cv::Mat sparse_interp = applyKneighborSearch_simple(sparse);

    //cout << sparse_interp;

    //cv::imwrite(out_map_sparse_interp, sparse_interp);

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
