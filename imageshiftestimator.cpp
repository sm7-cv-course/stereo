#include "imageshiftestimator.h"
#include <iostream>

using namespace cv;
using namespace cv::xfeatures2d;
using std::cout;
using std::endl;

ImageShiftEstimator::ImageShiftEstimator()
{

}

/*template<typename T>
SURFImageShiftEstimator<T>::SURFImageShiftEstimator()
{

}*/

// Returns 0 on success
int ImageShiftEstimator::run()
{
    int N = m_pointsL.size();
    if (m_pointsL.size() != m_pointsR.size())
        return -1;

    // Mean point
    Point2f meanL, meanR;
    //double meanYshift = 0, meanXshift = 0;
    for (size_t i = 0; i < N; ++i) {
        meanL += m_pointsL[i];
        meanR += m_pointsR[i];
    }
    cout << meanL << endl;

    meanL /= N;
    //meanL.y /= N;
    meanR /= N;
    //meanR.y /= N;

    cout << meanL << endl;

    // Estimate shifts and apply median filter for Y
    vector<float> Yshifts, Xshifts;
    for (size_t i = 0; i < m_pointsL.size(); ++i) {
        Yshifts.push_back(m_pointsL[i].y - m_pointsR[i].y);
        Xshifts.push_back(m_pointsL[i].x - m_pointsR[i].x);
    }

    std::sort(std::begin(Yshifts), std::end(Yshifts));
    std::sort(std::begin(Xshifts), std::end(Xshifts));

    m_shift.y = Yshifts[static_cast<int>(static_cast<float>(Yshifts.size()) / 2.0)];
    m_shift.x = Xshifts[Xshifts.size() / 2.0];

    m_min_shift = Point2f(Xshifts.front(), Yshifts.front());
    m_max_shift = Point2f(Xshifts.back(), Yshifts.back());

    return 0;
}

vector<float> ImageShiftEstimator::calcDisparityValues()
{
    vector<float> out;

    for (size_t i = 0; i < m_pointsL.size(); ++i) {
        out.push_back(m_pointsL[i].x - m_pointsR[i].x);
    }

    return out;
}

/*cv::Mat ImageShiftEstimator::calcSparseDisparityMap()
{
    cv::Mat out = cv::Mat::zeros();

    vector<float> disp_vec = calcDisparityValues();

    for (int i = 0; i < m_pointsL.size(); ++i) {
        out.at<uchar>(m_pointsL[i].y, m_pointsL[i].x) = disp_vec[i];
    }

    return out;
}*/

//template int ImageShiftEstimator<cv::Point2f>::run();

/*template<typename T>
int SIFTImageShiftEstimator<T>::run()
{

}*/
