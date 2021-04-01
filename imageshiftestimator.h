#ifndef IMAGESHIFTESTIMATOR_H
#define IMAGESHIFTESTIMATOR_H

#include <vector>

#include <opencv/cv.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/stereo.hpp>
#include <opencv2/calib3d.hpp>
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace std;
using namespace cv;

class ImageShiftEstimator
{
public:
    explicit ImageShiftEstimator();

    int run();

    // Returns Y shift if it is near the same for every input point pairs
    //virtual int estimatedXshift();

    // Returns Filteres LSM value of input point pairs in X dimension
    //virtual int estimatedYshift();

    // Median value
    Point2f estimatedShift() { return m_shift; }

    Point2f getMaxShift() { return m_max_shift; }
    Point2f getMinShift() { return m_min_shift; }

    // template<typename T>
    void setPoints(vector<Point2f> l, vector<Point2f> r) { m_pointsL = l, m_pointsR = r; }

    vector<float> calcDisparityValues();

    cv::Mat calcSparseDisparityMap();

protected:
    vector<Point2f> m_pointsL, m_pointsR;
    Point2f m_shift, m_min_shift, m_max_shift;
};

//template class ImageShiftEstimator<cv::Point2f>;
/*template<typename T>
class SURFImageShiftEstimator : public ImageShiftEstimator<T>
{
public:
    explicit SURFImageShiftEstimator();
    virtual ~SURFImageShiftEstimator() {}

    int run() override;

    // Returns Y shift if it is near the same for every input point pairs
    //int estimatedXshift() override;

    // Returns Filteres LSM value of input point pairs in X dimension
    //int estimatedYshift() override;
};*/


/*template<typename T>
class SIFTImageShiftEstimator : public ImageShiftEstimator<T>
{
public:
    SIFTImageShiftEstimator() {}

    virtual int run() override;

    // Returns Y shift if it is near the same for every input point pairs
    //virtual int estimatedXshift() override;

    // Returns Filteres LSM value of input point pairs in X dimension
    //virtual int estimatedYshift() override;
};*/

#endif // IMAGESHIFTESTIMATOR_H
