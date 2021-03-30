#ifndef IMAGEPAIRMATCHER_H
#define IMAGEPAIRMATCHER_H

#include <string>
#include <iostream>
#include <opencv/cv.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include "opencv2/features2d.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/stereo.hpp>
#include "opencv2/xfeatures2d.hpp"

using namespace cv;
using namespace cv::xfeatures2d;

using std::cout;
using std::endl;

class ImagePairMatcher
{
public:
    ImagePairMatcher();

    virtual int match(cv::InputArray l, cv::InputArray r)=0;
    virtual int transform();
    std::vector<KeyPoint> getPoints(std::string which = "R") { return which == "R" ? m_keypointsR : m_keypointsL; }
    std::vector<DMatch> getGoodMatches() { return m_good_matches; }

protected:
    std::vector<KeyPoint> m_keypointsL, m_keypointsR;
    std::vector<DMatch> m_good_matches;
};

class SURFImagePairMatcher : public ImagePairMatcher
{
public:
    SURFImagePairMatcher();

    virtual int match(cv::InputArray l, cv::InputArray r) override;
};

#endif // IMAGEPAIRMATCHER_H
