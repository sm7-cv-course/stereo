#include "fdmsolver.h"

using namespace fdm;

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

i_FDMSolver::i_FDMSolver()
{

}

FDMExplicit::FDMExplicit(FDMSettings st)
{
    m_settings = st;
    m_A = cv::Mat::ones(st.frameSize, CV_32F);
    m_A *= st.initialValue;
    m_A1 = m_A;
    m_Ct = cv::Mat::ones(st.frameSize, CV_32F);
    m_Ct *= st.Ct0;
}

void
FDMExplicit::addBoundaryPoints(const std::vector<cv::Point2i> &points, _REAL val, _REAL Ct)
{
    for (size_t i = 0; i < points.size(); ++i) {
        FDMNode node;
        node.x = points[i].x;
        node.y = points[i].y;
        node.val = val;
        node.C = Ct;
        m_boundaries.push_back(node);
        m_Ct.at<_REAL>(points[i].y, points[i].x) = Ct;
        m_A.at<_REAL>(points[i].y, points[i].x) = val;
    }

}

cv::Mat
FDMExplicit::doOneIteration()
{
    for (int i = 1; i < m_A.rows - 1; ++i) {
        for (int j = 1; j < m_A.cols - 1; ++j) {
            m_A1.at<_REAL>(i, j) =
                m_A.at<_REAL>(i, j) +
                m_Ct.at<_REAL>(i, j) *
                (m_A.at<_REAL>(i, j-1) + m_A.at<_REAL>(i-1, j) + m_A.at<_REAL>(i, j+1) + m_A.at<_REAL>(i+1, j) - 4 * m_A.at<_REAL>(i, j)) / (m_settings.H * m_settings.H);
        }
    }
    m_A = m_A1;

    return m_A1;
}

cv::Mat
FDMExplicit::doNIterations(int IterNum)
{
    for (int n = 0; n < IterNum; ++n) {
        for (int i = 1; i < m_A.rows - 1; ++i) {
            for (int j = 1; j < m_A.cols - 1; ++j) {
                m_A1.at<_REAL>(i, j) =
                    m_A.at<_REAL>(i, j) +
                    m_Ct.at<_REAL>(i, j) *
                    (m_A.at<_REAL>(i, j-1) + m_A.at<_REAL>(i-1, j) + m_A.at<_REAL>(i, j+1) + m_A.at<_REAL>(i+1, j) - 4 * m_A.at<_REAL>(i, j)) / (m_settings.H * m_settings.H);
            }
        }
        m_A = m_A1;
    }

    return m_A1;
}
