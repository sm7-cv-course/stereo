#ifndef FDMSOLVER_H
#define FDMSOLVER_H

#include <opencv/cv.h>
#include <opencv2/core.hpp>
#include <vector>

#define _REAL float

namespace fdm {

/**
 * @brief The FDMSettings struct
 *
 * Settings of any Finite Differences methods
 */
struct FDMSettings {
    //! Size of plate
    cv::Size frameSize;
    //! Initial "temperature" of each node
    _REAL initialValue = 0.0;
    //! Default thermal conductivity coefficient for each node
    _REAL Ct0 = 1.0;
    _REAL H = 10;
    FDMSettings(cv::Size s = cv::Size(0,0), _REAL IV = 0.0, _REAL Ct = 1.0, _REAL h = 10)
        :frameSize(s), initialValue(IV), Ct0(Ct), H(h) {}
};

/**
 * @brief The FDMBoundaryPoint struct
 * Node with coordinate and thermal conductivity coefficient
 */
struct FDMNode {
    //! Node(Pixel) coordinates
    int x;
    int y;
    //! Initial or current value
    _REAL val;
    //! Thermal conductivity coefficient
    _REAL C;
};

/**
 * @brief The FDMSolver class
 * Interface for Finite Difference solver of "Heat spread task" for 2D rectangular plate.
 *
 *  dU/dt - Ct * (d2U/dx + d2U/dy) = 0
 *
 *  Ct - "thermal conductivity coefficient"
 */
class i_FDMSolver
{
public:
    i_FDMSolver();

    //! Points with constant Ct
    virtual void addBoundaryPoints(const std::vector<cv::Point2i> &points, _REAL val, _REAL Ct)=0;
    virtual cv::Mat doOneIteration()=0;
    virtual cv::Mat doNIterations(int IterNum)=0;

    void setSettings(const FDMSettings &s) { m_settings = s; }
    FDMSettings getSettings() const { return m_settings; }

protected:
    FDMSettings m_settings;
};

/**
 * @brief The FDMExplicit class
 * Explicit variant of Finite Differences, see @FDMSolver
 */
class FDMExplicit : public i_FDMSolver
{
public:
    FDMExplicit(FDMSettings st);

    // Points with constant Ct
    void addBoundaryPoints(const std::vector<cv::Point2i> &points, _REAL val, _REAL Ct) override;
    cv::Mat doOneIteration() override;
    cv::Mat doNIterations(int IterNum) override;

private:
    // Boundaries or any nodes with non default Ct and initial value
    std::vector<FDMNode> m_boundaries;
    cv::Mat_<_REAL> m_A;
    cv::Mat_<_REAL> m_A1;
    cv::Mat_<_REAL> m_Ct;
};

};

#endif // FDMSOLVER_H
