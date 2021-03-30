#include "imagepairmatcher.h"


ImagePairMatcher::ImagePairMatcher()
{

}

int ImagePairMatcher::transform()
{

    return 0;
}

SURFImagePairMatcher::SURFImagePairMatcher()
{

}

int
SURFImagePairMatcher::match(cv::InputArray img1, cv::InputArray img2)
{
    if ( img1.empty() || img2.empty() )
    {
        cout << "Could not open or find the image!\n" << endl;
        return -1;
    }

    //-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
    int minHessian = 400;
    Ptr<SURF> detector = SURF::create( minHessian );
    std::vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;
    detector->detectAndCompute( img1, noArray(), keypoints1, descriptors1 );
    detector->detectAndCompute( img2, noArray(), keypoints2, descriptors2 );

    //-- Step 2: Matching descriptor vectors with a FLANN based matcher
    // Since SURF is a floating-point descriptor NORM_L2 is used
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    std::vector< std::vector<DMatch> > knn_matches;
    matcher->knnMatch( descriptors1, descriptors2, knn_matches, 2 );

    //-- Filter matches using the Lowe's ratio test
    const float ratio_thresh = 0.7f;

    for (size_t i = 0; i < knn_matches.size(); i++)
    {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
        {
            m_good_matches.push_back(knn_matches[i][0]);
            m_keypointsL.push_back(keypoints1[knn_matches[i][0].queryIdx]);
            m_keypointsR.push_back(keypoints2[knn_matches[i][1].trainIdx]);
        }
    }

#ifdef CV__DEBUG_NS_BEGIN
    //-- Draw matches
    Mat img_matches;
    drawMatches( img1, keypoints1, img2, keypoints2, m_good_matches, img_matches, Scalar::all(-1),
                 Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    // Save good matches
    cv::imwrite("good_matches.png", img_matches);
    //-- Show detected matches
    // imshow("Good Matches", img_matches );
    // waitKey();

#endif

    return 0;
}
