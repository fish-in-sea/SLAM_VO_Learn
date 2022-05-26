#include<iostream>
#include<orb_cv.hpp>
#include<memory>

int main(int argc, char **argv){

    LOG(INFO) << "ORB_CV"<<std::endl;
    cv::Mat image1, image2;

   
    std::unique_ptr<Feature_information<cv::KeyPoint>>orbcv(new ORB_CV<cv::KeyPoint>());
    std::unique_ptr<Feature_information<cv::line_descriptor::KeyLine>>linecv(new Line_CV<cv::line_descriptor::KeyLine>());
    std::unique_ptr<Feature_information<cv::Point2f>>lkcv(new LK_CV<cv::Point2f>());
    std::vector<cv::KeyPoint> keypoints1orb, keypoints2orb;
    std::vector<cv::line_descriptor::KeyLine> keypoints1line, keypoints2line;
    std::vector<cv::Point2f> keypoints1lk, keypoints2lk;
    orbcv->Run(image1,image2,keypoints1orb,keypoints2orb);
    linecv->Run(image1,image2,keypoints1line,keypoints2line);
    lkcv->Run(image1,image2,keypoints1lk,keypoints2lk);
    cv::waitKey(0);

    return 0;
}
