#include <iostream>
#include <orb_cv.hpp>
#include <memory>
#include <CameraFactory.h>
#include <features_pose.h>

using namespace std;
using namespace cv;


int main(int argc, char **argv){

    LOG(INFO) << "ORB_CV"<<std::endl;
    cv::Mat image1, image2;

    std::vector<cv::Point2f> keypoints1, keypoints2;

    std::unique_ptr<Feature_information<cv::Point2f>> lkcv(new LK_CV<cv::Point2f>());
    lkcv->Run(image1,image2,keypoints1,keypoints2);

    std::vector<cv::Point2f> unkeypoints1, unkeypoints2;

    CameraFactory *Factory=new CameraFactory();
    Camera *Camera = (Factory->Create());

    Camera->mK =  ( Mat_<double> ( 3,3 ) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1 );
    Camera->mDistCoef = ( Mat_<double> ( 4,1 ) << 0, 0, 0, 0 );
    cv::Mat K=Camera->mK;
    cv::Mat D=Camera->mDistCoef;
    Camera->liftproject(keypoints1,unkeypoints1,K,D);
    Camera->liftproject(keypoints2,unkeypoints2,K,D);


    cv::Mat F;
    F=cv::findFundamentalMat(unkeypoints1,unkeypoints2);

    cout<<"F:"<<endl<<F<<endl;
    cv::Mat E12=K.t()*F*K;

    cout<<"essential_matrix E12 is "<<endl<< E12<<endl;
    //-- 计算本质矩阵
    Mat essential_matrix;
    essential_matrix = findEssentialMat ( unkeypoints1, unkeypoints2,K);
    cout<<"essential_matrix is "<<endl<< essential_matrix<<endl;
    cv::Mat tmpK=K.inv();
    cv::Mat F12=tmpK.t()*essential_matrix*tmpK;
    cout<<"essential_matrix F12 is "<<endl<< F12<<endl;

    cv::Mat R1,t1,R,t;
    cv::recoverPose ( essential_matrix, unkeypoints1, unkeypoints2,K, R, t);
    cv::recoverPose ( E12, keypoints1, keypoints2,K, R1, t1);
    cout<<"R is "<<endl<<R<<endl;
    cout<<"t is "<<endl<<t<<endl;
    cout<<"R1 is "<<endl<<R1<<endl;
    cout<<"t1 is "<<endl<<t1<<endl;

    cout<<"RR1 is "<<endl<<R.t()*R1<<endl;
    cout<<"TT1 is "<<endl<<(t-t1)<<endl;
    cv::Mat P1(3,4,CV_32F);
    cv::Mat I(3,4,CV_32F);
    I=( Mat_<double> ( 3,4 ) <<  1, 0, 0, 0,
                                 0, 1, 0, 0,
                                 0, 0, 1, 0);
    P1=K*I;
    cv::Mat P2(3,4,CV_32F);
    cv::Mat T(3,4,CV_32F);
    T=( Mat_<double> ( 3,4 ) <<  R1.at<double>(0,0), R1.at<double>(0,1), R1.at<double>(0,2), t1.at<double>(0,0),
                                 R1.at<double>(1,0), R1.at<double>(1,1), R1.at<double>(1,2), t1.at<double>(0,1),
                                 R1.at<double>(2,0), R1.at<double>(2,1), R1.at<double>(2,2), t1.at<double>(0,2));
    P2=K*T;

    vector<cv::Mat >x3D;
    Triangulate(unkeypoints1,unkeypoints2,P1,P2,x3D);

    cv::waitKey(0);
    

    return 0;
}
