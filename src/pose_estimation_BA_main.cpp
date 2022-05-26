
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <chrono>
#include <orb_cv.hpp>
#include <pose_estimation_3d2d.hpp>
#include <CameraFactory.h>
using namespace std;
using namespace cv;




int main ( int argc, char** argv )
{

    //-- 读取图像
    Mat K = ( Mat_<double> ( 3,3 ) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1 );
    cv::Mat image1, image2;

    std::vector<cv::Point2f> keypoints1, keypoints2;
    std::unique_ptr<Feature_information<cv::Point2f>>lkcv(new LK_CV<cv::Point2f>());
    lkcv->Run(image1,image2,keypoints1,keypoints2);
    std::vector<cv::Point2f> unkeypoints1, unkeypoints2;

    CameraFactory *Factory=new CameraFactory();
    Camera *Camera = (Factory->Create());

    Camera->mK =  ( Mat_<double> ( 3,3 ) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1 );
    Camera->mDistCoef = ( Mat_<double> ( 4,1 ) << 0, 0, 0, 0 );
    // cv::Mat K=Camera->mK;
    cv::Mat D=Camera->mDistCoef;
    Camera->liftproject(keypoints1,unkeypoints1,K,D);
    Camera->liftproject(keypoints2,unkeypoints2,K,D);
    cout<<"一共找到了"<<unkeypoints1.size() <<"组匹配点"<<endl;

    // 建立3D点
    Mat d1 = imread ( "/home/fish/Code/SLAM/picture/1_depth.png", CV_LOAD_IMAGE_UNCHANGED );       // 深度图为16位无符号数，单通道图像
    
    vector<Point3f> pts_3d;
    vector<Point2f> pts_2d;
    for ( int i=0;i<unkeypoints1.size();++i)
    {
        ushort d = d1.ptr<unsigned short> (int ( unkeypoints1[i].y )) [ int ( unkeypoints1[i].x ) ];
        if ( d == 0 )   // bad depth
            continue;
        float dd = d/5000.0;
        Point2d p1 = pixel2cam ( unkeypoints1[i], K );
        pts_3d.push_back ( Point3f ( p1.x*dd, p1.y*dd, dd ) );
        pts_2d.push_back ( unkeypoints1[i] );
    }

    cout<<"3d-2d pairs: "<<pts_3d.size() <<endl;

    Mat r, t;
    solvePnP ( pts_3d, pts_2d, K, Mat(), r, t, false ); // 调用OpenCV 的 PnP 求解，可选择EPNP，DLS等方法
    Mat R;
    cv::Rodrigues ( r, R ); // r为旋转向量形式，用Rodrigues公式转换为矩阵

    cout<<"R="<<endl<<R<<endl;
    cout<<"t="<<endl<<t<<endl;

    cout<<"calling bundle adjustment"<<endl;
    cv:: Mat R_g2o=R.clone();
    cv:: Mat R_ceres=R.clone();

    cv:: Mat t_g2o=t.clone();
    cv:: Mat t_ceres=t.clone();

    bundleAdjustment_by_ceres ( pts_3d, pts_2d, K, R_ceres, t_ceres);
    bundleAdjustment_by_g2o ( pts_3d, pts_2d, K, R_g2o, t_g2o );
    cv::waitKey(0);
}
