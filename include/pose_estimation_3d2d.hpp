#pragma once

#include <orb_cv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "Thirdparty/g2o/g2o/core/block_solver.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_eigen.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include "Thirdparty/g2o/g2o/core/robust_kernel_impl.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_dense.h"
#include "Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"
#include "ceres/ceres.h"
#include "ceres/rotation.h"

using namespace cv;
class cost_function_define {
    public:
    cost_function_define(Point3f p1,Point2f p2):_p1(p1),_p2(p2){}
    virtual ~cost_function_define() {}
    template<typename T>
    bool operator()(const T* const cere_r,const T* const cere_t,T* residual)const
    {
        // 输入的3维点
        T p_1[3];
        // 预测3维点
        T p_2[3];
        // 类型转换
        p_1[0]=T(_p1.x);
        p_1[1]=T(_p1.y);
        p_1[2]=T(_p1.z);
        // RP
        ceres::AngleAxisRotatePoint(cere_r,p_1,p_2);
        // RP+t
        p_2[0]=p_2[0]+cere_t[0];
        p_2[1]=p_2[1]+cere_t[1];
        p_2[2]=p_2[2]+cere_t[2];
        // 归一
        const T x=p_2[0]/p_2[2];
        const T y=p_2[1]/p_2[2];

        //三维点重投影计算的像素坐标
        const T u=x*520.9+325.1;
        const T v=y*521.0+249.7;
        //观测的在图像坐标下的值
        T u1=T(_p2.x);
        T v1=T(_p2.y);
        // 误差函数
        residual[0]=u-u1;
        residual[1]=v-v1;
        return true;
    }
    private:
    Point3f _p1;
    Point2f _p2;
};

// void find_feature_matches (
//     const Mat& img_1, const Mat& img_2,
//     std::vector<KeyPoint>& keypoints_1,
//     std::vector<KeyPoint>& keypoints_2,
//     std::vector< DMatch >& matches );

// // 像素坐标转相机归一化坐标
// Point2d pixel2cam ( const Point2d& p, const Mat& K );

// void bundleAdjustment (
//     const vector<Point3f> points_3d,
//     const vector<Point2f> points_2d,
//     const Mat& K,
//     Mat& R, Mat& t
// );
// void bundleAdjustment_by_ceres (
//     const vector<Point3f> points_3d,
//     const vector<Point2f> points_2d,
//     const Mat& K,
//     Mat& R, Mat& t
// );

void find_feature_matches ( const Mat& img_1, const Mat& img_2,
                            std::vector<KeyPoint>& keypoints_1,
                            std::vector<KeyPoint>& keypoints_2,
                            std::vector< DMatch >& matches )
{
    //-- 初始化
    Mat descriptors_1, descriptors_2;
    // used in OpenCV3
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    // use this if you are in OpenCV2
    // Ptr<FeatureDetector> detector = FeatureDetector::create ( "ORB" );
    // Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create ( "ORB" );
    Ptr<DescriptorMatcher> matcher  = DescriptorMatcher::create ( "BruteForce-Hamming" );
    //-- 第一步:检测 Oriented FAST 角点位置
    detector->detect ( img_1,keypoints_1 );
    detector->detect ( img_2,keypoints_2 );

    //-- 第二步:根据角点位置计算 BRIEF 描述子
    descriptor->compute ( img_1, keypoints_1, descriptors_1 );
    descriptor->compute ( img_2, keypoints_2, descriptors_2 );

    //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
    vector<DMatch> match;
    // BFMatcher matcher ( NORM_HAMMING );
    matcher->match ( descriptors_1, descriptors_2, match );

    //-- 第四步:匹配点对筛选
    double min_dist=10000, max_dist=0;

    //找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        double dist = match[i].distance;
        if ( dist < min_dist ) min_dist = dist;
        if ( dist > max_dist ) max_dist = dist;
    }

    printf ( "-- Max dist : %f \n", max_dist );
    printf ( "-- Min dist : %f \n", min_dist );

    //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        if ( match[i].distance <= max ( 2*min_dist, 30.0 ) )
        {
            matches.push_back ( match[i] );
        }
    }
}

Point2d pixel2cam ( const Point2d& p, const Mat& K )
{
    return Point2d
           (
               ( p.x - K.at<double> ( 0,2 ) ) / K.at<double> ( 0,0 ),
               ( p.y - K.at<double> ( 1,2 ) ) / K.at<double> ( 1,1 )
           );
}

void bundleAdjustment_by_g2o (
    const vector< Point3f > points_3d,
    const vector< Point2f > points_2d,
    const Mat& K,
    Mat& R, Mat& t )
{

    // 初始化G2O
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);
    g2o::VertexSE3Expmap * pose = new g2o::VertexSE3Expmap();
    cv::Mat T;    

    Eigen::Matrix3d R_mat;
    R_mat <<
          R.at<double> ( 0,0 ), R.at<double> ( 0,1 ), R.at<double> ( 0,2 ),
               R.at<double> ( 1,0 ), R.at<double> ( 1,1 ), R.at<double> ( 1,2 ),
               R.at<double> ( 2,0 ), R.at<double> ( 2,1 ), R.at<double> ( 2,2 );
    T=( Mat_<double> ( 3,4 ) <<  R_mat(0,0), R_mat(0,1), R_mat(0,2), t.at<double>(0,0),
                                 R_mat(1,0), R_mat(1,1), R_mat(1,2), t.at<double>(1,0),
                                 R_mat(2,0), R_mat(2,1), R_mat(2,2), t.at<double>(2,0));
    
    // T<<
    pose->setEstimate(g2o::SE3Quat (
                            R_mat,
                            Eigen::Vector3d ( t.at<double> ( 0,0 ), t.at<double> ( 1,0 ), t.at<double> ( 2,0 ) )
                        ));
    pose->setId(0);
    pose->setFixed(false);
    optimizer.addVertex(pose);

    // 顶点
    int index = 1;
    for ( const Point3f p:points_3d )  {
        g2o::VertexSBAPointXYZ *point =new g2o::VertexSBAPointXYZ();
        point->setId ( index++ );
        point->setEstimate ( Eigen::Vector3d ( p.x, p.y, p.z ) );
        point->setMarginalized ( true ); // g2o 中必须设置 marg 参见第十讲内容
        optimizer.addVertex ( point );
    }
     // edges
    index = 1;
    for ( const Point2f p:points_2d )
    {
        g2o::EdgeSE3ProjectXYZ* edge = new g2o::EdgeSE3ProjectXYZ();
        edge->setId ( index );
        edge->setVertex ( 0, dynamic_cast<g2o::VertexSBAPointXYZ*> ( optimizer.vertex ( index ) ) );
        edge->setVertex ( 1, pose );
        edge->setMeasurement ( Eigen::Vector2d ( p.x, p.y ) );
        edge->setParameterId ( 0,0 );
        edge->setInformation ( Eigen::Matrix2d::Identity() );
        edge->fx = K.at<double> ( 0,0);
        edge->cx = K.at<double> ( 0,2);
        edge->fy = K.at<double> ( 1,1);
        edge->cy = K.at<double> ( 1,2);

        optimizer.addEdge ( edge );
        index++;
    }
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    optimizer.setVerbose ( true );
    optimizer.initializeOptimization(0);
    optimizer.optimize ( 2 );
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>> ( t2-t1 );
    cout<<"optimization costs time: "<<time_used.count() <<" seconds."<<endl;

    cout<<endl<<"after optimization:"<<endl;

    g2o::VertexSE3Expmap* vSE3_recov = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
    g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
    Eigen::Matrix<double,4,4>  pose_recov = SE3quat_recov.to_homogeneous_matrix();
    cout<<"T="<<endl<<pose_recov <<endl;

   
}
void bundleAdjustment_by_ceres (
    const vector< Point3f > points_3d,
    const vector< Point2f > points_2d,
    const Mat& K,
    Mat& R, Mat& t )
{
    ceres::Problem problem;



    cv::Mat R_in;
    cv::Rodrigues(R,R_in);
    double cere_rot[3],cere_tranf[3];

    cere_rot[0]=R_in.at<double>(0,0);
    cere_rot[1]=R_in.at<double>(1,0);
    cere_rot[2]=R_in.at<double>(2,0);

    cere_tranf[0]=t.at<double>(0,0);
    cere_tranf[1]=t.at<double>(1,0);
    cere_tranf[2]=t.at<double>(2,0);

    for(int i =0; i < points_3d.size(); i++){
        ceres::CostFunction *costfunction=new ceres::AutoDiffCostFunction<cost_function_define,2,3,3>(new cost_function_define(points_3d[i],points_2d[i]));

        ceres::LossFunction *loss_function;
    //loss_function = new ceres::HuberLoss(1.0);
        loss_function = new ceres::CauchyLoss(1.0);
        problem.AddResidualBlock(costfunction,loss_function,cere_rot,cere_tranf);
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::LinearSolverType::SPARSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    ceres::Solve(options,&problem,&summary);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>> ( t2-t1 );
    cout<<"optimization costs time: "<<time_used.count() <<" seconds."<<endl;
    cout<<summary.BriefReport()<<endl;

    
    cout<<"----------------optional after--------------------"<<endl;
    Mat cam_3d = ( Mat_<double> ( 3,1 )<<cere_rot[0],cere_rot[1],cere_rot[2]);
    Mat cam_9d;
    cv::Rodrigues ( cam_3d, cam_9d ); // r为旋转向量形式，用Rodrigues公式转换为矩阵

    cout<<"cam_9d:"<<endl<<cam_9d<<endl;

    cout<<"cam_t:"<<cere_tranf[0]<<"  "<<cere_tranf[1]<<"  "<<cere_tranf[2]<<endl;


}