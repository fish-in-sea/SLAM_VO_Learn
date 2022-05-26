#pragma once

#include <common.h>
#include <orb_cv.hpp>

enum CameraType{
    Pinhole
};


class Camera{
    public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    // Camera();

    virtual void getParameters()=0;
    virtual void liftproject(const std::vector<cv::Point2f>&Inkeypoints, 
                                std::vector<cv::Point2f>&Outkeypoints, cv::Mat& K, cv::Mat& DistCoef)=0;
    virtual void undistortPoints(const std::vector<cv::Point2f>&Inkeypoints, 
                                std::vector<cv::Point2f>&Outkeypoints, cv::Mat& K, cv::Mat& DistCoef)=0;
    // virtual void distortion(const Eigen::Vector2d& p_u, Eigen::Vector2d& d_u,cv::Mat& K, cv::Mat& DistCoef)=0;
    cv::Mat mK;
    cv::Mat mDistCoef;
};


// 去畸变
class PinholeCameras:public Camera {
    public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    void getParameters(){
       this->mK = (cv::Mat_<float>(3, 3) << 4.616e+02, 0, 3.630e+02, 0, 4.603e+02, 2.481e+02, 0, 0, 1 );
       this->mDistCoef = (cv::Mat_<float>(4, 1)<<-2.917e-01, 8.228e-02, 5.333e-05, -1.578e-04);
    }
    void liftproject(const std::vector<cv::Point2f>&Inkeypoints,std::vector<cv::Point2f>&Outkeypoints
                    ,cv::Mat &K,cv::Mat &DistCoef){
        int N=Inkeypoints.size();
        cv::Mat mat(N,2,CV_32F);
        for(int i=0; i<N; i++)
        {
            mat.at<float>(i,0)=Inkeypoints[i].x;
            mat.at<float>(i,1)=Inkeypoints[i].y;
        }
        // Undistort points
        mat=mat.reshape(2);
        cv::Mat mk=K.clone();
        cv::Mat md=DistCoef.clone();
        cv::undistortPoints(mat,mat,mk,md,cv::Mat(),mk);
        mat=mat.reshape(1);

        // Fill undistorted keypoint vector
        Outkeypoints.clear();
        Outkeypoints.resize(N);
        for(int i=0; i<N; i++)
        {
            cv::Point2f kp = Inkeypoints[i];
            kp.x=mat.at<float>(i,0);
            kp.y=mat.at<float>(i,1);
            Outkeypoints[i]=kp;
        }
      
    }
    void undistortPoints(const std::vector<cv::Point2f>&Inkeypoints, 
                                std::vector<cv::Point2f>&Outkeypoints, cv::Mat& K, cv::Mat& DistCoef){
        
        for(int i=0; i<Inkeypoints.size(); ++i){

        Eigen::Vector3d tmp_p;
            // 得到相机归一化坐标系的值
        cv::Mat mk=K.clone();
        cv::Mat md=DistCoef.clone();
        Outkeypoints.resize(Inkeypoints.size());
        liftProjective(Eigen::Vector2d(Inkeypoints[i].x, Inkeypoints[i].y), tmp_p,mk,md);
        tmp_p.x() = 460 * tmp_p.x() / tmp_p.z() + 752 / 2.0;
        tmp_p.y() = 460 * tmp_p.y() / tmp_p.z() + 480 / 2.0;
        Outkeypoints[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
        }
        
            

    }
    void liftProjective(const Eigen::Vector2d& p, Eigen::Vector3d& P,cv::Mat& K,cv::Mat& D) {

        double mx_d, my_d, mx_u, my_u;
        // double rho2_d, rho4_d, radDist_d, Dx_d, Dy_d, inv_denom_d;
        //double lambda;

        // Lift points to normalised plane
        // 投影到归一化相机坐标系
        double m_inv_K11=1/K.at<double>(0,0);
        double m_inv_K13=1/K.at<double>(0,2);

        double m_inv_K22=1/K.at<double>(1,1);
        double m_inv_K23=1/K.at<double>(1,2);

        mx_d = m_inv_K11 * p(0) + m_inv_K13;
        my_d = m_inv_K22 * p(1) + m_inv_K23;


        // Recursive distortion model

        int n = 8;
        Eigen::Vector2d d_u;
            // 这里mx_d + du = 畸变后
        cv::Mat d=D.clone();
        cv::Mat k=K.clone();
        Eigen::Vector2d tmp=Eigen::Vector2d(mx_d, my_d);
        this->distortion(tmp, d_u,k,d);
        // // Approximate value
        mx_u = mx_d - d_u(0);
        my_u = my_d - d_u(1);

        for (int i = 1; i < n; ++i)
        {
                this->distortion(Eigen::Vector2d(mx_u, my_u), d_u,k,d);
                mx_u = mx_d - d_u(0);
                my_u = my_d - d_u(1);
        }
        P<<mx_u, my_u, 1.0;

    }
    void distortion(const Eigen::Vector2d& p_u, Eigen::Vector2d& d_u,cv::Mat& K, cv::Mat& DistCoef){
        double k1 = DistCoef.at<double>(0);
        double k2 = DistCoef.at<double>(1);
        double p1 = DistCoef.at<double>(2);
        double p2 = DistCoef.at<double>(3);

        double mx2_u, my2_u, mxy_u, rho2_u, rad_dist_u;
        mx2_u = p_u(0) * p_u(0);
        my2_u = p_u(1) * p_u(1);
        mxy_u = p_u(0) * p_u(1);
        rho2_u = mx2_u + my2_u;
        rad_dist_u = k1 * rho2_u + k2 * rho2_u * rho2_u;
        d_u << p_u(0) * rad_dist_u + 2.0 * p1 * mxy_u + p2 * (rho2_u + 2.0 * mx2_u),
            p_u(1) * rad_dist_u + 2.0 * p2 * mxy_u + p1 * (rho2_u + 2.0 * my2_u);
    }
};

class CameraFactory{
    public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    CameraFactory(){};
    virtual ~CameraFactory(){};

    Camera *Create(){
         return new PinholeCameras;
    }

};