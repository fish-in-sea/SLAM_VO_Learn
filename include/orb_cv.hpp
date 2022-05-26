#pragma once
#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <glog/logging.h>
#include <algorithm>
#include <opencv2/line_descriptor/descriptor.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/tracking.hpp>

#include <Eigen/Core>
#include <Eigen/Eigen>

using namespace std;
struct projection_parameters{
    double fx = 4.616e+02;
    double fy = 4.603e+02;
    double cx = 3.630e+02;
    double cy = 2.481e+02; 
};

struct distortion_parameters{
    double k1 = -2.917e-01;
    double k2 = 8.228e-02;
    double p1 = 5.333e-05;
    double p2 = -1.578e-04;
};


template <typename T>
class Feature_information{
    public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Feature_information(){
        image1 = cv::imread("/home/fish/Code/SLAM/picture/1.png");
        image2 = cv::imread("/home/fish/Code/SLAM/picture/2.png");
    };
    virtual ~Feature_information(){};
    
    virtual void Extract(const cv::Mat &image,std::vector<T> &keyPoint,cv::Mat &desriptors)=0;
    virtual void Match(const cv::Mat &desriptors1,const cv::Mat &desriptors2,std::vector<cv::DMatch>&matchesORB,std::vector<cv::DMatch> &goodMatches)=0;
    virtual void DrawMatch(const cv::Mat image1_,const cv::Mat image2_,std::vector<T> &keyPoint1_,std::vector<T> &keyPoint2_,
    std::vector<cv::DMatch>&matchesORB)=0;
    virtual void Run(cv::Mat &image1_,cv::Mat &image2_,std::vector<T> &keyPoint1_,std::vector<T> &keyPoint2_)=0;


    public:
    cv::Mat image1;
    cv::Mat image2;
    std::vector<T> keyPoint1,keyPoint2;
    cv::Mat desriptors1,desriptors2;
    std::vector<cv::DMatch>matchesORB;
    std::vector<cv::DMatch>goodmatchesORB;


    Eigen::Matrix3d mK;
    Eigen::Vector4d mDistCoef;

};

template <typename T>
class ORB_CV:public Feature_information<T>{
    public:
    ORB_CV(){
        // image1 = cv::imread("/home/fish/Code/SLAM14/picture/1403636579763555584.png");
        // image2 = cv::imread("/home/fish/Code/SLAM14/picture/1403636579813555456.png");
    };
    virtual ~ORB_CV(){};


    void Extract(const cv::Mat &image,std::vector<T> &keyPoint,cv::Mat &desriptors){
        if(image.empty()) return;
        cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
        cv::Ptr<cv::DescriptorExtractor> desriptor = cv::ORB::create();
        detector->detect(image,keyPoint);
        desriptor->compute(image,keyPoint,desriptors);
        LOG(INFO)<<"keyPoint SIZE:"<<keyPoint.size()<<std::endl;
        LOG(INFO)<<"desriptors SIZE:"<<desriptors.size()<<std::endl;

    }
    void Match(const cv::Mat &desriptors1,const cv::Mat &desriptors2,
        std::vector<cv::DMatch>&matchesORB,std::vector<cv::DMatch> &goodMatches){
        cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
        matcher->match(desriptors1,desriptors2,matchesORB);

        sort(matchesORB.begin(),matchesORB.end(),[](cv::DMatch &a,cv::DMatch &b){
            return a.distance<b.distance;
        });

        if(matchesORB.size()>150) matchesORB.resize(150);
        // else matchesORB.resize(150);

        double min=(matchesORB.begin()->distance);
        double max=((matchesORB.end()-1)->distance);
        
        LOG(INFO)<<"min and max distance"<<min<<" "<<max<<std::endl;
        for(int i=0;i<matchesORB.size();++i){
            if(matchesORB[i].distance<std::max(2*min,10.0));
                goodMatches.push_back(matchesORB[i]);
        }

    }
    void DrawMatch(const cv::Mat image1_,const cv::Mat image2_,std::vector<T> &keyPoint1_,std::vector<T> &keyPoint2_,
    std::vector<cv::DMatch>&matchesORB){
        cv::Mat imageMatch;
        cv::drawMatches(image1_,keyPoint1_,image2_,keyPoint2_,matchesORB,imageMatch);
        cv::imshow("ORBmatch",imageMatch);
        // cv::waitKey(0);
    }
    void Run(cv::Mat &image1_,cv::Mat &image2_,std::vector<T> &keyPoint1_,std::vector<T> &keyPoint2_){
        // 提取
        Extract(this->image1,this->keyPoint1,this->desriptors1);
        Extract(this->image2,this->keyPoint2,this->desriptors2);
        // 匹配
        Match(this->desriptors1,this->desriptors2,this->matchesORB,this->goodmatchesORB);
        // 绘制结果
        DrawMatch(this->image1,this->image2,this->keyPoint1,this->keyPoint2,this->goodmatchesORB);
    }

};


template <typename T>
class Line_CV:public Feature_information<T>{
    public:
    Line_CV(
    ){};
    virtual ~Line_CV(){};

    void Extract(const cv::Mat &image,std::vector<T> &keyPoint,cv::Mat &desriptors){

        cv::Ptr<cv::line_descriptor::LSDDetector> lsd = cv::line_descriptor::LSDDetector::createLSDDetector();
        cv::Ptr<cv::line_descriptor::BinaryDescriptor> lbd = cv::line_descriptor::BinaryDescriptor::createBinaryDescriptor(); 
        lsd->detect(image, keyPoint, 1.2,1);
        LOG(INFO)<<"keyPoint SIZE:"<<keyPoint.size()<<std::endl;
        if(keyPoint.size()>50)
        {
            sort(keyPoint.begin(), keyPoint.end(), [](const T& a, const T& b){
                return ( a.response > b.response );
            });
            keyPoint.resize(50);
            for( int i=0; i<50; i++)
                keyPoint[i].class_id = i;
        }
        lbd->compute(image, keyPoint, desriptors);
        LOG(INFO)<<"desriptors SIZE:"<<desriptors.size()<<std::endl;
    }
    void Match(const cv::Mat &desriptors1,const cv::Mat &desriptors2,
            std::vector<cv::DMatch>&matchesORB,std::vector<cv::DMatch> &goodMatches){
        cv::BFMatcher* bfm = new cv::BFMatcher(cv::NORM_HAMMING, false);

        std::vector<std::vector<cv::DMatch > > lmatches;
        bfm->knnMatch(desriptors1, desriptors2, lmatches, 2);
        for(size_t i=0;i<lmatches.size();i++)
        {
            const cv::DMatch& bestMatch = lmatches[i][0];
            const cv::DMatch& betterMatch = lmatches[i][1];
            float  distanceRatio = bestMatch.distance / betterMatch.distance;
            if (distanceRatio < 0.75)
                goodMatches.push_back(bestMatch);
            
        }
        LOG(INFO)<<"goodMatches SIZE:"<<goodMatches.size()<<std::endl;

    }
    void DrawMatch(const cv::Mat image1_,const cv::Mat image2_,std::vector<T> &keyPoint1_,std::vector<T> &keyPoint2_,
    std::vector<cv::DMatch>&matchesORB){
    cv::Mat imageMatch;
    // std::vector<int >mask(matchesORB.size(),1);
    cv::line_descriptor::drawLineMatches( image1_, keyPoint1_, image2_, keyPoint2_, matchesORB, imageMatch,
                        cv::Scalar::all( -1 ), cv::Scalar::all( -1 ), std::vector<char >(matchesORB.size(),1),
                     cv::line_descriptor::DrawLinesMatchesFlags::DEFAULT);
    cv::imshow("linematch",imageMatch);

    }
    void Run(cv::Mat &image1_,cv::Mat &image2_,std::vector<T> &keyPoint1_,std::vector<T> &keyPoint2_){
        // 提取
        Extract(this->image1,this->keyPoint1,this->desriptors1);
        Extract(this->image2,this->keyPoint2,this->desriptors2);
        // 匹配
        Match(this->desriptors1,this->desriptors2,this->matchesORB,this->goodmatchesORB);
        // 绘制结果
        DrawMatch(this->image1,this->image2,this->keyPoint1,this->keyPoint2,this->goodmatchesORB);
    }


};



template <typename T>
class LK_CV:public Feature_information<T>{
    public:
    LK_CV(){};
    virtual ~LK_CV(){};

    void Extract(const cv::Mat &image,std::vector<T> &keyPoint,cv::Mat &desriptors){
        std::vector<cv::Point2f>pts;
        if(image.empty()) return;
        
        cv::Mat image_gray;
        cv::cvtColor(image, image_gray, cv::COLOR_BGR2GRAY);
        // cv::imshow("image",image_gray);
        cv::goodFeaturesToTrack(image_gray,pts,500,0.01,10,cv::Mat());
        for (auto &p : pts){
            keyPoint.push_back(p);
        }
        firstImage=false;
    };
    void Match(const cv::Mat &desriptors1,const cv::Mat &desriptors2,std::vector<cv::DMatch>&matchesORB,
                std::vector<cv::DMatch> &goodMatches){
        std::vector<uchar> status;
        std::vector<float> err;
        cv::Mat image_gray1,image_gray2;
        // cv::cvtColor(this->image1, image_gray1, cv::COLOR_BGR2GRAY);
        // cv::cvtColor(this->image2, image_gray2, cv::COLOR_BGR2GRAY);

        // cv::calcOpticalFlowPyrLK(image_gray1, image_gray2,this->keyPoint1, this->keyPoint2, status, err, cv::Size(21, 21), 3);

        cv::calcOpticalFlowPyrLK(this->image1, this->image2,this->keyPoint1, this->keyPoint2, status, err, cv::Size(21, 21), 3);
        reduceVector(this->keyPoint2,status);
        reduceVector(this->keyPoint1,status);

    };
    void reduceVector(std::vector<cv::Point2f> &v, std::vector<uchar> status){
        int j=0;
        for(int i = 0;i < int(v.size());++i){
            if(status[i]) v[j++]=v[i];

        }
        v.resize(j);
    }
    void DrawMatch(const cv::Mat image1_,const cv::Mat image2_,std::vector<T> &keyPoint1_,std::vector<T> &keyPoint2_,
    std::vector<cv::DMatch>&matchesORB){
        for ( auto kp:keyPoint2_ )
                cv::circle(image1_, kp, 10, cv::Scalar(0, 240, 0), 1);
            cv::imshow("corners", image1_);
    }
    void Run(cv::Mat &image1_,cv::Mat &image2_,std::vector<T> &keyPoint1_,std::vector<T> &keyPoint2_){
        // 提取
        
        if(firstImage){
            Extract(this->image1,this->keyPoint1,this->desriptors1);
        }
        // 匹配
        Match(this->desriptors1,this->desriptors2,this->matchesORB,this->goodmatchesORB);
        // 绘制结果
        DrawMatch(this->image1,this->image2,this->keyPoint1,this->keyPoint2,this->goodmatchesORB);

        // update
        image1_=this->image1.clone();
        image2_=this->image2.clone();
        for(auto i:this->keyPoint1) keyPoint1_.push_back(i);
        for(auto i:this->keyPoint2) keyPoint2_.push_back(i);

    }
    private:
    bool firstImage=true;
};