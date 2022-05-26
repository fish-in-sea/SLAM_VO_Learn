#include <iostream>
#include <orb_cv.hpp>
#include "features.h"
#include <common.h>

cv::Point2f pixel_to_cam ( cv::Point2f& p, const cv::Mat& K )
{
    return cv::Point2f
    (
        ( p.x - K.at<double>(0,2) ) / K.at<double>(0,0), 
        ( p.y - K.at<double>(1,2) ) / K.at<double>(1,1) 
    );
}

void Normalize( vector<cv::Point2f> &vKeys, vector<cv::Point2f> &vNormalizedPoints, cv::Mat &T) //将特征点归一化的矩阵
{
    // 归一化的是这些点在x方向和在y方向上的一阶绝对矩（随机变量的期望）。

    // Step 1 计算特征点X,Y坐标的均值 meanX, meanY
    float meanX = 0;
    float meanY = 0;

	//获取特征点的数量
    const int N = vKeys.size();

	//设置用来存储归一后特征点的向量大小，和归一化前保持一致
    vNormalizedPoints.resize(N);

	//开始遍历所有的特征点
    for(int i=0; i<N; i++)
    {
		//分别累加特征点的X、Y坐标
        meanX += vKeys[i].x;
        meanY += vKeys[i].y;
    }

    //计算X、Y坐标的均值
    meanX = meanX/N;
    meanY = meanY/N;

    // Step 2 计算特征点X,Y坐标离均值的平均偏离程度 meanDevX, meanDevY，注意不是标准差
    float meanDevX = 0;
    float meanDevY = 0;

    // 将原始特征点减去均值坐标，使x坐标和y坐标均值分别为0
    for(int i=0; i<N; i++)
    {
        vNormalizedPoints[i].x = vKeys[i].x - meanX;
        vNormalizedPoints[i].y = vKeys[i].y - meanY;

		//累计这些特征点偏离横纵坐标均值的程度
        meanDevX += fabs(vNormalizedPoints[i].x);
        meanDevY += fabs(vNormalizedPoints[i].y);
    }

    // 求出平均到每个点上，其坐标偏离横纵坐标均值的程度；将其倒数作为一个尺度缩放因子
    meanDevX = meanDevX/N;
    meanDevY = meanDevY/N;

    float sX = 1.0/meanDevX;
    float sY = 1.0/meanDevY;

    // Step 3 将x坐标和y坐标分别进行尺度归一化，使得x坐标和y坐标的一阶绝对矩分别为1 
    // 这里所谓的一阶绝对矩其实就是随机变量到取值的中心的绝对值的平均值（期望）
    for(int i=0; i<N; i++)
    {
		//对，就是简单地对特征点的坐标进行进一步的缩放
        vNormalizedPoints[i].x = vNormalizedPoints[i].x * sX;
        vNormalizedPoints[i].y = vNormalizedPoints[i].y * sY;
    }

    // Step 4 计算归一化矩阵：其实就是前面做的操作用矩阵变换来表示而已
    // |sX  0  -meanx*sX|
    // |0   sY -meany*sY|
    // |0   0      1    |
    T = cv::Mat::eye(3,3,CV_32F);
    T.at<float>(0,0) = sX;
    T.at<float>(1,1) = sY;
    T.at<float>(0,2) = -meanX*sX;
    T.at<float>(1,2) = -meanY*sY;
}
void DecomposeE(const cv::Mat &E, cv::Mat &R1, cv::Mat &R2, cv::Mat &t)
{

    // 对本质矩阵进行奇异值分解
	//准备存储对本质矩阵进行奇异值分解的结果
    cv::Mat u,w,vt;
	//对本质矩阵进行奇异值分解
    cv::SVD::compute(E,w,u,vt);

    u.col(2).copyTo(t);
    t=t/cv::norm(t);

    // 构造一个绕Z轴旋转pi/2的旋转矩阵W，按照下式组合得到旋转矩阵 R1 = u*W*vt
    //计算完成后要检查一下旋转矩阵行列式的数值，使其满足行列式为1的约束
    cv::Mat W(3,3,CV_32F,cv::Scalar(0));
    W.at<float>(0,1)=-1;
    W.at<float>(1,0)=1;
    W.at<float>(2,2)=1;

	//计算
    R1 = u*W*vt;
	//旋转矩阵有行列式为+1的约束，所以如果算出来为负值，需要取反
    if(cv::determinant(R1)<0)
        R1=-R1;

	// 同理将矩阵W取转置来按照相同的公式计算旋转矩阵R2 = u*W.t()*vt
    R2 = u*W.t()*vt;
	//旋转矩阵有行列式为1的约束
    if(cv::determinant(R2)<0)
        R2=-R2;
}
// void mFindFundamental(vector<cv::Point2f> &mvKeys1, vector<cv::Point2f> &mvKeys2,float &score, cv::Mat &F21){

//     const int N = mvKeys1.size();

//     vector<cv::Point2f> vPn1, vPn2;
//     cv::Mat T1, T2;
//     Normalize(mvKeys1,vPn1, T1);
//     Normalize(mvKeys2,vPn2, T2);
//     cv::Mat T2t = T2.t();  


// }
void Triangulate(vector<cv::Point2f> &mvKeys1, vector<cv::Point2f> &mvKeys2,
                cv::Mat &P1, cv::Mat &P2,vector<cv::Mat> &x3D){


    for(int i = 0; i < mvKeys1.size(); ++i){

        cv::Mat A(4,4,CV_32F);;
        A.row(0)=mvKeys1[i].x*P1.row(2)-P1.row(0);
        A.row(1)=mvKeys1[i].y*P1.row(2)-P2.row(0);
        A.row(2)=mvKeys2[i].x*P1.row(2)-P1.row(0);
        A.row(3)=mvKeys2[i].y*P1.row(2)-P2.row(0);
        //奇异值分解的结果
        cv::Mat u,w,vt;

        cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A|cv::SVD::FULL_UV);
        cv::Mat M3D=vt.row(3).t();
        M3D= M3D/M3D.at<float>(3);
        if(M3D.at<double>(0,2)<0.001||M3D.at<double>(0,2)>20) continue;
        cv::Mat m3d;
        m3d =( cv::Mat_<double> ( 3,1 ) <<M3D.at<double>(0,0), M3D.at<double>(0,1),M3D.at<double>(0,2) );
        

        x3D.push_back(m3d);
        
        

    }
  
}





