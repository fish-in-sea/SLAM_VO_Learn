#pragma once
#include <common.h>

#include <orb_cv.hpp>

cv::Point2f pixel_to_cam ( cv::Point2f& p, const cv::Mat& K );
void Normalize( vector<cv::Point2f> &vKeys, vector<cv::Point2f> &vNormalizedPoints, cv::Mat &T);
void DecomposeE(const cv::Mat &E, cv::Mat &R1, cv::Mat &R2, cv::Mat &t);
void Triangulate(vector<cv::Point2f> &mvKeys1, vector<cv::Point2f> &mvKeys2,
                cv::Mat &P1, cv::Mat &P2,vector<cv::Mat> &x3D);