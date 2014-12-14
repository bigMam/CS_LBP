#pragma once

#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>

#include <iostream>
#include "DESCache.h"
using namespace cv;




//����ṹ�壬�����Զ���������ȡ�õ��������б��棬���������Ƿ�������д���һ������֤����������������һ���򵥿��
//���⻹��Ȩ�صĵ�����������ʱ���迼��
typedef struct _feature
{
	cv::MatND hueHist;
	cv::MatND satHist;
	cv::MatND valHist;
	vector<float> cs_lbpFeature;
	vector<float> cannyFeature;
	cv::MatND horDerHist;
	cv::MatND verDerHist;
	float EHD[5];
}blockFeature;

//��ȡ��ͬ��������������Բ�ͬ������������ά�Ȳ�ͬ���������������ͬ
class FeatureExtractor
{
public:
	FeatureExtractor();
	~FeatureExtractor();

	void initCache();
	const float* getBlockHistogram(float* buf,cv::Mat& img,int nbins);//��õ�ǰ�����ڵ�ֱ��ͼ
    virtual void normalizeBlockHistogram(float* histogram) const;//��һ��ֱ��ͼ

	void HSVExtractor(const cv::Mat& src,blockFeature& feature);
	void CS_LBPExtractor(const cv::Mat& gray,blockFeature& feature);
	void CannyExtractor(const cv::Mat& gray,blockFeature& feature);
	void horVerDerExtractor(const cv::Mat& gray,blockFeature& feature);
	void EHDExtarctor(const cv::Mat& gray,blockFeature& feature);

	void computeFeature(const cv::Mat& src,blockFeature &feature);

private:
	int nbins;//��ǰ����cell�е�ֱ��ͼά�ȣ�lbpΪ16��H��S��V�ֱ�Ϊ1����ȷ
	int winHistogramSize;//�������ڼ���õ���������ά��
	DESCache cache;
	//blockFeature feature;
};