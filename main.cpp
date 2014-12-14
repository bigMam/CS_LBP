//#include <opencv2\core\core.hpp>
//#include <opencv2\highgui\highgui.hpp>
//#include <opencv2\imgproc\imgproc.hpp>

#include <iostream>
#include "featureExtractor.h"
#include <opencv2\opencv.hpp>
using namespace cv;
//ά���˲�����ԭʼͼ�����
void wiener(const cv::Mat gray, cv::Mat& dest)
{
	cv::Mat localMean = cv::Mat(gray.rows,gray.cols,CV_32FC1);
	cv::Mat localVar = cv::Mat(gray.rows,gray.cols,CV_32FC1);
	cv::Mat gray2 = cv::Mat(gray.rows,gray.cols,CV_32FC1);
	/*�õ��˲�ģ��*/
	cv::Mat model = cv::Mat::zeros(3,3,CV_32FC1);
	for (int row=0;row < model.rows;row++)
	{
		float* pData = model.ptr<float>(row);//��ȡ��row�е�����ָ�룬��Ϊ��������Ϊ�����ͣ���ˣ�ͨ��data.ptr��step��õ��ֽ�ָ����Ҫת��Ϊfloat* ������ָ��
		for (int col=0;col < model.cols;col++)
		{
			pData[col] = 1;//ȫ1�˲�ģ�壿
		}
	}
	/*�õ�ԭʼͼ����ģ�������ͼ��*/
	cv::Mat dst;
	cv::filter2D(gray,dst,-1,model);

	/*�õ�ԭʼͼ������ƽ����ģ�������ͼ��*/
	cv::Mat multi_gray;
	cv::multiply(gray,gray,multi_gray);
	cv::Mat dst2;
	cv::filter2D(multi_gray,dst2,-1,model);

	for(int i = 0; i < gray.rows; i++)
	{
		float* pdst = dst.ptr<float>(i);
		float* pMean = localMean.ptr<float>(i);
		float* pdst2 = dst2.ptr<float>(i);
		float* pVar = localVar.ptr<float>(i);
	}
}



//���㵱ǰ������Ŀ������֮��Ĳ�ֵ
double distinguish(blockFeature target, blockFeature current)
{
	cv::MatND targetLBP = cv::Mat(target.cs_lbpFeature);
	cv::MatND currentLBP = cv::Mat(current.cs_lbpFeature);

	cv::MatND targetCanny = cv::Mat(target.cannyFeature);
	cv::MatND currentCanny = cv::Mat(current.cannyFeature);

	double hueDistance = compareHist(target.hueHist,current.hueHist,CV_COMP_BHATTACHARYYA);
	double satDistance = compareHist(target.satHist,current.satHist,CV_COMP_BHATTACHARYYA);
	double valDistance = compareHist(target.valHist,current.valHist,CV_COMP_BHATTACHARYYA);
	double lbpDistance = compareHist(targetLBP,currentLBP,CV_COMP_BHATTACHARYYA);
	double cannyDistance = compareHist(targetCanny,currentCanny,CV_COMP_BHATTACHARYYA);
	double horDerDistance = compareHist(target.horDerHist,current.horDerHist,CV_COMP_BHATTACHARYYA);
	double verDerDistance = compareHist(target.verDerHist,current.verDerHist,CV_COMP_BHATTACHARYYA);

	cv::MatND targetEHD = cv::Mat(5,1,CV_32F);
	cv::MatND currentEHD = cv::Mat(5,1,CV_32F);
	for(int i = 0; i < 5; i++)
	{
		float* targetPtr = targetEHD.ptr<float>(i);
		float* currentPtr = currentEHD.ptr<float>(i);
		targetPtr[0] = target.EHD[i];
		currentPtr[0] = current.EHD[i];
	}
	double EHDDistance = compareHist(targetEHD,currentEHD,CV_COMP_BHATTACHARYYA);
	//��ɾ��������̣�

	//���㵱ǰͼ�����Ŀ��ͼ���Ĳ���ֵ
	double dissimilarity = (hueDistance + satDistance + valDistance + lbpDistance 
		+ cannyDistance + horDerDistance + verDerDistance + EHDDistance) / 8.0f;

	std::cout<<"dissimilarity is :"<<dissimilarity<<std::endl;
	return dissimilarity;
}

int main()
{   

	//const char* filename = "D:\\ImageDataSets\\trackingSamples\\forward_li_2.avi";
	//cv::VideoCapture cap;
	//if(!cap.open(filename))
	//{
	//	std::cout<<"can not open the cource viedo"<<std::endl;
	//}
	//cv::Mat frame;
	//while(cap.read(frame))
	//{
	//	cv::imshow("viedo",frame);
	//	cv::waitKey(20);
	//}
    const char* imageName = "E:/lena.jpg"; //����һ��jpgͼƬ��E�̻���������Ŀ¼
	cv::Mat src = cv::imread(imageName);//��ȡԭʼͼ��
	cv::imshow("src",src);
	cv::Mat srcSub = src(cv::Rect(200 ,330,60,60));
	cv::Mat srcSub2 = src(cv::Rect(100,300,60,60));
	
	cv::imshow("sub",srcSub);
	cv::imshow("sub2",srcSub2);
	FeatureExtractor extractor;
	extractor.initCache();

	blockFeature target;
    extractor.computeFeature(srcSub,target);


	//std::cout<<std::endl;
	blockFeature feature;
	extractor.computeFeature(srcSub2,feature);

	distinguish(target,feature);
    cv::waitKey();
	cv::destroyAllWindows();
	system("pause");
    return 0;
}