#pragma once
#include <opencv2\core\core.hpp>
#include <iostream>

using namespace cv;
//Ŀ���ǵõ�һ������ģ�壬��Թ̶��ߴ細�ڣ�32*32�����Բ�ͬԭͼ��������Ӧ����������
//�õ�������������ά��Ϊ4*4*x��x��ʾ��ǰֱ��ͼ��ά�ȣ��ɱ�
struct PixData//�洢ĳ����������,1��PixData�ṹ��1�����ص������
{
	size_t offset;//LBPofs����ʾƫ����������ֱ�Ӷ�λ����λ��
	int histOfs[4];//histOfs[]//����ָ���ǵ�ǰpixel������cell����ֱ��ͼ��ʼλ�õ�ƫ���� ������cell�����4����
	float histWeights[4];//histWeight[]����Ȩ�أ���
};
class DESCache
{
public:
	DESCache();
    virtual ~DESCache();

	virtual void init();//�Ե�ǰ�ߴ�ļ���õ�cell�����أ�����ֱ��ͼλ�ü���ӦȨ�ع�ʽ

	vector<PixData> pixData;//�����������ص�����
	Size winSize;//winSize,ɨ�贰�ڴ�С��
	Size cellSize;
    Size ncells;//��ǰ������cell�ĸ���
    int count1, count2, count4;//ͳ��һ�������в�ͬ�������صĸ���
};

