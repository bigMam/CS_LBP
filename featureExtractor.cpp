#include "featureExtractor.h"

//�ֱ����Ե�ж�ϵ��
const float verCof[] = {1, -1, 1, -1};
const float horCof[] = {1, 1, -1, -1};
const float invCof[] = {1.414f, 0, 0, -1.414f};//45�ȶԽ���
const float diaCof[] = {0, 1.414f, -1.414f, 0};//135�ȶԽ���
const float nodCof[] = {1, -1, -1, 1};

FeatureExtractor::FeatureExtractor()
{
	//cache.init();
}

FeatureExtractor::~FeatureExtractor()
{
}
void FeatureExtractor::initCache()
{
	cache.init();
}
//��õ�ǰ�����ڵ�ֱ��ͼ
const float* FeatureExtractor::getBlockHistogram(float* buf,cv::Mat& img,int nbins)
{
	//cv::resize(img,img,cv::Size(32,32));
	assert(img.cols == 32 && img.rows == 32);

	winHistogramSize =  cache.ncells.width * cache.ncells.height * nbins;//ȷ����ǰ��ȡ��������ά��

	float* blockHist = buf;//�õ�ָ��ֱ��ͼλ�õ�ָ��

	int k, C1 = cache.count1, C2 = cache.count2, C4 = cache.count4;

	for( k = 0; k < winHistogramSize; k++ )
        blockHist[k] = 0.f;//�Ե�ǰֱ��ͼ���г�ʼ����������ʼ��Ϊ0.f


	const PixData* _pixData = &cache.pixData[0];//���pixData��ָ��
	const uchar* lbpPtr = img.ptr<uchar>(0);

	//pixData�Ĵ洢��ʽ��������ŵ�[...C1...C2...C4],���Կ��Ծ���kֵ���ζ�ȡ��������ɶ�һ��block���������صı���
    //�ȶ�Ӱ�����Ϊ1�����ؽ���ͳ�ƣ�Ҳ�����ĸ��ǵ�����
    for( k = 0; k < C1; k++ )
    {
        const PixData& pk = _pixData[k];
        const uchar* h = lbpPtr + pk.offset;
		float w = pk.histWeights[0];

        int h0 = h[0];//
        float* hist = blockHist + pk.histOfs[0] * nbins;//ȷ����ǰ����Ӱ���cell
        float t0 = hist[h0] + w;//�ۼӣ���Ӧ��ͬbinֵ
        hist[h0] = t0;//��Ӱ��cell�Ķ�Ӧ��ֱ��ͼ���и�ֵ
    }
	for( ; k < C2; k++ )//���Ƽ���
    {
        const PixData& pk = _pixData[k];
        const uchar* h = lbpPtr + pk.offset;
		int h0 = h[0];

        float* hist = blockHist + pk.histOfs[0] * nbins;
        float w = pk.histWeights[0];
        float t0 = hist[h0] + w;
        hist[h0] = t0; 

        hist = blockHist + pk.histOfs[1] * nbins;
        w = pk.histWeights[1];
        t0 = hist[h0] + w;
        hist[h0] = t0;
    }
	for( ; k < C4; k++ )//���Ƽ���
    {
        const PixData& pk = _pixData[k];
        const uchar* h = lbpPtr + pk.offset;
		int h0 = h[0];

        float* hist = blockHist + pk.histOfs[0] * nbins;
        float w = pk.histWeights[0];
        float t0 = hist[h0] + w;
        hist[h0] = t0; 

        hist = blockHist + pk.histOfs[1] * nbins;
        w = pk.histWeights[1];
        t0 = hist[h0] + w;
        hist[h0] = t0;

		hist = blockHist + pk.histOfs[2] * nbins;
        w = pk.histWeights[2];
        t0 = hist[h0] + w;
        hist[h0] = t0;

		hist = blockHist + pk.histOfs[3] * nbins;
        w = pk.histWeights[3];
        t0 = hist[h0] + w;
        hist[h0] = t0;
    }
	normalizeBlockHistogram(blockHist);//�����ɵ�blockHist���й�һ������

	return blockHist;
}
void FeatureExtractor::normalizeBlockHistogram(float* _hist) const
{
    float* hist = &_hist[0];
    size_t i, sz = winHistogramSize;//blockHistogramSize��ʾֱ��ͼ����ά��
    float sum = 0;
    for( i = 0; i < sz; i++ )
        sum += hist[i]*hist[i];//ƽ���ͣ�
    float scale = 1.f/(std::sqrt(sum)+sz*0.1f), thresh = 0.2;
    //��ñ任ϵ�����������ֵ
    for( i = 0, sum = 0; i < sz; i++ )
    {
        hist[i] = std::min(hist[i]*scale, thresh);//�ڵ�һ�εĻ����ϼ������ƽ����
        sum += hist[i]*hist[i];
    }

    scale = 1.f/(std::sqrt(sum)+1e-3f);

    for( i = 0; i < sz; i++ )
        hist[i] *= scale;//ֱ�ӳ���ϵ�����õ����յĹ�һ�����
}


/**************************��ȡHSV��ɫ�ռ�����************************/
void FeatureExtractor::HSVExtractor(const cv::Mat& src,blockFeature& feature)
{
	Mat  hsv;
	cvtColor(src, hsv, CV_RGB2HSV);

	int hueChannel = 0;
	const int hueSize = 180;
	float hranges[] = { 0, 180 };
	const float *hueRange[] = { hranges };
	calcHist( &hsv, 1, &hueChannel, Mat(), // do not use mask
		feature.hueHist, 1, &hueSize,hueRange,
		true, // the histogram is uniform
		false );
	cv::normalize(feature.hueHist,feature.hueHist,1.0,NORM_MINMAX);


	int satChannel = 1;
	const int satSize = 180;
	float sranges[] = { 0, 256 };
	const float *satRange[] = { sranges };
	calcHist( &hsv, 1, &satChannel, Mat(), // do not use mask
		feature.satHist, 1, &satSize,satRange,
		true, // the histogram is uniform
		false );
	cv::normalize(feature.satHist,feature.satHist,1.0,NORM_MINMAX);


	int valChannel = 2;
	const int valSize = 180;
	float vranges[] = { 0, 256 };
	const float *valRange[] = { vranges };
	calcHist( &hsv, 1, &valChannel, Mat(), // do not use mask
		feature.valHist, 1, &valSize,valRange,
		true, // the histogram is uniform
		false );
	cv::normalize(feature.valHist,feature.valHist,1.0,NORM_MINMAX);
	
	//��������һ�����⣬�ǽ����ݷ��������н��м��㣬
	//���Ǵ��ھ����н��м����Ϊ�����أ��д���һ���Ĳ鿴
	//���Ƿ��ھ����и�Ϊ���㣬ԭ���Ǵ���һ��compareHist����������ֱ�Ӽ�����ֱ��ͼ֮��ľ���
	//���������и�����BHATTACHARYYA�������
}


/**************************��ȡCS_LBP�ֲ���ֵͼ����************************/
template <typename _Tp> static
	void olbp_(InputArray _src, OutputArray _dst) {
		// get matrices
		Mat src = _src.getMat();
		// allocate memory for result
		_dst.create(src.rows-2, src.cols-2, CV_8UC1);
		Mat dst = _dst.getMat();
		// zero the result matrix
		dst.setTo(0);

		//std::cout<<"rows "<<src.rows<<" cols "<<src.cols<<std::endl;
		//std::cout<<"channels "<<src.channels()<<std::endl;
		//getchar();
		// calculate patterns
		for(int i=1;i<src.rows-1;i++) {
			_Tp *pre = src.ptr<_Tp>(i - 1);
			_Tp *cur = src.ptr<_Tp>(i);
			_Tp *post = src.ptr<_Tp>(i + 1);
			_Tp *dest = dst.ptr<_Tp>(i-1);

			for(int j=1;j<src.cols-1;j++) {
				_Tp center = cur[j];
				//cout<<"center"<<(int)center<<"  ";
				unsigned char code = 0;
				code |= (post[j-1] - pre[j+1] > 3) << 3;
				code |= (post[j]   - pre[j]   > 3) << 2;
				code |= (post[j+1] - pre[j-1] > 3) << 1;
				code |= (cur[j+1]  - cur[j-1] > 3) << 0;
				dest[j-1] = (int)code;//simple uniform weight��ʮ�߾��ȷֲ���ȡֵΪ0~15
				//code |= (pre[j-1]  >= center) << 7;  
				//code |= (pre[j]    >= center) << 6;  
				//code |= (pre[j+1]  >= center) << 5;  
				//code |= (cur[j+1]  >= center) << 4;  
				//code |= (post[j+1] >= center) << 3;  
				//code |= (post[j]   >= center) << 2;  
				//code |= (post[j-1] >= center) << 1;  
				//code |= (cur[j-1]  >= center) << 0;  
				//dest[j-1] = code;
				//std::cout<<(int)code<<" ";
				//cout<<(int)code<<endl;
			}
		}
}
void FeatureExtractor::CS_LBPExtractor(const cv::Mat& gray,blockFeature& feature)
{
	
	cv::Mat lbp;
	olbp_<uchar>(gray,lbp);
	//cv::imshow("lbp",lbp);

	int nbins = 16;
	int winHistogramSize = 4 * 4 * nbins;

	feature.cs_lbpFeature.resize(winHistogramSize);
	float *buf = &feature.cs_lbpFeature[0];
	if (cache.count4 == 0)
		cache.init();

	cv::Mat dest;
	cv::resize(lbp,dest,cv::Size(32,32));
	getBlockHistogram(buf,dest,nbins);
}


/**************************��ȡCanny��Ե����************************/
void FeatureExtractor::CannyExtractor(const cv::Mat& gray,blockFeature& feature)
{
	
	cv::Mat edge;
	int lower = 40;
	int upper = 40 * 1.5;
	cv::Canny(gray,edge,lower,upper);
	//cv::imshow("edge",edge);
	//cv::normalize(edge,edge,1.0,0.0,NORM_MINMAX);
	for(int i = 0; i < edge.rows; i++)
	{
		uchar* ptr = edge.ptr<uchar>(i);
		for(int j = 0; j < edge.cols; j++)
		{
			if((int)ptr[j] == 255)
			{
				ptr[j] = 1;
			}
		}
		//std::cout<<std::endl;
	}

	int nbins = 2;
	int winHistogramSize = 4 * 4 * nbins;

	feature.cannyFeature.resize(winHistogramSize);
	float *buf = &feature.cannyFeature[0];
	if (cache.count4 == 0)
		cache.init();

	getBlockHistogram(buf,edge,nbins);
}


/**************************��ȡ��ֱˮƽ����һ��������************************/
void FeatureExtractor::horVerDerExtractor(const cv::Mat& gray,blockFeature& feature)
{

	cv::Mat sobelX;
	//����Ϊ��Դͼ�񣬽��ͼ��ͼ����ȣ�x���������y����������˵Ĵ�С���߶����ӣ����ӵ�ֵ
	Sobel(gray,sobelX,CV_8U,1,0,CV_SCHARR,0.4,128);
	//imshow("X����Sobel���",sobelX);

	cv::Mat sobelY;
	Sobel(gray,sobelY,CV_8U,0,1,CV_SCHARR,0.4,128);
	//imshow("Y����Sobel���",sobelY);

	int Channel = 0;
	const int histSize = 180;
	float ranges[] = { 0, 256 };
	const float *Range[] = { ranges };
	calcHist( &sobelX, 1, &Channel, Mat(), // do not use mask
		feature.horDerHist, 1, &histSize,Range,
		true, // the histogram is uniform
		false );

	cv::normalize(feature.horDerHist,feature.horDerHist,1.0,NORM_MINMAX);
	

	calcHist( &sobelY, 1, &Channel, Mat(), // do not use mask
		feature.verDerHist, 1, &histSize,Range,
		true, // the histogram is uniform
		false );
	cv::normalize(feature.verDerHist,feature.verDerHist,1.0,NORM_MINMAX);
}


/**************************��ȡEHD��Եֱ��ͼ����************************/
//���ڴ洢��Ե�ж�ϵ��
typedef struct _filter
{
	float LT;
	float RT;
	float LB;
	float RB;
}edgeFilter;
//����4���ط������жϵ�ǰcell�Ƿ�Ϊ����
bool isMonotone(cv::Mat& cell)
{
	assert(cell.cols == 2 && cell.rows == 2 );

	uchar* ptr = cell.ptr<uchar>(0);
	uchar* ptr2 = cell.ptr<uchar>(1);
	float r = (ptr[0] + ptr[1] + ptr2[0] + ptr2[1]) / 4.0f;

	float delta = ((ptr[0] - r) * (ptr[0] - r)  + (ptr[1] - r) * (ptr[1] - r) + 
		(ptr2[0] - r) * (ptr2[0] - r) + (ptr2[1] - r) * (ptr2[1] - r)) / 4.0f;
	if (delta < 15)
		return true;
	else
		return false;
}
bool isNoDirection(cv::Mat& cell)
{
	return false;
}
//����ֵ0~5,�ֱ��ʾ��ֱ��ˮƽ��45�ȡ�135�ȡ��޹����Ե�������ޱ�Ե��Ϣ
int judgeEdgeType(cv::Mat& cell)
{
	assert(cell.cols == 2 && cell.rows == 2 );

	//��ǰcell�Ƿ�Ϊ����cell�����÷�������ж�
	if (isMonotone(cell))
	{	
		return 5;
	}
	if(isNoDirection(cell))
	{
		return 4;
	}
	int thresholdOfEdge = 11;

	float m[5] = {0,0,0,0,0};

	//ͨ������ϵ���ֱ�ǰ��Ԫ������͵���Ӧֵ
	for(int i = 0; i < 2; i++)
	{
		uchar* ptr = cell.ptr<uchar>(i);
		for(int j = 0; j < 2 ; j++)
		{
			int grayValue = ptr[j];
			m[0] += grayValue * verCof[i * 2 + j];
			m[1] += grayValue * horCof[i * 2 + j];
			m[2] += grayValue * invCof[i * 2 + j];
			m[3] += grayValue * diaCof[i * 2 + j];
			m[4] += grayValue * nodCof[i * 2 + j];
		}
	}

	float max = 0;
	int index = 0;
	//Ѱ��m�����е����ֵ
	for(int i = 0; i < 5; i++)
	{
		if(max < abs(m[i]))
		{
			max = abs(m[i]);
			index = i;
		}
	}
	if(max < thresholdOfEdge)//û��������Ӧֵ�����趨��ֵ
	{
		return 5;
	}
	else
	{
		return index;
	}
}
void FeatureExtractor::EHDExtarctor(const cv::Mat& gray,blockFeature& feature)
{
	int winWidth = gray.cols;
	int winHeight = gray.rows;
	int cellWidth = 2;//��Ԫ��ߴ�
	int cellHeight = 2;
	cv::Size numOfCells = cv::Size(winWidth / cellWidth,winHeight / cellHeight);//��ǰ�����ڵ�Ԫ�����

	int count = 0;
	for(int i = 0; i < 5;i ++)
	{
		feature.EHD[i] = 0;
	}
	//����Ҫ��������������ο������ͳ�ƹ��̣�
	for(int i = 0; i < numOfCells.width; i++)
	{
		for(int j = 0; j < numOfCells.height; j++)
		{
			cv::Mat cell = gray(cv::Rect(i * cellWidth,j * cellHeight,2,2));
			int type = judgeEdgeType(cell);
			if (type != 5)
			{
				feature.EHD[type]++;
			}
			else
			{
				count++;
			}
		}
	}

	float numOfCell = numOfCells.width * numOfCells.height;
	for(int i = 0; i < 5; i ++)
	{
		feature.EHD[i] = feature.EHD[i] / numOfCell;
	}
}


//ͳһ����������ȡ����
void FeatureExtractor::computeFeature(const cv::Mat& src,blockFeature& feature)
{
	cv::Mat dest;
	cv::resize(src,dest,cv::Size(32,32));
	cv::Mat gray;
	cv::cvtColor(dest,gray,CV_RGB2GRAY);

	HSVExtractor(dest,feature);
	CS_LBPExtractor(gray,feature);
	CannyExtractor(gray,feature);
	horVerDerExtractor(gray,feature);
	EHDExtarctor(gray,feature);
}