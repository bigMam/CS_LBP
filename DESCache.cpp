#include "DESCache.h"

DESCache::DESCache()
{
    count1 = count2 = count4 = 0;
}
DESCache::~DESCache()
{
	pixData.clear();
}


//�����Ŀ��������һ������ģ�壬Ŀ���ܹ�������������ͼ����ټ������Ӧ����ֵ������һ����LBP,�����������Ҳ�ǿ��Խ��м���ģ����ܹؼ�
//
void DESCache::init()
{
	winSize = Size(32,32);
	cellSize = Size(8,8);//cell��С��8*8��

	int i, j;
	//int nbins = 16;//������Ҫ��������ж���ô�����������������������вŻ���п��ǵģ����������ﲢ���Ǳ����
	int rawBlockSize = winSize.width * winSize.height;//block�����ظ���

	ncells = Size(winSize.width/cellSize.width, winSize.height/cellSize.height);//һ��block�а���cell������4*4��
	//winHistogramSize = ncells.width * ncells.height*nbins;
	//һ��block������������ά����4*4*16 = 256ά��ÿ��cell����һ��ֱ��ͼ��ÿ��ֱ��ͼ����9��bins��
	//ͬ������winHistogramSize��ʾ������������ά����������г�ʼ���Ĺ�������û�б�Ҫ�ģ�֮���ٵ������д����������߼�Ӧ����������
	pixData.resize(rawBlockSize*3);//��¼ÿ�����������Ϣ�������ڵ�ǰͼ���е�ƫ����������ֱ��ͼλ�ã�����Ȩ��

	//��ʼ��ÿ��pixel����ͳ�ƹ���
	count1 = count2 = count4 = 0;
	//����ɨ�贰����ĳ��block
	//���㵥��block�е����������ص�pixDataֵ
	//�Ե���block�������򻮷����£�
	//{[A][B] [C][D]}
	//{[E][F] [G][H]}
	//
	//{[I][J] [K][L]}
	//{[M][N] [O][P]}    //�ο�tornadomeet��������

	//��ÿ�����ؽ��м���
	for( j = 0; j < winSize.width; j++ )//winSize.width == 32
	{
		for( i = 0; i < winSize.height; i++ )//winSize.height == 32
		{
			PixData* data = 0;//�½�PixDataָ��
			float cellX = (j+0.5f)/(cellSize.width) - 0.5f;//cellSize.width == 8
			int icellX0 = cvFloor(cellX);
			int icellX1 = icellX0 + 1;
			cellX -= icellX0;//��ֵ
			//j = [0,3] icellX0 = -1��icellX1 = 0;
			//j = [4,11] icellX0 = 0,icellX1 = 1
			//j = [12,19] icellX0 = 1,icellX1 = 2
			//j = [20.27] icellX0 = 2,icellX1 = 3
			//j = [28,31] icellX0 = 3,icellX1 = 4

			float cellY = (i+0.5f)/(cellSize.height) - 0.5f;
			int icellY0 = cvFloor(cellY);
			int icellY1 = icellY0 + 1;
			cellY -= icellY0;
			//i = [0,3] icellY0 = -1��icellY1 = 0;
			//i = [4,11] icellY0 = 0, icellY1 = 1
			//i = [12,19] icellY0 = 1,icellY1 = 2
			//i = [20.27] icellY0 = 2,icellY1 = 3
			//i = [28,31] icellY0 = 3,icellY1 = 4

			//����������ɺ�ֱ�Ӹ���icellY0��icellY1��icellX0��icellX1���жϵ�ǰ��������λ��

			//cellY��ʾ��ֵ
			//ncells(2,2),��߾�Ϊ2
			if( (unsigned)icellX0 < (unsigned)ncells.width &&
				(unsigned)icellX1 < (unsigned)ncells.width )
			{
				if( (unsigned)icellY0 < (unsigned)ncells.height &&
					(unsigned)icellY1 < (unsigned)ncells.height )
				{
					//�ܹ����ĸ�cell����Ӱ������أ�F��G��J��K
					//ע�������unsigned�����������Լ��������ֻ����icellX0 == 0;icellY0 == 0
					//��ǰ���������ض��ĸ�cellֵ����Ӱ��
					//
					//ncells.height == 2
					//ԭ��ֱ��*nbins����ȷ��ƫ�������޸Ľ�*nbins���̺��ƣ���ǰ����ȷ��ֱ��ͼ���
					data = &pixData[rawBlockSize*2 + (count4++)];//����ǰ���ֱ࣬�ӶԵ����ࣨ4�����и�ֵ����
					data->histOfs[0] = (icellX0*ncells.height + icellY0);//
					data->histWeights[0] = (1.f - cellX)*(1.f - cellY);//Ȩ�أ��Ƚ�����ļ��㣬��ʡ�ܶ෱���Ĺ���
					data->histOfs[1] = (icellX1*ncells.height + icellY0);//
					data->histWeights[1] = cellX*(1.f - cellY);
					data->histOfs[2] = (icellX0*ncells.height + icellY1);//
					data->histWeights[2] = (1.f - cellX)*cellY;
					data->histOfs[3] = (icellX1*ncells.height + icellY1);//
					data->histWeights[3] = cellX*cellY;
					//histOfs��ʾ��ǰ���ض��ĸ�ֱ��ͼ����Ӱ�죬histWeight��ʾ��ֱ��ͼ����Ӱ���Ȩ��
					//������������

				}
				else
				{
					//����B��C��N��O
					data = &pixData[rawBlockSize + (count2++)];
					if( (unsigned)icellY0 < (unsigned)ncells.height )//unsigned(-1) > 2
					{
						//N��O
						icellY1 = icellY0;//icellY1 = 1,ԭֵΪ2
						cellY = 1.f - cellY;
					}
					data->histOfs[0] = (icellX0*ncells.height + icellY1);
					data->histWeights[0] = (1.f - cellX)*cellY;
					data->histOfs[1] = (icellX1*ncells.height + icellY1);
					data->histWeights[1] = cellX*cellY;
					//�趨����Ȩ��
					data->histOfs[2] = data->histOfs[3] = 0;
					data->histWeights[2] = data->histWeights[3] = 0;
				}
			}
			else
			{
				if( (unsigned)icellX0 < (unsigned)ncells.width )//icellX0 == 1
				{
					icellX1 = icellX0;
					cellX = 1.f - cellX;
				}

				if( (unsigned)icellY0 < (unsigned)ncells.height &&
					(unsigned)icellY1 < (unsigned)ncells.height )
				{
					//����E��H��I��L
					data = &pixData[rawBlockSize + (count2++)];
					data->histOfs[0] = (icellX1*ncells.height + icellY0);
					data->histWeights[0] = cellX*(1.f - cellY);
					data->histOfs[1] = (icellX1*ncells.height + icellY1);
					data->histWeights[1] = cellX*cellY;
					data->histOfs[2] = data->histOfs[3] = 0;
					data->histWeights[2] = data->histWeights[3] = 0;
				}
				else
				{
					//����A��D��M��P
					data = &pixData[count1++];
					if( (unsigned)icellY0 < (unsigned)ncells.height )
					{
						icellY1 = icellY0;
						cellY = 1.f - cellY;//��������������˵�������
					}
					data->histOfs[0] = (icellX1*ncells.height + icellY1);
					data->histWeights[0] = cellX*cellY;
					//������������cell����Ӱ��
					data->histOfs[1] = data->histOfs[2] = data->histOfs[3] = 0;
					data->histWeights[1] = data->histWeights[2] = data->histWeights[3] = 0;
				}
			}//������ɶԵ�ǰλ������Ӱ��cellλ�ü���Ӧ��Ȩ�ؼ������
			data->offset = (winSize.width * i + j);//��ǰ����������win�е�ƫ����
		}//forѭ����������ɶ�pixData�ĸ�ֵ��������ȷһ��block��ÿ�����ظ����bins�����乱��Ȩ��
	}

	assert( count1 + count2 + count4 == rawBlockSize );//���ձ�֤ÿ�����ؾ����봦�����ܺ�ӦΪrawBlockSize
    // defragment pixData//��Ƭ������֤�����ԣ�Ҳ���Ƕ�������ƶ�
    for( j = 0; j < count2; j++ )
        pixData[j + count1] = pixData[j + rawBlockSize];
    for( j = 0; j < count4; j++ )
        pixData[j + count1 + count2] = pixData[j + rawBlockSize*2];
    count2 += count1;
    count4 += count2;//��¼���ԵĽ���λ��

	//��pixData���б������鿴���׵õ�������������
	//const PixData* _pixData = &pixData[0];//���pixData��ָ��
	//int k, C1 = count1, C2 = count2, C4 = count4;
	//for(k = 0; k < C1; k++)
	//{
	//	const PixData& pk = _pixData[k];
	//	std::cout<<"offset:"<<pk.offset;
	//	std::cout<<" histOfs:"<<pk.histOfs[0];
	//	std::cout<<" weight:"<<pk.histWeights[0]<<std::endl;

	//}
	//for(;k < C2; k++)
	//{
	//	const PixData& pk = _pixData[k];
	//	std::cout<<"offset: "<<pk.offset;

	//	std::cout<<" histOfs0:"<<pk.histOfs[0];
	//	std::cout<<" weight0:"<<pk.histWeights[0]<<" ";

	//	std::cout<<" histOfs1:"<<pk.histOfs[1];
	//	std::cout<<" weight1:"<<pk.histWeights[1]<<std::endl;
	//}
	//for(;k < C4;k++)
	//{
	//	const PixData& pk = _pixData[k];
	//	std::cout<<"offset: "<<pk.offset;

	//	std::cout<<"histOfs0: "<<pk.histOfs[0];
	//	std::cout<<"weight0: "<<pk.histWeights[0]<<" ";

	//	std::cout<<"histOfs1: "<<pk.histOfs[1];
	//	std::cout<<"weight1: "<<pk.histWeights[1]<<" ";

	//	std::cout<<"histOfs2: "<<pk.histOfs[2];
	//	std::cout<<"weight2: "<<pk.histWeights[2]<<" ";
	//
	//	std::cout<<"histOfs3: "<<pk.histOfs[3];
	//	std::cout<<"weight3: "<<pk.histWeights[3]<<std::endl;
	//}
}