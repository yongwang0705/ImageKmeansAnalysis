#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <unistd.h>
using namespace std;
using namespace cv;

int Array_LBP[10000][10000];
float weight=0.1;
string window_name;
static Mat img_original,img_tmp,img_Gray;
int palette[100][3]={0};
int clusters[10000][10000]={0}; // image 10000*10000 k can be 100 maximum
void create_palette(int k){
    srand((unsigned int)time(NULL));//set current time as the seed of random function
    for(int indiceK=0;indiceK<k;indiceK++){
        palette[indiceK][0]=rand()%255;
        palette[indiceK][1]=rand()%255;
        palette[indiceK][2]=rand()%255;
    }
    printf("The colors for labeling are:\n");
    for(int indiceK=0;indiceK<k;indiceK++){
        printf("%d,%d,%d\n",palette[indiceK][0],palette[indiceK][1],palette[indiceK][2]);
    }
}
void extractTexture()//extract LBP features into Array_LBP
{
    int temp[8];
    cvtColor(img_original,img_Gray,COLOR_BGR2GRAY);
    int step1=img_Gray.cols;

    for(int j=1; j<img_Gray.rows-1; j++)
        for(int i=1; i<img_Gray.cols-1; i++) {

            if(img_Gray.at<uchar>(j,i-1)>img_Gray.at<uchar>(j,i))
                temp[0]=1;
            else
                temp[0]=0;

            if(img_Gray.at<uchar>((j-1),i)>img_Gray.at<uchar>(j,i))
                temp[1]=1;
            else
                temp[1]=0;

            if(img_Gray.at<uchar>((j-1),i+1)>img_Gray.at<uchar>(j,i))
                temp[2]=1;
            else
                temp[2]=0;

            if(img_Gray.at<uchar>(j,i+1)>img_Gray.at<uchar>(j,i))
                temp[3]=1;
            else
                temp[3]=0;

            if(img_Gray.at<uchar>((j+1),i+1)>img_Gray.at<uchar>(j,i))
                temp[4]=1;
            else
                temp[4]=0;
            if(img_Gray.at<uchar>((j+1),i)>img_Gray.at<uchar>(j,i))
                temp[5]=1;
            else
                temp[5]=0;
            if(img_Gray.at<uchar>((j+1),i-1)>img_Gray.at<uchar>(j,i))
                temp[6]=1;
            else
                temp[6]=0;
            if(img_Gray.at<uchar>(j,i-1)>img_Gray.at<uchar>(j,i))
                temp[7]=1;
            else
                temp[7]=0;
            int center_lbp=temp[0]*128+temp[1]*64+temp[2]*32+temp[3]*16
                          +temp[4]*8+temp[5]*4+temp[6]*2+temp[7]*1;
            Array_LBP[j][i]=center_lbp;
        }

        for(int j=0; j<img_Gray.rows; j++){
            Array_LBP[j][0]=Array_LBP[j][1];
            Array_LBP[j][img_Gray.rows-1]=Array_LBP[j][img_Gray.rows-2];
        }
        for(int i=0; i<img_Gray.cols; i++){
            Array_LBP[0][i]=Array_LBP[1][i];
            Array_LBP[img_Gray.cols-1][i]=Array_LBP[img_Gray.cols-1][i];
        }
}
void kmeans_ppm(int k)
{
//random points
    struct s_knodes{
        int blue;
        int green;
        int red;
        int x;
        int y;
        int LBP;
    };
    struct s_knodes knodes[100];
    for(int i=0;i<100;i++)
        knodes[i]={0,0,0,0,0};

    int blue,red,green,sum=0,sums[100][6],means[100][6],count_k[100];

    srand((unsigned int)time(NULL));//
    for(int i=0;i<k;i++)
    {
        knodes[i].x = rand()%300+1;//
        knodes[i].y = rand()%300+1;//
    }

    float Edistance,old_Edistance=65536;
    //first calculate Eud distance
	for (int i = 0; i < img_original.rows; i++){
		for (int j = 0; j < img_original.cols; j++){
            old_Edistance=65536;
			for(int indiceK=0;indiceK<k;indiceK++){//calculate and compare the Eud distance, and choose cluster
                sum += pow(img_original.at<Vec3b>(knodes[indiceK].x, knodes[indiceK].y)[0] - img_original.at<Vec3b>(i, j)[0], 2.0);
                sum += pow(img_original.at<Vec3b>(knodes[indiceK].x, knodes[indiceK].y)[1] - img_original.at<Vec3b>(i, j)[1], 2.0);
                sum += pow(img_original.at<Vec3b>(knodes[indiceK].x, knodes[indiceK].y)[2] - img_original.at<Vec3b>(i, j)[2], 2.0);
                sum += weight*pow(knodes[indiceK].x- i, 2.0);
                sum += weight*pow(knodes[indiceK].y- j, 2.0);
                sum += pow(Array_LBP[knodes[indiceK].x][knodes[indiceK].y]-Array_LBP[i][j], 2.0);
                Edistance=sqrt(sum);
                sum=0;
                if(Edistance<old_Edistance){//new cluster
                    clusters[i][j]=indiceK;
                    old_Edistance=Edistance;
                }
			}
		}
	}
	//calculate 50 times
	old_Edistance=65536;
	for(int n=0; n<50; n++){
        for (int indiceK = 0; indiceK < k; indiceK++){ //calculate means
            for (int i = 0; i < img_original.rows; i++){
                for (int j = 0; j < img_original.cols; j++){
                    if(clusters[i][j]==indiceK)
                    sums[indiceK][0]+=img_original.at<Vec3b>(i,j)[0];
                    sums[indiceK][1]+=img_original.at<Vec3b>(i,j)[1];
                    sums[indiceK][2]+=img_original.at<Vec3b>(i,j)[2];
                    sums[indiceK][3]+=i;
                    sums[indiceK][4]+=j;
                    sums[indiceK][5]+=Array_LBP[i][j];
                    count_k[indiceK]++;
                }
            }
            knodes[indiceK].blue=sums[indiceK][0]/count_k[indiceK];
            knodes[indiceK].green=sums[indiceK][1]/count_k[indiceK];
            knodes[indiceK].red=sums[indiceK][2]/count_k[indiceK];
            knodes[indiceK].x=sums[indiceK][3]/count_k[indiceK];
            knodes[indiceK].y=sums[indiceK][4]/count_k[indiceK];
            knodes[indiceK].LBP=sums[indiceK][5]/count_k[indiceK];
            memset(sums,0,sizeof(sum)*sizeof(int));
            memset(count_k,0,sizeof(count_k)*sizeof(int));
        }

        for (int i = 0; i < img_original.rows; i++){
            for (int j = 0; j < img_original.cols; j++){
                //old_Edistance=65536;
                for(int indiceK=0;indiceK<k;indiceK++){//calculate and compare the Eud distance, and choose cluster
                    sum += pow(knodes[indiceK].blue - img_original.at<Vec3b>(i, j)[0], 2.0);
                    sum += pow(knodes[indiceK].green - img_original.at<Vec3b>(i, j)[1], 2.0);
                    sum += pow(knodes[indiceK].red - img_original.at<Vec3b>(i, j)[2], 2.0);
                    sum += weight*pow(knodes[indiceK].x - i, 2.0);
                    sum += weight*pow(knodes[indiceK].y - j, 2.0);
                    sum += pow(knodes[indiceK].LBP-Array_LBP[i][j], 2.0);
                    Edistance=sqrt(sum);
                    sum=0;
                    if(old_Edistance>Edistance){//new cluster
                        clusters[i][j]=indiceK;
                        old_Edistance=Edistance;
                    }
                }
            }
        }
    }
}

void save_ppm(int k)
{
//according the result of kmeans, differents colors are put on differents regions.
    for(int indiceK=0;indiceK<k;indiceK++){
        for (int i = 0; i < img_original.rows; i++){
            for (int j = 0; j < img_original.cols; j++){
                if(clusters[i][j]==indiceK){
                    img_tmp.at<Vec3b>(i, j)[0]=palette[indiceK][0];
                    img_tmp.at<Vec3b>(i, j)[1]=palette[indiceK][1];
                    img_tmp.at<Vec3b>(i, j)[2]=palette[indiceK][2];
                }
            }
        }
    }

}

int main(int argc, char** argv)
{
    int k;
    cout<<"Please input the number of K: ";
    cin >> k;

    CommandLineParser parser( argc, argv, "{@input | ../data/lena.jpg | input image}" );
    window_name=parser.get<String>( "@input" );

    img_original = imread(window_name);
    if( img_original.empty() )
    {
      cout << "Could not open or find the image!\n" << endl;
      cout << "Usage: " << argv[0] << " <Input image>" << endl;
      return -1;
    }
    img_tmp=img_original.clone();

	if ( !img_original.data ) {
		cout << "error read image" << endl;
		return -1;
	}
	create_palette(k);
    extractTexture();
    kmeans_ppm(k);
    save_ppm(k);
	namedWindow(window_name);
	imshow(window_name,img_original);
	imshow("with x,y",img_tmp);
	waitKey();
	return 0;

}
