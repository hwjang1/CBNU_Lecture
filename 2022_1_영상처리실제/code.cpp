#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <malloc.h>
#include <vector>

#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/types_c.h"


using namespace cv;
using namespace std;

typedef struct {
	int r, g, b;
}int_rgb;

#define SQ(x) ((x)*(x))

int** IntAlloc2(int height, int width)
{
	int** tmp;
	tmp = (int**)calloc(height, sizeof(int*));
	for (int i = 0; i < height; i++)
		tmp[i] = (int*)calloc(width, sizeof(int));
	return(tmp);
}

void IntFree2(int** image, int height, int width)
{
	for (int i = 0; i < height; i++)
		free(image[i]);

	free(image);
}


float** FloatAlloc2(int height, int width)
{
	float** tmp;
	tmp = (float**)calloc(height, sizeof(float*));
	for (int i = 0; i < height; i++)
		tmp[i] = (float*)calloc(width, sizeof(float));
	return(tmp);
}

void FloatFree2(float** image, int height, int width)
{
	for (int i = 0; i < height; i++)
		free(image[i]);

	free(image);
}

int_rgb** IntColorAlloc2(int height, int width)
{
	int_rgb** tmp;
	tmp = (int_rgb**)calloc(height, sizeof(int_rgb*));
	for (int i = 0; i < height; i++)
		tmp[i] = (int_rgb*)calloc(width, sizeof(int_rgb));
	return(tmp);
}

void IntColorFree2(int_rgb** image, int height, int width)
{
	for (int i = 0; i < height; i++)
		free(image[i]);

	free(image);
}

int** ReadImage(char* name, int* height, int* width)
{
	Mat img = imread(name, IMREAD_GRAYSCALE);
	int** image = (int**)IntAlloc2(img.rows, img.cols);

	*width = img.cols;
	*height = img.rows;

	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
			image[i][j] = img.at<unsigned char>(i, j);

	return(image);
}

void WriteImage(char* name, int** image, int height, int width)
{
	Mat img(height, width, CV_8UC1);
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
			img.at<unsigned char>(i, j) = (unsigned char)image[i][j];

	imwrite(name, img);
}


void ImageShow(char* winname, int** image, int height, int width)
{
	Mat img(height, width, CV_8UC1);
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
			img.at<unsigned char>(i, j) = (unsigned char)image[i][j];
	imshow(winname, img);
	waitKey(0);
}



int_rgb** ReadColorImage(char* name, int* height, int* width)
{
	Mat img = imread(name, IMREAD_COLOR);
	int_rgb** image = (int_rgb**)IntColorAlloc2(img.rows, img.cols);

	*width = img.cols;
	*height = img.rows;

	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++) {
			image[i][j].b = img.at<Vec3b>(i, j)[0];
			image[i][j].g = img.at<Vec3b>(i, j)[1];
			image[i][j].r = img.at<Vec3b>(i, j)[2];
		}

	return(image);
}

void WriteColorImage(char* name, int_rgb** image, int height, int width)
{
	Mat img(height, width, CV_8UC3);
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++) {
			img.at<Vec3b>(i, j)[0] = (unsigned char)image[i][j].b;
			img.at<Vec3b>(i, j)[1] = (unsigned char)image[i][j].g;
			img.at<Vec3b>(i, j)[2] = (unsigned char)image[i][j].r;
		}

	imwrite(name, img);
}

void ColorImageShow(char* winname, int_rgb** image, int height, int width)
{
	Mat img(height, width, CV_8UC3);
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++) {
			img.at<Vec3b>(i, j)[0] = (unsigned char)image[i][j].b;
			img.at<Vec3b>(i, j)[1] = (unsigned char)image[i][j].g;
			img.at<Vec3b>(i, j)[2] = (unsigned char)image[i][j].r;
		}
	imshow(winname, img);

}

void DrawHistogram(char* comments, int* Hist)
{
	int histSize = 256; /// Establish the number of bins
						// Draw the histograms for B, G and R
	int hist_w = 512; int hist_h = 512;
	int bin_w = cvRound((double)hist_w / histSize);

	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(255, 255, 255));
	Mat r_hist(histSize, 1, CV_32FC1);
	for (int i = 0; i < histSize; i++)
		r_hist.at<float>(i, 0) = (float)Hist[i];
	/// Normalize the result to [ 0, histImage.rows ]
	normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

	/// Draw for each channel
	for (int i = 1; i < histSize; i++)
	{
		line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
			Point(bin_w * (i), hist_h - cvRound(r_hist.at<float>(i))),
			Scalar(255, 0, 0), 2, 8, 0);
	}

	/// Display
	namedWindow(comments, WINDOW_AUTOSIZE);
	imshow(comments, histImage);

	waitKey(0);

}

template <typename _TP>
void ConnectedComponentLabeling(_TP** seg, int height, int width, int** label, int* no_label)
{

	//Mat bw = threshval < 128 ? (img < threshval) : (img > threshval);
	Mat bw(height, width, CV_8U);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++)
			bw.at<unsigned char>(i, j) = (unsigned char)seg[i][j];
	}
	Mat labelImage(bw.size(), CV_32S);
	*no_label = connectedComponents(bw, labelImage, 8); // 0源뚯� �ы븿�� 媛�닔��

	(*no_label)--;

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++)
			label[i][j] = labelImage.at<int>(i, j);
	}
}


#define GetMax(x, y) ((x)>(y) ? x : y)
#define GetMin(x, y) ((x)<(y) ? x : y)

void AddGaussianNoise(float mean, float std, int** img, int height, int width, int** img_out)
{
	Mat bw(height, width, CV_16SC1);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++)
			bw.at<short>(i, j) = (short)img[i][j];
	}

	Mat noise_img(height, width, CV_16SC1);	randn(noise_img, mean, std);

	addWeighted(bw, 1.0, noise_img, 1.0, 0.0, bw);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++)
			img_out[i][j] = GetMin(GetMax(bw.at<short>(i, j), 0), 255);
	}
}



void SetImageValue(int** img_ptr, int height, int width, int value)
{
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			img_ptr[y][x] = value;
		}
	}
}

void SetBlockValue(int** img_ptr, int value, int y0, int y1, int x0, int x1) {

	for (int y = y0; y < y1; y++) {
		for (int x = x0; x < x1; x++) {
			img_ptr[y][x] = value;
		}
	}
}

int ex_0314()
{
	int height = 128;
	int width = 256;

	int** img_ptr = (int**)IntAlloc2(height, width);

	/*for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			img_ptr[y][x] = 255;
		}
	}*/

	SetImageValue(img_ptr, height, width, 256);
	SetBlockValue(img_ptr, 128, 32, 96, 64, 96);


	ImageShow((char*)"영상보기", img_ptr, height, width);
	IntFree2(img_ptr, height, width);

	return 0;
}


void Binarization(int** img, int** img_out, int s_height, int e_height, int s_width, int e_width, int threshold)
{

	for (int y = s_height; y < e_height; y++) {
		for (int x = s_width; x < e_width; x++) {
			if (img[y][x] >= threshold) {
				img_out[y][x] = 255;
			}
			else {
				img_out[y][x] = 0;
			}
		}
	}
}

int ex_0321()
{
	int height, width;
	int** img = (int**)ReadImage((char*)"barbara.png", &height, &width);
	int** img_out = (int**)IntColorAlloc2(height, width);


	Binarization(img, img_out, 0, height / 2, 0, width / 2, 50);
	Binarization(img, img_out, 0, height / 2, width / 2, width, 100);
	Binarization(img, img_out, height / 2, height, 0, width / 2, 150);
	Binarization(img, img_out, height / 2, height / 2, width / 2, width / 2, 200);

	ImageShow((char*)"출력", img_out, height, width);

	//WriteImage((char*)"barbara.jpg", img, height, width);
	IntFree2(img, height, width);
	IntFree2(img_out, height, width);

	return 0;
}



void AddValue(int** img, int** img_out, int height, int width, int value)
{
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			img_out[y][x] = img[y][x] + value;
		}
	}
}


#define imax(x,y) ((x>y) ? x:y)
#define imin(x,y) ((x<=y) ? x:y)

void Clipping(int** img_in, int** img_out, int height, int width)
{
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			img_out[y][x] = imin(imax(img_in[y][x], 0), 255);
		}
	}
}


int ex_0321_1()
{
	int height, width;

	int** img = (int**)ReadImage((char*)"barbara.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width);
	int** img_out2 = (int**)IntAlloc2(height, width);

	int maxvalue, minvalue;
	maxvalue = imax(10, 20);
	minvalue = imin(10, 20);
	std::cout <<"maxvalue = " << maxvalue << ", minvalue = " << minvalue <<  std::endl;

	int value = -50;
	AddValue(img, img_out, height, width, value);

	Clipping(img_out, img_out2, height, width);

	ImageShow((char*)"입력", img, height, width);
	ImageShow((char*)"출력", img_out, height, width);
	ImageShow((char*)"출력2", img_out2, height, width);

	//WriteImage((char*)"barbara.jpg", img, height, width);
	IntFree2(img, height, width);
	IntFree2(img_out, height, width);
	IntFree2(img_out2, height, width);

	return 0;
}

void AddImage(float alpha, int** img1, int** img2, int** img_out, int height, int width)
{
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			img_out[y][x] = alpha * img1[y][x] + (1 - alpha) * img2[y][x];
		}
	}

}

int ex_0321_2()
{
	int height, width;

	int** img = (int**)ReadImage((char*)"barbara.png", &height, &width);
	int** img2 = (int**)ReadImage((char*)"lena.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width);

	//float alpha = 0.5;

	ImageShow((char*)"입력1", img, height, width);
	ImageShow((char*)"입력2", img2, height, width);

	for (float alpha = 0.1; alpha < 1.0; alpha += 0.1) {
		AddImage(alpha, img, img2, img_out, height, width);
		ImageShow((char*)"출력", img_out, height, width);
	}

	for (float alpha = 0.9; alpha > 0.0; alpha -= 0.1) {
		AddImage(alpha, img, img2, img_out, height, width);
		ImageShow((char*)"출력", img_out, height, width);
	}

	

	IntFree2(img, height, width);
	IntFree2(img2, height, width);
	IntFree2(img_out, height, width);

	return 0;
}
void CopyBlock(int** img, int** img_out, int y0, int x0, int dy, int dx)
{
	for (int y = 0; y < dy; y++) {
		for (int x = 0; x < dx; x++) {
			img_out[y][x] = img[imin(y + y0, 511)][imin(x + x0, 511)];
		}
	}
}

int ex_0328()
{
	int height, width;
	int** img = (int**)ReadImage((char*)"barbara.png", &height, &width);
	int** img2 = (int**)ReadImage((char*)"lena.png", &height, &width);

	int y0 = 0, x0 = width / 2;
	int dy = height / 2, dx = width / 2;
	int** img_out = (int**)IntAlloc2(dy, dx);
	int** img_out2 = (int**)IntAlloc2(dy, dx);
	int** img_out_out = (int**)IntAlloc2(dy, dx);

	CopyBlock(img, img_out, y0, x0, dy, dx);
	CopyBlock(img2, img_out2, y0, x0, dy, dx);

	AddImage(0.5, img_out, img_out2, img_out_out, dy, dx);

	ImageShow((char*)"입력1", img_out, dy, dx);
	ImageShow((char*)"입력2", img_out2, dy, dx);
	ImageShow((char*)"출력", img_out_out, dy, dx);

	IntFree2(img, height, width);
	IntFree2(img2, height, width);
	IntFree2(img_out, dy, dx);
	IntFree2(img_out2, dy, dx);
	IntFree2(img_out_out, dy, dx);

	
	return 0;
}

void Streching(int** img, int** img_out, int height, int width, int a, int b, int c, int d)
{
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			if (img[y][x] < a) {
				img_out[y][x] = c / a * img[y][x];
			}
			else if (a <= img[y][x] && img[y][x] < b) {
				img_out[y][x] = ((float)d - c) / (b - a) * (img[y][x] - a) + c;
			}
			else {
				img_out[y][x] = (255 - d) / (255 - b) * (img[y][x] - b) + d;
			}
		}
	}
}

void GammaCorrection(float gamma, int** img, int** img_out, int height, int width)
{

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			float output = pow(img[y][x] / 255.0, 1.0 / gamma);
			img_out[y][x] = 255.0 * output + 0.5;
		}
	}
}

void find_histogram(int* hist, int** img, int height, int width)
{
	for (int i = 0; i < 256; i++) {
		hist[i] = 0;
	}

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			int p = img[y][x];
			hist[p] = hist[p] + 1;
			//for (int p = 0; p < 256; p++) {
			//	if (img[y][x] == p) {
			//		hist[p] = hist[p] + 1;
			//	}
			//}
		}
	}
}

void add_histogram(int* chist, int* hist)
{
	chist[0] = hist[0];
	for (int k = 1; k < 256; k++) {
		chist[k] = chist[k - 1] + hist[k];
	}
}

int ex_0404()
{
	int height, width;
	int** img = (int**)ReadImage((char*)"barbara.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width);

	//int a = 100, b = 140, c = 200, d = 100;
	//Streching(img, img_out, height, width, a, b, c, d);

	//GammaCorrection(1.5, img, img_out, height, width);

	int hist[256], chist[256];


	find_histogram(hist, img, height, width);

	/*for (int i = 0; i < 256; i++) {
		printf("hist[%d] = %d \n", i, hist[i]);
	}*/
	add_histogram(chist, hist);


	

	ImageShow((char*)"입력", img, height, width);
	//ImageShow((char*)"출력", img_out, height, width);
	DrawHistogram((char*)"histogram", hist);
	DrawHistogram((char*)"acc_histogram", chist);

	IntFree2(img, height, width);
	//IntFree2(img_out, height, width);
	return 0;
}

void avg33(int** img, int** img_out, int height, int width)
{
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			if (x == 0 || x == (width - 1) || y == 0 || y == (height - 1)) {
				img_out[y][x] = img[y][x];
			}
			else {
				int sum = 0;
				for (int m = -1; m <= 1; m++) {
					for (int n = -1; n <= 1; n++) {
						sum += img[y + m][x + n];
					}
				}

				img_out[y][x] = sum / 9.0 + 0.5;
			}
		}
	}

	int x, y;

	y = 0;
	for (x = 0; x < width; x++) img_out[y][x] = img[y][x];
	y = height - 1;
	for (x = 0; x < width; x++) img_out[y][x] = img[y][x];
	x = 0;
	for (y = 0; y < height; y++) img_out[y][x] = img[y][x];
	x = width - 1;
	for (y = 0; y < height; y++) img_out[y][x] = img[y][x];
}

void avg55(int** img, int** img_out, int height, int width)
{
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			if (x == 0 || x == 1 || x == (width - 1) || x == (width - 2) || y == 0 || y == 1 || y == (height - 1) || y == (height - 2)) {
				img_out[y][x] = img[y][x];
			}
			else {
				int sum = 0;
				for (int m = -2; m <= 2; m++) {
					for (int n = -2; n <= 2; n++) {
						sum += img[y + m][x + n];
					}
				}

				img_out[y][x] = sum / 25.0 + 0.5;
			}
		}
	}

	int x, y;

y = 0;
for (y = 0; y < 2; y++) {
	for (x = 0; x < width; x++) img_out[y][x] = img[y][x];
}
for (y = height - 2; y < height; y++) {
	for (x = 0; x < width; x++) img_out[y][x] = img[y][x];
}

for (x = 0; x < 2; x++) {
	for (y = 0; y < height; y++) img_out[y][x] = img[y][x];
}
for (x = width - 2; x < width; x++) {
	for (y = 0; y < height; y++) img_out[y][x] = img[y][x];
}

}


void avgNN(int N, int** img, int** img_out, int height, int width)
{
	int delta = (N - 1) / 2;
	for (int y = delta; y < height - delta; y++) {
		for (int x = delta; x < width - delta; x++) {

			int sum = 0;
			for (int m = -delta; m <= delta; m++) {
				for (int n = -delta; n <= delta; n++) {
					sum += img[y + m][x + n];
				}
			}

			img_out[y][x] = sum / (N * N) + 0.5;
		}
	}

	int x, y;

	y = 0;
	for (y = 0; y < delta; y++) {
		for (x = 0; x < width; x++) img_out[y][x] = img[y][x];
	}
	for (y = height - delta; y < height; y++) {
		for (x = 0; x < width; x++) img_out[y][x] = img[y][x];
	}

	for (x = 0; x < delta; x++) {
		for (y = 0; y < height; y++) img_out[y][x] = img[y][x];
	}
	for (x = width - delta; x < width; x++) {
		for (y = 0; y < height; y++) img_out[y][x] = img[y][x];
	}
}

int ex_0411()
{
	int height, width;
	int** img = (int**)ReadImage((char*)"barbara.png", &height, &width);
	int** img_out3x3 = (int**)IntAlloc2(width, height);
	int** img_out5x5 = (int**)IntAlloc2(width, height);
	int** img_outNN = (int**)IntAlloc2(width, height);

	avg33(img, img_out3x3, height, width);
	avg55(img, img_out5x5, height, width);
	avgNN(25, img, img_outNN, height, width);



	ImageShow((char*)"입력", img, height, width);
	ImageShow((char*)"출력3x3", img_out3x3, height, width);
	ImageShow((char*)"출력5x5", img_out5x5, height, width);
	ImageShow((char*)"출력NN", img_outNN, height, width);

	IntFree2(img, height, width);
	return 0;
}

void Masking(int** img, int** img_out, float** mask, int height, int width)
{
	for (int y = 1; y < height - 1; y++) {
		for (int x = 1; x < width - 1; x++) {

			float sum = 0.0;
			for (int m = -1; m <= 1; m++) {
				for (int n = -1; n <= 1; n++) {
					sum += mask[m + 1][n + 1] * img[imin(imax(y + m, 0), height - 1)][imin(imax(x + n, 0), width - 1)];
				}
			}
			img_out[y][x] = imin(imax(sum + 0.5, 0), 255);
		}
	}
}


int ex_0418()
{
	int height, width;
	int** img = (int**)ReadImage((char*)"barbara.png", &height, &width);
	int** img_out = (int**)IntAlloc2(width, height);
	float** mask = (float**)FloatAlloc2(3, 3);

	//mask[0][0] = 1 / 9.0;  mask[0][1] = 1 / 9.0; mask[0][2] = 1 / 9.0;
	//mask[1][0] = 1 / 9.0;  mask[1][1] = 1 / 9.0; mask[1][2] = 1 / 9.0;
	//mask[2][0] = 1 / 9.0;  mask[2][1] = 1 / 9.0; mask[2][2] = 1 / 9.0;


	//mask[0][0] = 0.0;  mask[0][1] = -0.25; mask[0][2] = 0.0;
	//mask[1][0] = -0.25;  mask[1][1] = 2.0; mask[1][2] = -0.25;
	//mask[2][0] = 0.0;  mask[2][1] = -0.25; mask[2][2] = 0.0;


	mask[0][0] = -1;  mask[0][1] = 0; mask[0][2] = 1;
	mask[1][0] = -1;  mask[1][1] = 0; mask[1][2] = 1;
	mask[2][0] = -1;  mask[2][1] = 0; mask[2][2] = 1;


	Masking(img, img_out, mask, height, width);
	


	ImageShow((char*)"입력", img, height, width);
	ImageShow((char*)"출력", img_out, height, width);

	IntFree2(img, height, width);
	IntFree2(img_out, height, width);

	return 0;
}

void Gradient(int** img, int** img_out, int height, int width)
{
	for (int y = 0; y < height - 1; y++) {
		for (int x = 0; x < width - 1; x++) {

			int fx = img[y][x + 1] - img[y][x];
			int fy = img[y + 1][x] - img[y][x];

			img_out[y][x] = abs(fx) + abs(fy);
		}
	}

}

int FindMaxValue(int** img, int height, int width);

void NormalizedByMaxvalue(int** img, int height, int width)
{
	int maxvalue = FindMaxValue(img, height, width);


	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			img[y][x] = 255 * img[y][x] / maxvalue;
		}
	}
}

int ex_0425()
{
	int height, width;
	int** img = (int**)ReadImage((char*)"lena.png", &height, &width);
	int** img_out_grad = (int**)IntAlloc2(width, height);
	int** img_out_laplace = (int**)IntAlloc2(width, height);


	Gradient(img, img_out_grad, height, width);
	NormalizedByMaxvalue(img_out_grad, height, width);

	float** mask = (float**)FloatAlloc2(3, 3);

	mask[0][0] = 0;  mask[0][1] = -1; mask[0][2] = 0;
	mask[1][0] = -1;  mask[1][1] = 4; mask[1][2] = -1;
	mask[2][0] = 0;  mask[2][1] = -1; mask[2][2] = 0;

	Masking(img, img_out_laplace, mask, height, width);
	Clipping(img_out_laplace, img_out_laplace, height, width);


	int** img_out_sobel1 = (int**)IntAlloc2(width, height);
	mask[0][0] = -1;  mask[0][1] = -2; mask[0][2] = -1;
	mask[1][0] = 0;  mask[1][1] = 0; mask[1][2] = 0;
	mask[2][0] = 1;  mask[2][1] = 2; mask[2][2] = 1;

	Masking(img, img_out_sobel1, mask, height, width);
	NormalizedByMaxvalue(img_out_sobel1, height, width);

	int** img_out_sobel2 = (int**)IntAlloc2(width, height);
	mask[0][0] = -1;  mask[0][1] = 0; mask[0][2] = 1;
	mask[1][0] = -2;  mask[1][1] = 0; mask[1][2] = 2;
	mask[2][0] = -1;  mask[2][1] = 0; mask[2][2] = 1;

	Masking(img, img_out_sobel2, mask, height, width);
	NormalizedByMaxvalue(img_out_sobel2, height, width);


	ImageShow((char*)"입력", img, height, width);
	ImageShow((char*)"출력_grad", img_out_grad, height, width);
	ImageShow((char*)"출력_laplace", img_out_laplace, height, width);

	ImageShow((char*)"출력_sobel1", img_out_sobel1, height, width);
	ImageShow((char*)"출력_sobel2", img_out_sobel2, height, width);

	IntFree2(img, height, width);
	IntFree2(img_out_grad, height, width);
	IntFree2(img_out_laplace, height, width);

	IntFree2(img_out_sobel1, height, width);
	IntFree2(img_out_sobel2, height, width);

	return 0;


}

int FindMaxValue(int** img, int height, int width) {
	int maxvalue = img[0][0];

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			maxvalue = imax(img[y][x], maxvalue);
		}
	}

	return maxvalue;
}

void Swap(int* a, int* b) {
	int buff = *a;
	*a = *b; *b = buff;
}

void Bubbling(int* A, int num)
{
	for (int i = 0; i < num - 1; i++) {
		if (A[i] > A[i + 1]) Swap(&A[i], &A[i + 1]); // 바로 이웃한 값끼리 위치 바꾸기
	}
}

void BubbleSort(int* A, int N)
{
	for (int i = 0; i < N - 1; i++) {
		Bubbling(A, N - i);
	}
}

void Get9Pixels(int* C, int y, int x, int** img)
{
	int index = 0;
	
	for (int i = -1; i <= 1; i++) {
		for (int j = -1; j <= 1; j++) {
			C[index] = img[y + i][x + j];
			index++;
		}
	}
}

void Get25Pixels(int* C, int y, int x, int** img)
{
	int index = 0;

	for (int i = -2; i <= 2; i++) {
		for (int j = -2; j <= 2; j++) {
			C[index] = img[y + i][x + j];
			index++;
		}
	}
}

void Get_NxN_Pixels(int N, int* C, int** img, int y, int x)
{
	int hN = (N - 1) / 2;
	int index = 0;

	for (int i = -hN; i <= hN; i++) {
		for (int j = -hN; j <= hN; j++) {
			C[index] = img[y + i][x + j];
			index++;
		}
	}
}

void MedianFiltering3x3(int** img, int** img_out, int height, int width)
{
	int C[9];


	for (int y = 1; y < height - 1; y++) {
		for (int x = 1; x < width - 1; x++) {
			//Get9Pixels(C, y, x, img);
			Get_NxN_Pixels(3, C, img, y, x);
			BubbleSort(C, 9);
			img_out[y][x] = C[(9 - 1) / 2];
		}
	}
}

void MedianFiltering5x5(int** img, int** img_out, int height, int width)
{
	int C[25];


	for (int y = 2; y < height - 2; y++) {
		for (int x = 2; x < width - 2; x++) {
			//Get25Pixels(C, y, x, img);
			Get_NxN_Pixels(5, C, img, y, x);
			BubbleSort(C, 25);
			img_out[y][x] = C[(25 - 1) / 2];
		}
	}

}

void MedianFilteringNxN(int N, int** img, int** img_out, int height, int width)
{
	int* C = (int*)malloc(N * N * sizeof(int));

	int hN = (N - 1) / 2;
	for (int y = hN; y < height - hN; y++) {
		for (int x = hN; x < width - hN; x++) {
			Get_NxN_Pixels(N, C, img, y, x);
			BubbleSort(C, N * N);
			img_out[y][x] = C[(N * N - 1) / 2];
		}
	}

	free(C);
}



int ex_0502()
{
	int height, width;
	int** img = (int**)ReadImage((char*)"lenaSP5.png", &height, &width);
	int** img_out = (int**)IntAlloc2(width, height);
	int** img_out_5 = (int**)IntAlloc2(width, height);

	//MedianFiltering3x3(img, img_out, height, width);
	//MedianFiltering5x5(img, img_out_5, height, width);

	MedianFilteringNxN(3, img, img_out, height, width);
	MedianFilteringNxN(5, img, img_out_5, height, width);

	ImageShow((char*)"입력", img, height, width);
	ImageShow((char*)"출력", img_out, height, width);
	ImageShow((char*)"출력5", img_out_5, height, width);


	IntFree2(img, height, width);
	IntFree2(img_out, height, width);
	IntFree2(img_out_5, height, width);

	return 0;
}

int BiLinearInterpolation(int A, int B, int C, int D, float dx, float dy)
{
	int l = (1 - dx) * (1 - dy) * A + dx * (1 - dy) * B + (1 - dx) * dy * C + dx * dy * D;
	return (l);
}

int BilinearInterpolation_2(double y, double x, int** image, int height, int width)
{
	int x_int = (int)x; // A 좌표 계산 (예) 1.9  1
	int y_int = (int)y; // A 좌표 계산 (예) 1.9  1
	int A = image[y_int][x_int];
	int B = image[y_int][x_int + 1];
	int C = image[y_int + 1][x_int];
	int D = image[y_int + 1][x_int] + 1;
	double dx = x - x_int;
	double dy = y - y_int;
	double value = (1.0 - dx) * (1.0 - dy) * A + dx * (1.0 - dy) * B + (1.0 - dx) * dy * C + dx * dy * D;
	return((int)(value + 0.5));
}

int BilinearInterpolation_3(double y, double x, int** image, int height, int width)
{
	int x_int = (int)x;
	int y_int = (int)y;
	int A = image[GetMin(GetMax(y_int, 0), height - 1)][GetMin(GetMax(x_int, 0), width - 1)];
	int B = image[GetMin(GetMax(y_int, 0), height - 1)][GetMin(GetMax(x_int + 1, 0), width - 1)];
	int C = image[GetMin(GetMax(y_int + 1, 0), height - 1)][GetMin(GetMax(x_int, 0), width - 1)];
	int D = image[GetMin(GetMax(y_int + 1, 0), height - 1)][GetMin(GetMax(x_int + 1, 0), width - 1)];
	double dx = x - x_int;
	double dy = y - y_int;
	double value = (1.0 - dx) * (1.0 - dy) * A + dx * (1.0 - dy) * B + (1.0 - dx) * dy * C + dx * dy * D;
	return((int)(value + 0.5));
}

void Upsamplingx2(int** img, int** img_out, int height_out, int width_out)
{
	for (int y = 0; y < height_out; y += 2) {
		for (int x = 0; x < width_out; x += 2) {
			img_out[y][x] = img[y / 2][x / 2];
		}
	}


	for (int y = 0; y < height_out - 2; y += 2) {
		for (int x = 0; x < width_out - 2; x += 2) {
			int A = img_out[y][x];
			int B = img_out[y][x + 2];
			int C = img_out[y + 2][x];
			int D = img_out[y + 2][x + 2];
			img_out[y][x + 1] = BiLinearInterpolation(A, B, C, D, 0.5, 0.0);
			img_out[y + 1][x] = BiLinearInterpolation(A, B, C, D, 0.0, 0.5);
			img_out[y + 1][x + 1] = BiLinearInterpolation(A, B, C, D, 0.5, 0.5);
		}
	}
}

int bilinearInterpolation3(double y, double x, int** image, int height, int width) {

	if (x < 0.0 || y < 0.0 || x > width - 2.0 || y > height - 2.0) {
		return 0;
	}

	int x_int = (int)x; // A 좌표 계산 (예) 1.9 = 1
	int y_int = (int)y; // A 좌표 계산 (예) 1.9 = 1
	int A = image[y_int][x_int];
	int B = image[y_int][x_int];
	int C = image[y_int][x_int];
	int D = image[y_int][x_int];
	double dx = x - x_int;
	double dy = y - y_int;

	double value = (1.0 - dx) * (1.0 - dy) * A + dx * (1.0 - dy) * B + (1.0 - dx) * dy * C + dx * dy * D;
	return((int)(value + 0.5));
}

void rotation(double scale, double theta, int y0, int x0, int** img, int** img_out, int height, int width) {

	for (int y_prime = 0; y_prime < height; y_prime++) {
		for (int x_prime = 0; x_prime < width; x_prime++) {

			float x = 1.0 / scale * (cos(theta) * (x_prime - x0) + sin(theta) * (y_prime - y0)) + x0;
			float y = 1.0 / scale * (-sin(theta) * (x_prime - x0) + cos(theta) * (y_prime - y0)) + y0;

			img_out[y_prime][x_prime] = bilinearInterpolation3(y, x, img, height, width);
		}
	}
}


void affineTransform(double a, double b, double c, double d, double tx, double ty, int** img, int** img_out, int height, int width)
{
	int x0 = width / 2;
	int y0 = width / 2;

	double D = a * d - b * c;

	double a_prime = d / D;
	double b_prime = -b / D;
	double c_prime = -c / D;
	double d_prime = a / D;

	printf("\n D = %f \n", D);

	for (int y_prime = 0; y_prime < height; y_prime++) {
		for (int x_prime = 0; x_prime < width; x_prime++) {

			double x = a_prime * (x_prime - x0 - tx) + b_prime + (y_prime - y0 - ty) + x0;
			double y = c_prime * (x_prime - x0 - tx) + d_prime + (y_prime - y0 - ty) + y0;

			img_out[y_prime][x_prime] = bilinearInterpolation3(y, x, img, height, width);
		}
	}
}

int ex_20220523()
{
	int height, width;
	int** img = (int**)ReadImage((char*)"lena.png", &height, &width);

	float scale = 2.3;

	int height_out = scale * height, width_out = scale * width;
	int** img_out = (int**)IntAlloc2(height_out, width_out);

	for (int y = 0; y < height_out; y++) {
		for (int x = 0; x < width_out; x++) {
			img_out[y][x] = BilinearInterpolation_3(y / scale, x / scale, img, height, width);
		}
	}
	

	ImageShow((char*)"입력", img, height, width);
	ImageShow((char*)"출력", img_out, height_out, width_out);


	IntFree2(img, height, width);
	IntFree2(img_out, height_out, width_out);
	
	return 0;
}

float MAD(int** img1, int** img2, int height, int width)
{
	int diff = 0;
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			diff += abs(img1[y][x] - img2[y][x]);

		}
	}

	float mad = (float)diff / (height * width);

	return mad;
}

float MADColor(int_rgb** img1, int_rgb** img2, int height, int width)
{
	int diff = 0;
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			diff += abs((img1[y][x].r + img1[y][x].g + img1[y][x].b) - (img2[y][x].r + img2[y][x].g + img2[y][x].b));

		}
	}

	float mad = (float)diff / (height * width);

	return mad;
}

float MSE(int** img1, int** img2, int height, int width)
{
	int diff = 0;
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			diff += (img1[y][x] - img2[y][x]) * (img1[y][x] - img2[y][x]);

		}
	}

	float mse = (float)diff / (height * width);

	return mse;
}

float MAD2(int yp, int xp, int** tp1, int dy, int dx, int** img2)
{
	
	int diff = 0;
	for (int y = 0; y < dy; y++) {
		for (int x = 0; x < dx; x++) {
			diff += abs(tp1[y][x] - img2[y + yp][x + xp]);

		}
	}

	float mad = (float)diff / (dy + dx);

	return mad;
}

void DrawBox(int y0, int x0, int dy, int dx, int** img)
{
	int y, x;
	for (x = 0; x < dx; x++) {
		img[y0][x + x0] = 255;
		img[y0 + dy][x + x0] = 255;
	}
	for (int y = 0; y < dy; y++) {
		img[y + y0][x0] = 255;
		img[y + y0][x0 + dx] = 255;
	}
}

void TemplateMaching(int** tpl, int dy, int dx, int** img, int height, int width, int* yp_out, int* xp_out, float* mad_out)
{
	float mad_min = FLT_MAX;
	int yp_min = 0, xp_min = 0;
	for (int yp = 0; yp <= height - dy; yp++) {
		for (int xp = 0; xp <= width - dx; xp++) {
			float mad = MAD2(yp, xp, tpl, dy, dx, img);
			if (mad < mad_min) {
				mad_min = mad;
				yp_min = yp;
				xp_min = xp;
			}
		}
	}

	*yp_out = yp_min;
	*xp_out = xp_min;
	*mad_out = mad_min;
}

int ex_0523()
{
	int height1, width1, height2, width2;
	int** img1 = (int**)ReadImage((char*)"lena_template.png", &height1, &width1);
	int** img2 = (int**)ReadImage((char*)"lena.png", &height2, &width2);


	int yp, xp;
	float mad;

	TemplateMaching(img1, height1, width1, img2, height2, width2, &yp, &xp, &mad);
	printf("mad(%d, %d) == %f\n", yp, xp, mad);
	
	DrawBox(yp, xp, height1, width1, img2);
	ImageShow((char*)"출력", img2, height2, width2);

	IntFree2(img1, height1, width1);
	IntFree2(img2, height2, width2);

	return 0;
}

#define DB_SIZE 510

void ReadBlock(int** img, int y, int x, int dy, int dx, int** block)
{
	for (int i = 0; i < dy; i++) {
		for (int j = 0; j < dx; j++) {
			block[i][j] = img[y + i][x + j];
		}
	}
}

void WriteBlock(int** img, int y, int x, int dy, int dx, int** block)
{
	for (int i = 0; i < dy; i++) {
		for (int j = 0; j < dx; j++) {
			img[y + i][x + j] = block[i][j];
		}
	}
}

void ReadColorBlock(int_rgb** img, int y, int x, int dy, int dx, int_rgb** block)
{
	for (int i = 0; i < dy; i++) {
		for (int j = 0; j < dx; j++) {
			block[i][j].r = img[y + i][x + j].r;
			block[i][j].g = img[y + i][x + j].g;
			block[i][j].b = img[y + i][x + j].b;
		}
	}
}

void WriteColorBlock(int_rgb** img, int y, int x, int dy, int dx, int_rgb** block)
{
	for (int i = 0; i < dy; i++) {
		for (int j = 0; j < dx; j++) {
			img[y + i][x + j].r = block[i][j].r;
			img[y + i][x + j].g = block[i][j].g;
			img[y + i][x + j].b = block[i][j].b;
		}
	}
}

void ReadAllDBImages(char filename[100], int*** tplate, int* dy_out, int* dx_out)
{
	int dy, dx;
	for (int i = 0; i < DB_SIZE; i++) {
		sprintf_s(filename, 100, ".\\db4mosaic\\dbs%04d.jpg", i);
		tplate[i] = (int**)ReadImage(filename, &dy, &dx);
	}

	*dy_out = dy;
	*dx_out = dx;
}

void ReadAllDBImages_8(char filename[100], int*** tplate, int* dy_out, int* dx_out)
{
	int dy, dx;
	for (int i = 0; i < DB_SIZE; i++) {
		sprintf_s(filename, 100, ".\\db4mosaic\\s_s_dbs%04d.jpg", i);
		tplate[i] = (int**)ReadImage(filename, &dy, &dx);
	}

	*dy_out = dy;
	*dx_out = dx;
}

void ReadAllDBImages_16(char filename[100], int*** tplate, int* dy_out, int* dx_out)
{
	int dy, dx;
	for (int i = 0; i < DB_SIZE; i++) {
		sprintf_s(filename, 100, ".\\db4mosaic\\s_dbs%04d.jpg", i);
		tplate[i] = (int**)ReadImage(filename, &dy, &dx);
	}

	*dy_out = dy;
	*dx_out = dx;
}

void ReadAllDBImages_32(char filename[100], int*** tplate, int* dy_out, int* dx_out)
{
	int dy, dx;
	for (int i = 0; i < DB_SIZE; i++) {
		sprintf_s(filename, 100, ".\\db4mosaic\\dbs%04d.jpg", i);
		tplate[i] = (int**)ReadImage(filename, &dy, &dx);
	}

	*dy_out = dy;
	*dx_out = dx;
}


void ReadAllDBColorImages(char filename[100], int_rgb*** tplate, int* dy_out, int* dx_out)
{
	int dy, dx;
	for (int i = 0; i < DB_SIZE; i++) {
		sprintf_s(filename, 100, ".\\db4mosaic\\dbs%04d.jpg", i);
		tplate[i] = (int_rgb**)ReadColorImage(filename, &dy, &dx);
	}

	*dy_out = dy;
	*dx_out = dx;
}

void ReadAllDBColorImages_8(char filename[100], int_rgb*** tplate, int* dy_out, int* dx_out)
{
	int dy, dx;
	for (int i = 0; i < DB_SIZE; i++) {
		sprintf_s(filename, 100, ".\\db4mosaic\\s_s_dbs%04d.jpg", i);
		tplate[i] = (int_rgb**)ReadColorImage(filename, &dy, &dx);
	}

	*dy_out = dy;
	*dx_out = dx;
}

void ReadAllDBColorImages_16(char filename[100], int_rgb*** tplate, int* dy_out, int* dx_out)
{
	int dy, dx;
	for (int i = 0; i < DB_SIZE; i++) {
		sprintf_s(filename, 100, ".\\db4mosaic\\s_dbs%04d.jpg", i);
		tplate[i] = (int_rgb**)ReadColorImage(filename, &dy, &dx);
	}

	*dy_out = dy;
	*dx_out = dx;
}

void ReadAllDBColorImages_32(char filename[100], int_rgb*** tplate, int* dy_out, int* dx_out)
{
	int dy, dx;
	for (int i = 0; i < DB_SIZE; i++) {
		sprintf_s(filename, 100, ".\\db4mosaic\\dbs%04d.jpg", i);
		tplate[i] = (int_rgb**)ReadColorImage(filename, &dy, &dx);
	}

	*dy_out = dy;
	*dx_out = dx;
}

int FindBestTemplate(int** block, int*** tplate, int dy, int dx)
{
	float min_mad = FLT_MAX;
	int find_idx = 0;

	for (int idx = 0; idx < DB_SIZE; idx++) {
		float mad = MAD(block, tplate[idx], dy, dx);
		if (mad < min_mad) {
			min_mad = mad;
			find_idx = idx;
		}
	}

	return find_idx;
}

int FindBestTemplateColor(int_rgb** block, int_rgb*** tplate, int dy, int dx)
{
	float min_mad = FLT_MAX;
	int find_idx = 0;

	for (int idx = 0; idx < DB_SIZE; idx++) {
		float mad = MADColor(block, tplate[idx], dy, dx);
		if (mad < min_mad) {
			min_mad = mad;
			find_idx = idx;
		}
	}

	return find_idx;
}


void MakeMosaicImage(int** img, int*** tplate, int height, int width, int dy, int dx, int** img_out)
{
	int** block = (int**)IntAlloc2(dy, dx);

	for (int y = 0; y < height; y += dy) {
		for (int x = 0; x < width; x += dx) {
			ReadBlock(img, y, x, dy, dx, block);
			int find_idx = FindBestTemplate(block, tplate, dy, dx);
			WriteBlock(img_out, y, x, dy, dx, tplate[find_idx]);
		}
	}

	IntFree2(block, dy, dx);
}

void MakeMosaicColorImage(int_rgb** img, int_rgb*** tplate, int height, int width, int dy, int dx, int_rgb** img_out)
{
	int_rgb** block = (int_rgb**)IntColorAlloc2(dy, dx);

	for (int y = 0; y < height; y += dy) {
		for (int x = 0; x < width; x += dx) {
			ReadColorBlock(img, y, x, dy, dx, block);
			int find_idx = FindBestTemplateColor(block, tplate, dy, dx);
			WriteColorBlock(img_out, y, x, dy, dx, tplate[find_idx]);
		}
	}

	IntColorFree2(block, dy, dx);
}

int ex_0603()
{
	char filename[100];
	int** tplate_8[DB_SIZE];
	int** tplate_16[DB_SIZE];
	int** tplate_32[DB_SIZE];
	int dy_8, dx_8, dy_16, dx_16, dy_32, dx_32;

	ReadAllDBImages_8(filename, tplate_8, &dy_8, &dx_8);
	ReadAllDBImages_16(filename, tplate_16, &dy_16, &dx_16);
	ReadAllDBImages_32(filename, tplate_32, &dy_32, &dx_32);



	int height, width;
	int** img = (int**)ReadImage((char*)"lena.png", &height, &width);	
	int** img_out_8 = (int**)IntAlloc2(height, width);
	int** img_out_16 = (int**)IntAlloc2(height, width);
	int** img_out_32 = (int**)IntAlloc2(height, width);
		
	MakeMosaicImage(img, tplate_8, height, width, dy_8, dx_8, img_out_8);
	MakeMosaicImage(img, tplate_16, height, width, dy_16, dx_16, img_out_16);
	MakeMosaicImage(img, tplate_32, height, width, dy_32, dx_32, img_out_32);

	ImageShow((char*)"원본", img, height, width);
	ImageShow((char*)"출력_8", img_out_8, height, width);
	ImageShow((char*)"출력_16", img_out_16, height, width);
	ImageShow((char*)"출력_32", img_out_32, height, width);

	IntFree2(img, height, width);	
	IntFree2(img_out_8, height, width);
	IntFree2(img_out_16, height, width);
	IntFree2(img_out_32, height, width);
	return 0;
}

int main()
{
	char filename[100];
	int_rgb** tplate_8[DB_SIZE];
	int_rgb** tplate_16[DB_SIZE];
	int dy_8, dx_8, dy_32, dx_32;

	ReadAllDBColorImages_8(filename, tplate_8, &dy_8, &dx_8);
	ReadAllDBColorImages_16(filename, tplate_16, &dy_32, &dx_32);

	int height, width;
	int_rgb** img = (int_rgb**)ReadColorImage((char*)"iu.jpg", &height, &width);
	int_rgb** img_out_8 = (int_rgb**)IntColorAlloc2(height, width);
	int_rgb** img_out_16 = (int_rgb**)IntColorAlloc2(height, width);

	MakeMosaicColorImage(img, tplate_16, height, width, dy_32, dx_32, img_out_16);
	MakeMosaicColorImage(img_out_16, tplate_8, height, width, dy_8, dx_8, img_out_8);

	ColorImageShow((char*)"원본", img, height, width);
	cv::waitKey(0);
	ColorImageShow((char*)"출력_16", img_out_16, height, width);
	cv::waitKey(0);
	ColorImageShow((char*)"출력_8", img_out_8, height, width);
	cv::waitKey(0);


	IntColorFree2(img, height, width);
	IntColorFree2(img_out_8, height, width);
	IntColorFree2(img_out_16, height, width);
	return 0;
	
}

//int main()
//{
//	int height, width;
//	int_rgb** img = (int_rgb**)ReadColorImage((char*)"iu.jpg", &height, &width);
//	ColorImageShow((char*)"입력", img, height, width);
//	cv::waitKey(0);
//	return 0;
//}