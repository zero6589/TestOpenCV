//
//  main.cpp
//  TestOpenCV
//
//  Created by 건우 김 on 14. 2. 27..
//  Copyright (c) 2014년 zero6589@naver.com. All rights reserved.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/legacy/compat.hpp>
#include <vector>
//test

IplImage* m_image = NULL;
IplImage* m_gray = NULL;
IplImage* m_surfImage = NULL;

IplImage* m_image2 = NULL;
IplImage* m_matching_sum = NULL;
IplImage* m_gray2 = NULL;
IplImage* m_matching_gray = NULL;
IplImage* m_sumImage = NULL;

int m_contours_thresh_low = 100;
int m_contours_thresh_high = 200;
int m_surf_thresh = 500;
int m_matching_thresh = 30;
CvMemStorage* m_storage = NULL;
CvMemStorage* m_surf_storage = NULL;
CvMemStorage* m_matching_storage = NULL;

CvSURFParams m_params;
CvSeq* m_imageKeypoints = NULL;
CvSeq* m_imageDescriptors = NULL;


const char* load_file = "/Users/Macintosh/Desktop";
const char* load_file2 = "/Users/Macintosh/Desktop";
const int DIM_VECTOR = 128;

bool Init(int argc, const char * argv[]);

void change_Contour_Low(int pos);
void change_Contour_High(int pos);
void on_Contours_Trackbar();
void on_SURF_Trackbar(int pos);
void on_Matching_Trackbar(int pos);

double euclidDistance(float* vec1, float* vec2, int length);
int nearestNeighbor(float* vec, int laplacian, CvSeq* keypoints, CvSeq* descripts, int pos);
void findPairs(CvSeq* keypoints1, CvSeq* descriptors1, CvSeq* keypoints2, CvSeq* descriptors2, cv::vector<int>& ptpairs, int pos);

void Destroy();



int main (int argc, const char * argv[])
{
    // insert code here...
    if(Init(argc, argv)) {
        on_Contours_Trackbar();
        on_SURF_Trackbar(500);
        on_Matching_Trackbar(30);
        
        cvWaitKey();

        Destroy();
        
        return 0;
    }
    
    printf("Error : Init Failed!\n");
    
    return -1;
}


//초기화 함수 : 아규먼트 할당과 이미지파일 로딩, 윈도우 생성
bool Init(int argc, const char * argv[])
{
    if (argc != 3) {
        printf("Error : Argument Error!\n");
        return false;
    }
    
    load_file = argv[1];
    load_file2 = argv[2];
    
    m_image = cvLoadImage(load_file);
    //m_image = cvLoadImage("/Users/seonghyeonbag/Desktop/TestOpenCV/TestOpenCV/test.jpg");
    m_image2 = cvLoadImage(load_file2);
    m_matching_gray = cvLoadImage(load_file, CV_LOAD_IMAGE_GRAYSCALE);
    m_gray2 = cvLoadImage(load_file2, CV_LOAD_IMAGE_GRAYSCALE);
    
    if(m_image == NULL || m_image2 == NULL) {
        printf("Error : Load Image Failed!\n");
        return false;
    }
    
    CvSize sz = cvSize(m_image->width + m_image2->width, m_image->height + m_image2->height);
    
    m_sumImage = cvCreateImage(sz, IPL_DEPTH_8U, 3);
    
    //이미지1 표현
    cvSetImageROI(m_sumImage, cvRect(0, 0, m_image->width, m_image->height));
    cvCopy(m_image, m_sumImage);
    
    //이미지2 표현
    cvSetImageROI(m_sumImage, cvRect(m_image->width, m_image->height, m_image2->width, m_image2->height));
    cvCopy(m_image2, m_sumImage);
    
    cvResetImageROI(m_sumImage);
    
    cvNamedWindow("Contours", 1);
    cvMoveWindow("Contours", 300, 0);
    cvNamedWindow("SURF", 1);
    cvMoveWindow("SURF", 300 + m_image->width, 0);        
    cvNamedWindow("Key Point Matching", 1);
    cvMoveWindow("Key Point Matching", 0, 70 + m_image->height);   
    
    cvNamedWindow("Contours Control", 1);
    cvMoveWindow("Contours Control", 0, 0);
    cvResizeWindow("Contours Control", 300, 30);
    
    cvNamedWindow("Control", 1);
    cvMoveWindow("Control", 0, 200);
    cvResizeWindow("Control", 300, 50);
    
    cvCreateTrackbar("Contours Low", "Contours Control", &m_contours_thresh_low, 255, change_Contour_Low);
    cvCreateTrackbar("Contours High", "Contours Control", &m_contours_thresh_high, 255, change_Contour_High);
    
    cvCreateTrackbar("SURF", "Control", &m_surf_thresh, 2000, on_SURF_Trackbar);
    cvCreateTrackbar("Matching", "Control", &m_matching_thresh, 50, on_Matching_Trackbar);
    
    printf("Load Image Success!\n");
    printf("Init Complete!\n");
    
    return true;
}


void Destroy()
{
    cvClearSeq(m_imageKeypoints);
    cvClearSeq(m_imageDescriptors);   
    
    cvReleaseImage(&m_image);
    cvReleaseImage(&m_image2);
    cvReleaseImage(&m_gray);
    cvReleaseImage(&m_gray2);
    cvReleaseImage(&m_surfImage);
    cvReleaseImage(&m_sumImage);
    cvReleaseImage(&m_matching_sum);
    
    cvReleaseMemStorage(&m_storage);
    cvReleaseMemStorage(&m_surf_storage);        
    cvReleaseMemStorage(&m_matching_storage);    
    cvDestroyAllWindows();
}


void change_Contour_Low(int pos)
{
    m_contours_thresh_low = pos;
    
    on_Contours_Trackbar();
}

void change_Contour_High(int pos)
{
    m_contours_thresh_high = pos;
    
    on_Contours_Trackbar();
}


//외곽선 처리
void on_Contours_Trackbar()
{
    if(m_storage == NULL) {
        m_gray = cvCreateImage(cvGetSize(m_image), 8, 1);
        m_storage = cvCreateMemStorage(0);
    }
    else {
        cvClearMemStorage(m_storage);
    }
    		
    CvSeq* contours = 0;
    	
    cvCvtColor(m_image, m_gray, CV_BGR2GRAY);
    
    cvCanny(m_gray, m_gray, m_contours_thresh_low, m_contours_thresh_high);
    
    cvFindContours(m_gray, m_storage, &contours, sizeof(CvContour), CV_RETR_TREE);
    
//    cvZero(m_gray);
        
    if(contours) {
        cvDrawContours(m_gray, contours, cvScalarAll(255), cvScalarAll(128), 5);
        
        cvShowImage("Contours", m_gray);
    }
    
    on_SURF_Trackbar(m_surf_thresh);
}


//특징점 추출(SURF) 함수
void on_SURF_Trackbar(int pos)
{
    if(m_surf_storage == NULL) {
        m_surfImage = cvCreateImage(cvGetSize(m_image), 8, 1);
        m_surf_storage = cvCreateMemStorage(0);
    }
    else {
        cvClearMemStorage(m_surf_storage);	
    }
    
    m_params = cvSURFParams(pos, 1);
    
    cvExtractSURF(m_gray, 0, &m_imageKeypoints, &m_imageDescriptors, m_surf_storage, m_params, 0);
    
    int i = 0;

    m_surfImage = cvCloneImage(m_image);
    
    for(i = 0; i < m_imageKeypoints->total; ++i)
    {
        CvSURFPoint* point = (CvSURFPoint*)cvGetSeqElem(m_imageKeypoints, i);
        CvPoint center;
        int radius;
        
        center.x = cvRound(point->pt.x);
        center.y = cvRound(point->pt.y);
        radius = cvRound(point->size * 1.2 / 9.0 * 2.0);
        
        cvCircle(m_surfImage, center, radius, cvScalar(0, 0, 255), 1, 8, 0);
    }
    
    cvShowImage("SURF", m_surfImage);
}


void on_Matching_Trackbar(int pos)
{
    if(m_matching_storage == NULL) {
        m_matching_storage = cvCreateMemStorage(0);
        m_matching_sum = cvCreateImage(cvGetSize(m_sumImage), 8, 1);
    }
    else {
        cvClearMemStorage(m_matching_storage);	
    }
    
    CvSeq* keypoints2 = 0;
    CvSeq* descriptors2 = 0;
    
    cvExtractSURF(m_matching_gray, 0, &m_imageKeypoints, &m_imageDescriptors, m_matching_storage, m_params);    
    cvExtractSURF(m_gray2, 0, &keypoints2, &descriptors2, m_matching_storage, m_params);
    
    cv::vector<int> ptpairs;
    
    findPairs(m_imageKeypoints, m_imageDescriptors, keypoints2, descriptors2, ptpairs, pos);
    
    int i = 0;
    m_matching_sum = cvCloneImage(m_sumImage);
    
    for(i = 0; i < (int)ptpairs.size(); i += 2)
    {
        CvSURFPoint* pt1 = (CvSURFPoint*)cvGetSeqElem(m_imageKeypoints, ptpairs[i]);
        CvSURFPoint* pt2 = (CvSURFPoint*)cvGetSeqElem(keypoints2, ptpairs[i + 1]);
        
        CvPoint from = cvPointFrom32f(pt1->pt);
        CvPoint to = cvPoint(cvRound(m_image->width + pt2->pt.x), cvRound(m_image->height + pt2->pt.y));
        
        cvLine(m_matching_sum, from, to, cvScalar(0, 0, 255));
    }
    
    cvShowImage("Key Point Matching", m_matching_sum);
    
    cvClearSeq(keypoints2);
    cvClearSeq(descriptors2);
}


//---------2개 벡터의 유클리드 거리를 계산해서 반환-----------//
double euclidDistance(float* vec1, float* vec2, int length)
{
    double sum = 0.0;
    int i = 0;
    
    for(i = 0; i < length; ++i)
    {
        sum += (vec1[i] - vec2[i]) * (vec1[i] - vec2[i]);
    }    
    
    return sqrt(sum);
}


//---------최근접점 탐색-----------//
int nearestNeighbor(float* vec, int laplacian, CvSeq* keypoints, CvSeq* descripts, int pos)
{
    int neighbor = -1, i = 0;
    double minDist = 1e6;
    
    for(i = 0; i < descripts->total; ++i)
    {
        CvSURFPoint* pt = (CvSURFPoint*) cvGetSeqElem(keypoints, i);
        
        //라플라시안이 다른 키포인트는 무시한다.
        if(laplacian != pt->laplacian) continue;
        
        float* v = (float*) cvGetSeqElem(descripts, i);
        double d = euclidDistance(vec, v, DIM_VECTOR);
        
        //보다 가까운 점이 있으면 대치한다.
        if(d < minDist) {
            minDist = d;
            neighbor = i;
        }
    }
    
    //최근 접점에서도 거리가 임계값 이상이라면 무시한다.
    if(minDist < pos * 0.01) {
        return neighbor;
    }
    
    //최근접점이 없을 경우
    return -1;
}


//이미지 1의 키포인트와 가까운 이미지 2의 키포인트를 Pairs로 반환
void findPairs(CvSeq* keypoints1, CvSeq* descriptors1, CvSeq* keypoints2, CvSeq* descriptors2, cv::vector<int>& ptpairs, int pos)
{
    ptpairs.clear();
    int i = 0;
    
    //이미지 1의 각각의 키포인트에 대해 최근접점을 검색
    for(i = 0; i < descriptors1->total; ++i)
    {
        CvSURFPoint* pt1 = (CvSURFPoint*)cvGetSeqElem(keypoints1, i);
        
        float* desc1 = (float*)cvGetSeqElem(descriptors1, i);
        
        //최근접점을 검색
        int nn = nearestNeighbor(desc1, pt1->laplacian, keypoints2, descriptors2, pos);
        
        //최근접점이 있을 경우 이미지1과 이미지2의 인덱스를 차례로 등록
        if(nn >= 0) {
            ptpairs.push_back(i);
            ptpairs.push_back(nn);
        }
    }
}