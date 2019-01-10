

#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <cstdio>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>


using namespace cv;
using namespace std;


Mat FillHoles(Mat binarizedImage){
    
    Mat element = getStructuringElement( MORPH_ELLIPSE,
                                        Size( 2*2 + 1, 2*2+1 ),
                                        Point( 2, 2 ) );
    
    Mat erosion_dst2,dilation_dst2;
    
    erode(binarizedImage, erosion_dst2, element );
    dilate(erosion_dst2, dilation_dst2, element);
    
    
    return dilation_dst2;
}



void FindBlobs(const cv::Mat &binary, std::vector < std::vector<cv::Point2i> > &blobs)
{
    blobs.clear();
    
    Mat label_image(binary.size(), CV_8UC3);
    binary.convertTo(label_image, CV_32SC1);
    
    int label_count = 2;
    
    for(int y=0; y < label_image.rows; y++) {
        int *row = (int*)label_image.ptr(y);
        for(int x=0; x < label_image.cols; x++) {
            if(row[x] != 0) { // se è sfondo
                continue;
            }
            
            Rect rect;
            floodFill(label_image, cv::Point(x,y), label_count, &rect, 0, 0, 4);
            
            vector <cv::Point2i> blob;
            
            for(int i=rect.y; i < (rect.y+rect.height); i++) {
                int *row2 = (int*)label_image.ptr(i);
                for(int j=rect.x; j < (rect.x+rect.width); j++) {
                    if(row2[j] != label_count) {
                        continue;
                    }
                    
                    blob.push_back(cv::Point2i(j,i));
                }
            }
            
            blobs.push_back(blob);
            
            label_count++;
        }
    }
}


int main(int argc, char**argv){
    
    //========TASK 1==========================
    
    //Carico le immagini in scala di grigi
    Mat image11,image12,image13;
    image11 = imread("../../fruit_images/first task/C0_000001.png", CV_LOAD_IMAGE_GRAYSCALE);
    image12 = imread("../../fruit_images/first task/C0_000002.png", CV_LOAD_IMAGE_GRAYSCALE);
    image13 = imread("../../fruit_images/first task/C0_000003.png", CV_LOAD_IMAGE_GRAYSCALE);

  //  imshow( "Original Image 1", image11 );                   // Show our image inside it.
  //  imshow( "Original Image 2", image12 );                   // Show our image inside it.
   // imshow( "Original Image 3", image13 );                   // Show our image inside it.
    Mat currentGrayImage1 = image13.clone();
    
    //BINARIZZO PER AVERE UNA MASCHERA PRIVA DEI DIFETTI PIU' GRANDI
    Mat binarizeImage1;
    threshold(currentGrayImage1, binarizeImage1, 28, 255, THRESH_BINARY);
    
   // imshow("Bin Output 1", binarizeImage1);
    
    //ELIMINO I BUCHI ESTERNI AL FRUTTO
    Mat dilation_dst1 = FillHoles(binarizeImage1);
  //  imshow("EroDilat1", dilation_dst1);
    
    //Carico le immagini a colori
    Mat coloredImage11 = imread("../../fruit_images/first task/C1_000001.png", CV_LOAD_IMAGE_COLOR);
    Mat coloredImage12 = imread("../../fruit_images/first task/C1_000002.png", CV_LOAD_IMAGE_COLOR);
    Mat coloredImage13 = imread("../../fruit_images/first task/C1_000003.png", CV_LOAD_IMAGE_COLOR);
    Mat currentColorImage1 = coloredImage13.clone();

    //Vado ad eliminare i piccoli buchi lasciati fuori dalla binarizzazione e dalla dilation/erosion
    
    //Cerco il contorno del frutto
    vector<vector<Point>>cont;
    findContours(dilation_dst1, cont, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

    //Trovo il contorno più grande
    int iMax = 0;
    u_long sizeMax = 0;
    for(int i=0;i<cont.size();i++){
        u_long size = cont[i].size();
        if (size>sizeMax) {
            sizeMax = cont[i].size();
            iMax = i;
        }
    }
    
    //Disegno il contorno della mela
    dilation_dst1 = 0;
    for (int i = 0; i < cont[iMax].size(); i++)
        dilation_dst1.at<unsigned char>(cont[iMax][i]) = 255;
    
   // imshow("contour fruit", dilation_dst1);
    
    //elimino con il flood fill tutti i buchi interni al frutto
    floodFill(dilation_dst1, cv::Point(0,0), Scalar(255));
    //imshow("floodFill", dilation_dst1);

    //Uso la maschera per eliminare lo sfondo dalle immagini
    Mat inverted = 255 - dilation_dst1;
    Mat noBack1;
    currentGrayImage1.copyTo(noBack1, inverted);
   // imshow("noBack1", noBack1);
    
    
    //Blur per eliminare altre imperfezioni, e Canny per delineare gli edge
    Mat edge;
    blur(noBack1, edge, Size(3, 3));
    blur(edge, edge, Size(3, 3));
    
    erode(inverted,inverted,Mat(), cvPoint(-1, -1), 7);
   // imshow( "erosion", inverted );
    
    Mat edges_detected;
    Canny(edge, edges_detected, 70, 150);
   // imshow( "edges detected", edges_detected );
    
    Mat onlyDefects = edges_detected & inverted;
    drawContours(onlyDefects, cont, 0, 0, 8);
    //imshow( "onlyDefects", onlyDefects );

    //Operazione di chiusura per uniformare i contorni frastagliati trovati
    Mat structuringElement = getStructuringElement(MORPH_ELLIPSE, Size(40, 40));
    morphologyEx( onlyDefects, onlyDefects, MORPH_CLOSE, structuringElement );
    // imshow( "onlyDefectsUnify", onlyDefects );
    
    //Cerco i Blob per poi disegnare i rettangoli sui difetti
    vector<vector<Point>>defectBlobs;
    FindBlobs(255 - onlyDefects, defectBlobs);

    Mat finalColored1 = currentColorImage1.clone();
    for(int i=0;i<defectBlobs.size();i++){
        rectangle(finalColored1, boundingRect(defectBlobs[i]), Scalar{ 0,0,230 },2);
    }
    //imshow( "finalColored1", finalColored1 );
    

    //========TASK 2==========================
    
    //Carico le immagini in scala di grigi
    Mat image21,image22;
    image21 = imread("../../fruit_images/second task/C0_000004.png", CV_LOAD_IMAGE_GRAYSCALE);
    image22 = imread("../../fruit_images/second task/C0_000005.png", CV_LOAD_IMAGE_GRAYSCALE);
    Mat currentGrayImage2 = image21.clone();
    //imshow("greyImage", currentGrayImage2);
    
    Mat binarizeImage2;
    threshold(currentGrayImage2, binarizeImage2, 15, 255, THRESH_BINARY);
    
    //imshow("binarize", binarizeImage2);

    Mat dilation_dst2 = FillHoles(binarizeImage2);
    //imshow("dilation_dst2", dilation_dst2);

    //Cerco il contorno del frutto
    vector<vector<Point>>cont2;
    findContours(dilation_dst2, cont2, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

    int iMax2 = 0;
    u_long sizeMax2 = 0;
    for(int i=0;i<cont2.size();i++){
        u_long size2 = cont2[i].size();
        if (size2>sizeMax2) {
            sizeMax2 = cont2[i].size();
            iMax2 = i;
        }
    }
    
    //Disegno il contorno della mela
    dilation_dst2 = 0;
    for (int i = 0; i < cont2[iMax2].size(); i++)
        dilation_dst2.at<unsigned char>(cont2[iMax2][i]) = 255;
    
    
    floodFill(dilation_dst2, cv::Point(0,0), Scalar(255));
    //imshow("floodFill2", dilation_dst2);

    //Uso la maschera per eliminare lo sfondo dalle immagini
    Mat inverted2 = 255 - dilation_dst2;
    Mat noBack2;
    currentGrayImage2.copyTo(noBack2, inverted2);
    //imshow("noBack2", noBack2);
    
    erode(inverted2, inverted2, Mat(), cvPoint(-1, -1), 8);
  
 
    
    Mat imageColor21,imageColor22;
    imageColor21 = imread("../../fruit_images/second task/C1_000004.png", CV_LOAD_IMAGE_COLOR);
    //imshow("imageColor12", imageColor21);
    imageColor22 = imread("../../fruit_images/second task/C1_000005.png", CV_LOAD_IMAGE_COLOR);
    //imshow("imageColor22", imageColor22);
    Mat currentColorImage2 = imageColor21.clone();
    
    Mat original = currentColorImage2.clone();
    Mat coloured = currentColorImage2.clone();
    cvtColor(coloured, coloured, CV_BGR2HSV);
    //imshow("HSV", coloured);
    
    vector<Mat>array;
    split(coloured, array);
    //applico la maschera
    for (int i = 0; i < 3; i++)
    {
        array[i] = array[i] & inverted2;
    }
    
    merge(array, coloured);
    //imshow("TEST2", coloured);
    
   
    //CARICO I SAMPLE
    vector<Mat>samples;
    Mat sample1,sample2,sample3,sample4,sample5,sample6;
    sample1 = imread("../../fruit_images/second task/C1_000004T1.png", CV_LOAD_IMAGE_COLOR);
    samples.push_back(sample1);
  //  imshow("sample1", sample1);
    sample2 = imread("../../fruit_images/second task/C1_000004T2.png", CV_LOAD_IMAGE_COLOR);
    samples.push_back(sample2);
  //  imshow("sample2", sample2);
    sample3 = imread("../../fruit_images/second task/C1_000005T1.png", CV_LOAD_IMAGE_COLOR);
    samples.push_back(sample3);
 //   imshow("sample3", sample3);
    sample4= imread("../../fruit_images/second task/C1_000004T3.png", CV_LOAD_IMAGE_COLOR);
    samples.push_back(sample4);
   // imshow("sample4", sample4);
    sample5 = imread("../../fruit_images/second task/C1_000005T2.png", CV_LOAD_IMAGE_COLOR);
    samples.push_back(sample5);
   // imshow("sample5", sample5);
    sample6 = imread("../../fruit_images/second task/C1_000004T4.png", CV_LOAD_IMAGE_COLOR);
    samples.push_back(sample6);
//  imshow("sample6", sample6);
    

    //Calcolo della matrice di covarianza e vettore delle medie dei campioni
    Mat currentSample;
    Mat covarTot = (Mat_<double>(3, 3) << 0, 0, 0, 0, 0, 0, 0, 0, 0);
    Mat muTot = (Mat_<double>(1, 3) << 0, 0, 0);
    for(int i=0;i<samples.size();i++){
        currentSample = samples[i].clone();
        cvtColor(currentSample, currentSample, CV_BGR2HSV);
        Mat covar = (Mat_<double>(3, 3) << 0, 0, 0, 0, 0, 0, 0, 0, 0);
        Mat mu= (Mat_<double>(1, 3) << 0, 0, 0);
        Mat sample_copy = currentSample.clone();
        sample_copy = currentSample.reshape(1,currentSample.rows * currentSample.cols);
        
        //Uso SCALE per scalare le covarianze in base all'area del campione in considerazione
        calcCovarMatrix(sample_copy, covar, mu,CV_COVAR_NORMAL|CV_COVAR_ROWS | CV_COVAR_SCALE);
        covarTot = covarTot + covar;
        muTot = muTot + mu;
     }
    
    //Faccio la media delle medie
    muTot = muTot / samples.size();
    //Inverto la matrice delle covarianze
    invert(covarTot, covarTot, DECOMP_SVD);
   
    //CALCOLO DELLA DISTANZA DI MAHALANOBIS
    Mat currentHSVImage = coloured.clone();

    Mat pixel = (Mat_<double>(1, 3) << 0, 0, 0);
    Mat distanceImage = currentHSVImage.clone();
    
    Mat stains = Mat(currentHSVImage.size(), currentHSVImage.type());
    stains = Scalar{ 0,0,0 };
    
    //la distanza è calcolata per ogni pixel
    for(int i=0;i<currentHSVImage.rows;i++){
        for(int j=0;j<currentHSVImage.cols;j++){
            //Calcolo la distanza solo per i pixel che rappresentano il frutto, evitando così lo sfondo
            if (inverted2.at<uchar>(j, i) == 255) {
                pixel.at<double>(0) =  currentHSVImage.at<Vec3b>(j, i)[0];
                pixel.at<double>(1) =  currentHSVImage.at<Vec3b>(j, i)[1];
                pixel.at<double>(2) =  currentHSVImage.at<Vec3b>(j, i)[2];
                double distance = Mahalanobis(pixel, muTot, covarTot);
                
                //mi salvo solo i pixel sotto una certa distanza e che quindi rappresentano la macchia
                if (distance <=2.5)
                {
                    stains.at<Vec3b>(j,i) = pixel;
                }
                
            }
        }
    }
   // imshow("stains", stains);
  
    Mat stainsGrey;
    cvtColor(stains, stainsGrey, CV_BGR2GRAY);
    medianBlur(stainsGrey, stainsGrey, 5);
    //imshow("stains Grey", stainsGrey);
    for (int x = 0; x < stainsGrey.rows; x++){
        for (int y = 0; y < stainsGrey.cols; y++){
            //converto il grigio in bianco
            if (stainsGrey.at<unsigned char>(Point(y, x)) != 0){
                stainsGrey.at<unsigned char>(Point(y, x)) = 255;
            }
        }
    }
   // imshow("stains white", stainsGrey);
    
    
   vector<vector<Point>>russetBlobs;
   FindBlobs(255 - stainsGrey, russetBlobs);
    
    Mat finalColored = currentHSVImage.clone();
    for(int i=0;i<russetBlobs.size();i++){
        if(russetBlobs[i].size() > 50){
            rectangle(original, boundingRect(russetBlobs[i]), Scalar{ 0,0,230 },2);
        }
    }
    
    //imshow("originalWithRectangle", original);

    //========TASK 3==========================
    
    Mat image31,image32,image33,image34,image35;
    image31 = imread("../../fruit_images/final challenge/C0_000006.png", CV_LOAD_IMAGE_GRAYSCALE);
    image32 = imread("../../fruit_images/final challenge/C0_000007.png", CV_LOAD_IMAGE_GRAYSCALE);
    image33 = imread("../../fruit_images/final challenge/C0_000008.png", CV_LOAD_IMAGE_GRAYSCALE);
    image34 = imread("../../fruit_images/final challenge/C0_000009.png", CV_LOAD_IMAGE_GRAYSCALE);
    image35 = imread("../../fruit_images/final challenge/C0_000010.png", CV_LOAD_IMAGE_GRAYSCALE);

    Mat currentGrayImage3 = image32.clone();
    imshow("currentGrayImage3", currentGrayImage3);
    
    Mat binarizeImage3;
    threshold(currentGrayImage3, binarizeImage3, 25, 255, THRESH_BINARY);
    imshow("binImage", binarizeImage3);
    
    Mat openedBin;
    //Operazione di Open per eliminare imperfezioni dello sfondo
    Mat structuringElement31 = getStructuringElement(MORPH_ELLIPSE, Size(20, 20));
    morphologyEx( binarizeImage3, openedBin, MORPH_OPEN, structuringElement31 );
    imshow("openedBin", openedBin);
    
    
    vector<vector<Point>>cont3;
    findContours(openedBin, cont3, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

    int iMax3 = 0;
    u_long sizeMax3 = 0;
    for(int i=0;i<cont3.size();i++){
        u_long size3 = cont3[i].size();
        if (size3>sizeMax3) {
            sizeMax3 = cont3[i].size();
            iMax3 = i;
        }
    }

    //Disegno il contorno del kiwi
    openedBin = 0;
    for (int i = 0; i < cont3[iMax3].size(); i++)
        openedBin.at<unsigned char>(cont3[iMax3][i]) = 255;
    imshow("Kiwi senza difetti", openedBin);
    
    //creo la maschera senza difetti
    floodFill(openedBin, cv::Point(0,0), Scalar(255));
    imshow("fill kiwi", openedBin);
    
    Mat inverted3 = 255 - openedBin;
   
    //Uso la maschera per eliminare lo sfondo dalle immagini
    Mat noBack3;
    currentGrayImage3.copyTo(noBack3, inverted3);
    imshow("noback kiwi", noBack3);
    
    //rimpiccolisco la maschera così poi da poterla usare per trovare solo i difetti
    erode(inverted3,inverted3,Mat(), cvPoint(-1, -1), 7);
    imshow("eroded kiwi", inverted3);
    
    Mat edge3;
    blur(noBack3, edge3, Size(3, 3));
    blur(edge3, edge3, Size(3, 3));
    
    //Cerco gli edge
    Mat edges_detected3;
    Canny(edge3, edges_detected3, 90, 115);
    imshow("edge kiwi", edges_detected3);
    
    //Trovo solo i difetti
    Mat onlyDefects3 = edges_detected3 & inverted3;
    imshow( "onlyDefects3", onlyDefects3 );
    drawContours(onlyDefects3, cont3, 0, 0, 8);
    
    //Operazione di chiusura per uniformare i contorni frastagliati trovati
    Mat structuringElement32 = getStructuringElement(MORPH_ELLIPSE, Size(40, 40));
    morphologyEx( onlyDefects3, onlyDefects3, MORPH_CLOSE, structuringElement32 );
    imshow( "onlyDefectsWithClosing", onlyDefects3 );
    
    //erodo i difetti troppo piccoli
    erode(onlyDefects3,onlyDefects3,Mat(), cvPoint(-1, -1), 2);
    imshow( "onlyDefectsWithClosingNOLittle", onlyDefects3 );
    
    
    
    //Carico le immagini a colori per mostrare il difetto se c'è
    Mat imageColor31,imageColor32,imageColor33,imageColor34,imageColor35;
    imageColor31 = imread("../../fruit_images/final challenge/C1_000006.png", CV_LOAD_IMAGE_COLOR);
    imageColor32 = imread("../../fruit_images/final challenge/C1_000007.png", CV_LOAD_IMAGE_COLOR);
    imageColor33 = imread("../../fruit_images/final challenge/C1_000008.png", CV_LOAD_IMAGE_COLOR);
    imageColor34 = imread("../../fruit_images/final challenge/C1_000009.png", CV_LOAD_IMAGE_COLOR);
    imageColor35 = imread("../../fruit_images/final challenge/C1_000010.png", CV_LOAD_IMAGE_COLOR);
    
    Mat currentColorImage3 = imageColor32.clone();
    imshow("currentColorImage3", currentColorImage3);
    
    //Mostro i difetti se ci sono
    //Cerco i Blob per poi disegnare i rettangoli sui difetti
    vector<vector<Point>>defectKiwiBlobs;
    FindBlobs(255 - onlyDefects3, defectKiwiBlobs);
    
    Mat finalColored3 = currentColorImage3.clone();
    for(int i=0;i<defectKiwiBlobs.size();i++){
        rectangle(finalColored3, boundingRect(defectKiwiBlobs[i]), Scalar{ 0,0,230 },2);
    }
    imshow( "finalColored3", finalColored3 );
    
    waitKey(0);
    
    
    return 0;
}




