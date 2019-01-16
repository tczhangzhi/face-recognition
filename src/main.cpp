#include<opencv2/opencv.hpp>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include <opencv2/dnn.hpp>

using namespace std;
using namespace cv;
using namespace dnn;

#include "FaceDetectFMTCNN.hpp"

#define IMGNORHEIGHT 112
#define IMGNORWIDTH 96

double CosineSimilarity(Mat& a, Mat& b)
{
    double ab = a.dot(b);
    double norma=norm(a);
    double normb=norm(b); 
    if(norma!=0 && normb!=0){
        return ab / (norma * normb);
    }
    return -1;
}

Mat CaputreFeature(Net net)
{
    CFaceDetectFMTCNN detector;
    vector<FaceInfo> faceInfo;

    cout << "捕获摄像头图像：" << endl;
    VideoCapture capture(0);
    Mat frame;
    capture >> frame;
    imshow("摄像头捕捉结果", frame);
    detector.Detect(frame, faceInfo);

    for (int i = 0; i < faceInfo.size(); i++) {
        FaceInfo faceInfoVar = faceInfo[i];
        Mat dstImage(IMGNORHEIGHT, IMGNORWIDTH, CV_8UC3);
        detector.RotateFace(frame, faceInfoVar, dstImage);

        Mat normalizeImage;
        normalizeImage.create(dstImage.size(), CV_32FC3);
        detector.NormalizeFace(dstImage, normalizeImage);
        
        Mat inputBlob = blobFromImage(normalizeImage);
        Mat featureOutput;
        net.setInput(inputBlob, "data");
        featureOutput = net.forward("fc5");

        Mat result(IMGNORHEIGHT, IMGNORWIDTH, CV_8UC3);
        featureOutput.copyTo(result);
        return result;
    }

    cout << "出错啦" << endl;
}

int main(int argc, char* argv[])
{
    Net net;
    
    string proto_model_dir = "static/model";
    string modelTxt = proto_model_dir + "/sphereface_deploy.prototxt";
    string modelBin = proto_model_dir + "/sphereface_model.caffemodel";

    net = dnn::readNetFromCaffe(modelTxt, modelBin);
    if (net.empty()) {
        cerr << "无法加载模型" << endl;
        exit(-1);
    }

    String imageFile = "static/test.jpeg";
    Mat image = imread(imageFile, 1);
    if (image.empty()) {
        cerr << "无法加载图像" << endl;
        exit(-1);
    }

    Mat captureFeature = CaputreFeature(net);
    cout << captureFeature << endl;

    CFaceDetectFMTCNN detector;
    vector<FaceInfo> faceInfo;

    cout << "检测所有人脸：" << endl;
    detector.Detect(image, faceInfo);

    Mat dstImage(IMGNORHEIGHT, IMGNORWIDTH, CV_8UC3);

    for (int i = 0; i < faceInfo.size(); i++) {
        cout << "匹配照片中的图像是否有与该人脸相似的：" << endl;
        FaceInfo faceInfoVar = faceInfo[i];
        detector.RotateFace(image, faceInfoVar, dstImage);

        Mat normalizeImage;
        normalizeImage.create(dstImage.size(), CV_32FC3);
        detector.NormalizeFace(dstImage, normalizeImage);

        Mat inputBlob = blobFromImage(normalizeImage);
        Mat featureOutput;
        net.setInput(inputBlob, "data");
        featureOutput = net.forward("fc5");
        featureOutput.copyTo(faceInfo[i].FeatureVector);
        float similarity = CosineSimilarity(featureOutput, captureFeature);
        cout << similarity << endl;
        if (similarity > 0.8) {
            int x = (int) faceInfo[i].bbox.xmin;
            int y = (int) faceInfo[i].bbox.ymin;
            int w = (int) (faceInfo[i].bbox.xmax - faceInfo[i].bbox.xmin + 1);
            int h = (int) (faceInfo[i].bbox.ymax - faceInfo[i].bbox.ymin + 1);
            rectangle(image, Rect(x, y, w, h), Scalar(0, 0, 255), 2);
            imshow("匹配结果", image);
            waitKey(0);
        }
    }

    return 0;
}
