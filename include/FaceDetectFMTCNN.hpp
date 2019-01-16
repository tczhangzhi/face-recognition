#ifndef FaceDetectFMTCNN_hpp
#define FaceDetectFMTCNN_hpp

#include <vector>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

typedef struct FaceBox {
    float xmin;
    float ymin;
    float xmax;
    float ymax;
    float score;
} FaceBox;

typedef struct FaceInfo {
    float bbox_reg[4];
    float landmark_reg[10];
    float landmark[10];
    FaceBox bbox;
    Mat FeatureVector;
} FaceInfo;

class CFaceDetectFMTCNN {
public:
    dnn::Net PNet_;
    dnn::Net RNet_;
    dnn::Net ONet_;

    vector<FaceInfo> candidate_boxes_;
    vector<FaceInfo> total_boxes_;

    float factor = 0.709f;
    float threshold[3] = { 0.7f, 0.6f, 0.6f };
    int minSize = 12;

    CFaceDetectFMTCNN(void);
    ~CFaceDetectFMTCNN(void);
    Mat getTformMatrix(float* std_points, float* feat_points);
    vector<FaceInfo> DetectMTCNN(const Mat& image, const int minSize, const float* threshold, const float factor, const int stage);
    void RotateFace(Mat image, FaceInfo faceInfoVar, Mat &dstImage);
    void NormalizeFace(Mat dstImage, Mat &normalizeImg);
    vector<FaceInfo> ProposalNet(const Mat& img, int min_size, float threshold, float factor);
    vector<FaceInfo> NextStage(const Mat& image, vector<FaceInfo> &pre_stage_res, int input_w, int input_h, int stage_num, const float threshold);
    void BBoxRegression(vector<FaceInfo>& bboxes);
    void BBoxPadSquare(vector<FaceInfo>& bboxes, int width, int height);
    void BBoxPad(vector<FaceInfo>& bboxes, int width, int height);
    void GenerateBBox(Mat* confidence, Mat* reg_box, float scale, float thresh);
    vector<FaceInfo> NMS(vector<FaceInfo>& bboxes, float thresh, char methodType);
    float IoU(float xmin, float ymin, float xmax, float ymax, float xmin_, float ymin_, float xmax_, float ymax_, bool is_iom = false);
    void Detect(Mat image, vector<FaceInfo> &faceInfo);
    void DisplayBoxMarks(Mat image, vector<FaceInfo> FaceInfo);


private:
    const float pnet_stride = 2;
    const float pnet_cell_size = 12;
    const int pnet_max_detect_num = 5000;
    const float mean_val = 127.5f;
    const float std_val = 0.0078125f;
    const int step_size = 128;
    const string proto_model_dir = "static/model";
};

#endif /* FaceDetectFMTCNN_hpp */