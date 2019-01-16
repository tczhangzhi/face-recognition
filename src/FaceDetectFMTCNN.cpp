#include "FaceDetectFMTCNN.hpp"

#define IMGNORHEIGHT 112
#define IMGNORWIDTH 96

CFaceDetectFMTCNN::CFaceDetectFMTCNN() {
    PNet_ = dnn::readNetFromCaffe(proto_model_dir + "/det1_.prototxt", proto_model_dir + "/det1_.caffemodel");
    RNet_ = dnn::readNetFromCaffe(proto_model_dir + "/det2.prototxt", proto_model_dir + "/det2_half.caffemodel");
    ONet_ = dnn::readNetFromCaffe(proto_model_dir + "/det3-half.prototxt", proto_model_dir + "/det3-half.caffemodel");
}


float CFaceDetectFMTCNN::IoU(float xmin, float ymin, float xmax, float ymax,
                 float xmin_, float ymin_, float xmax_, float ymax_, bool is_iom) {
    float iw = min(xmax, xmax_) - max(xmin, xmin_) + 1;
    float ih = min(ymax, ymax_) - max(ymin, ymin_) + 1;
    if (iw <= 0 || ih <= 0)
        return 0;
    float s = iw*ih;
    if (is_iom) {
        float ov = s / min((xmax - xmin + 1)*(ymax - ymin + 1), (xmax_ - xmin_ + 1)*(ymax_ - ymin_ + 1));
        return ov;
    }
    else {
        float ov = s / ((xmax - xmin + 1)*(ymax - ymin + 1) + (xmax_ - xmin_ + 1)*(ymax_ - ymin_ + 1) - s);
        return ov;
    }
}
void CFaceDetectFMTCNN::BBoxRegression(vector<FaceInfo>& bboxes) {
    for (int i = 0; i < bboxes.size(); ++i) {
        FaceBox &bbox = bboxes[i].bbox;
        float *bbox_reg = bboxes[i].bbox_reg;
        float w = bbox.xmax - bbox.xmin + 1;
        float h = bbox.ymax - bbox.ymin + 1;
        bbox.xmin += bbox_reg[0] * w;
        bbox.ymin += bbox_reg[1] * h;
        bbox.xmax += bbox_reg[2] * w;
        bbox.ymax += bbox_reg[3] * h;
    }
}
void CFaceDetectFMTCNN::BBoxPad(vector<FaceInfo>& bboxes, int width, int height) {
    for (int i = 0; i < bboxes.size(); ++i) {
        FaceBox &bbox = bboxes[i].bbox;
        bbox.xmin = round(max(bbox.xmin, 0.f));
        bbox.ymin = round(max(bbox.ymin, 0.f));
        bbox.xmax = round(min(bbox.xmax, width - 1.f));
        bbox.ymax = round(min(bbox.ymax, height - 1.f));
    }
}
void CFaceDetectFMTCNN::BBoxPadSquare(vector<FaceInfo>& bboxes, int width, int height) {
    for (int i = 0; i < bboxes.size(); ++i) {
        FaceBox &bbox = bboxes[i].bbox;
        float w = bbox.xmax - bbox.xmin + 1;
        float h = bbox.ymax - bbox.ymin + 1;
        float side = h>w ? h : w;
        bbox.xmin = round(max(bbox.xmin + (w - side)*0.5f, 0.f));

        bbox.ymin = round(max(bbox.ymin + (h - side)*0.5f, 0.f));
        bbox.xmax = round(min(bbox.xmin + side - 1, width - 1.f));
        bbox.ymax = round(min(bbox.ymin + side - 1, height - 1.f));
    }
}
void CFaceDetectFMTCNN::GenerateBBox(Mat* confidence, Mat* reg_box,
                         float scale, float thresh) {
    int feature_map_w_ = confidence->size[3];
    int feature_map_h_ = confidence->size[2];
    int spatical_size = feature_map_w_*feature_map_h_;

    cout<<confidence->size;
    cout<<" "<<scale<<endl;

    const float* confidence_data = (float*)(confidence->data);
    confidence_data += spatical_size;

    Mat image(feature_map_h_,feature_map_w_,confidence->type());

    image.data =(unsigned  char*)(confidence_data);

    const float* reg_data = (float*)(reg_box->data);
    candidate_boxes_.clear();
    for (int i = 0; i<spatical_size; i++) {
        if (confidence_data[i] <= 1-thresh) {

            int y = i / feature_map_w_;
            int x = i - feature_map_w_ * y;
            FaceInfo faceInfo;
            FaceBox &faceBox = faceInfo.bbox;

            faceBox.xmin = (float)(x * pnet_stride) / scale;
            faceBox.ymin = (float)(y * pnet_stride) / scale;
            faceBox.xmax = (float)(x * pnet_stride + pnet_cell_size - 1.f) / scale;
            faceBox.ymax = (float)(y * pnet_stride + pnet_cell_size - 1.f) / scale;
            faceInfo.bbox_reg[0] = reg_data[i];
            faceInfo.bbox_reg[1] = reg_data[i + spatical_size];
            faceInfo.bbox_reg[2] = reg_data[i + 2 * spatical_size];
            faceInfo.bbox_reg[3] = reg_data[i + 3 * spatical_size];
            faceBox.score = confidence_data[i];
            candidate_boxes_.push_back(faceInfo);
        }
    }
}

bool CompareBBox(const FaceInfo & a, const FaceInfo & b) {
    return a.bbox.score > b.bbox.score;
}

vector<FaceInfo> CFaceDetectFMTCNN::NMS(vector<FaceInfo>& bboxes,
                                 float thresh, char methodType) {
    vector<FaceInfo> bboxes_nms;
    if (bboxes.size() == 0) {
        return bboxes_nms;
    }
    sort(bboxes.begin(), bboxes.end(), CompareBBox);

    int32_t select_idx = 0;
    int32_t num_bbox = static_cast<int32_t>(bboxes.size());
    vector<int32_t> mask_merged(num_bbox, 0);
    bool all_merged = false;

    while (!all_merged) {
        while (select_idx < num_bbox && mask_merged[select_idx] == 1)
            select_idx++;
        if (select_idx == num_bbox) {
            all_merged = true;
            continue;
        }

        bboxes_nms.push_back(bboxes[select_idx]);
        mask_merged[select_idx] = 1;

        FaceBox select_bbox = bboxes[select_idx].bbox;
        float area1 = static_cast<float>((select_bbox.xmax - select_bbox.xmin + 1) * (select_bbox.ymax - select_bbox.ymin + 1));
        float x1 = static_cast<float>(select_bbox.xmin);
        float y1 = static_cast<float>(select_bbox.ymin);
        float x2 = static_cast<float>(select_bbox.xmax);
        float y2 = static_cast<float>(select_bbox.ymax);

        select_idx++;
        for (int32_t i = select_idx; i < num_bbox; i++) {
            if (mask_merged[i] == 1)
                continue;

            FaceBox & bbox_i = bboxes[i].bbox;
            float x = max<float>(x1, static_cast<float>(bbox_i.xmin));
            float y = max<float>(y1, static_cast<float>(bbox_i.ymin));
            float w = min<float>(x2, static_cast<float>(bbox_i.xmax)) - x + 1;
            float h = min<float>(y2, static_cast<float>(bbox_i.ymax)) - y + 1;
            if (w <= 0 || h <= 0)
                continue;

            float area2 = static_cast<float>((bbox_i.xmax - bbox_i.xmin + 1) * (bbox_i.ymax - bbox_i.ymin + 1));
            float area_intersect = w * h;

            switch (methodType) {
                case 'u':
                    if (static_cast<float>(area_intersect) / (area1 + area2 - area_intersect) > thresh)
                        mask_merged[i] = 1;
                    break;
                case 'm':
                    if (static_cast<float>(area_intersect) / min(area1, area2) > thresh)
                        mask_merged[i] = 1;
                    break;
                default:
                    break;
            }
        }
    }
    return bboxes_nms;
}

vector<FaceInfo> CFaceDetectFMTCNN::NextStage(const Mat& image, vector<FaceInfo> &pre_stage_res, int input_w, int input_h, int stage_num, const float threshold) {
    vector<FaceInfo> res;
    int batch_size = (int)pre_stage_res.size();
    if (batch_size == 0)
        return res;
    Mat* input_layer = nullptr;
    Mat* confidence = nullptr;
    Mat* reg_box = nullptr;
    Mat* reg_landmark = nullptr;

    vector< Mat > targets_blobs;



    switch (stage_num) {
        case 2: {
        }break;
        case 3: {
        }break;
        default:
            return res;
            break;
    }
    int spatial_size = input_h*input_w;

    vector<Mat> inputs;

    for (int n = 0; n < batch_size; ++n) {
        FaceBox &box = pre_stage_res[n].bbox;
        Mat roi = image(Rect(Point((int)box.xmin, (int)box.ymin), Point((int)box.xmax, (int)box.ymax))).clone();
        resize(roi, roi, Size(input_w, input_h));
        inputs.push_back(roi);
    }

    Mat blob_input = dnn::blobFromImages(inputs, std_val,Size(),Scalar(mean_val,mean_val,mean_val),false);

    switch (stage_num) {
        case 2: {
            RNet_.setInput(blob_input, "data");
            const vector< String >  targets_node{"conv5-2","prob1"};
            RNet_.forward(targets_blobs,targets_node);
            confidence = &targets_blobs[1];
            reg_box = &targets_blobs[0];

            float* confidence_data = (float*)confidence->data;
        }break;
        case 3: {

            ONet_.setInput(blob_input, "data");
            const vector< String >  targets_node{"conv6-2","conv6-3","prob1"};
            ONet_.forward(targets_blobs,targets_node);
            reg_box = &targets_blobs[0];
            reg_landmark = &targets_blobs[1];
            confidence = &targets_blobs[2];

        }break;
    }


    const float* confidence_data = (float*)confidence->data;

    const float* reg_data = (float*)reg_box->data;
    const float* landmark_data = nullptr;
    if (reg_landmark) {
        landmark_data = (float*)reg_landmark->data;
    }
    for (int k = 0; k < batch_size; ++k) {
        if (confidence_data[2 * k + 1] >= threshold) {
            FaceInfo info;
            info.bbox.score = confidence_data[2 * k + 1];
            info.bbox.xmin = pre_stage_res[k].bbox.xmin;
            info.bbox.ymin = pre_stage_res[k].bbox.ymin;
            info.bbox.xmax = pre_stage_res[k].bbox.xmax;
            info.bbox.ymax = pre_stage_res[k].bbox.ymax;
            for (int i = 0; i < 4; ++i) {
                info.bbox_reg[i] = reg_data[4 * k + i];
            }
            if (reg_landmark) {
                float w = info.bbox.xmax - info.bbox.xmin + 1.f;
                float h = info.bbox.ymax - info.bbox.ymin + 1.f;
                for (int i = 0; i < 5; ++i){
                    info.landmark[2 * i] = landmark_data[10 * k + 2 * i] * w + info.bbox.xmin;
                    info.landmark[2 * i + 1] = landmark_data[10 * k + 2 * i + 1] * h + info.bbox.ymin;
                }
            }
            res.push_back(info);
        }
    }
    return res;
}

vector<FaceInfo> CFaceDetectFMTCNN::ProposalNet(const Mat& img, int minSize, float threshold, float factor) {
    Mat  resized;
    int width = img.cols;
    int height = img.rows;
    float scale = 12.f / minSize;
    float minWH = min(height, width) *scale;
    vector<float> scales;
    while (minWH >= 12) {
        scales.push_back(scale);
        minWH *= factor;
        scale *= factor;
    }

    total_boxes_.clear();
    for (int i = 0; i < scales.size(); i++) {
        int ws = (int)ceil(width*scales[i]);
        int hs = (int)ceil(height*scales[i]);
        resize(img, resized, Size(ws, hs), 0, 0, INTER_LINEAR);


        Mat inputBlob = dnn::blobFromImage(resized, 1/255.0,Size(),Scalar(0,0,0),false);

        float* c = (float*)inputBlob.data;
        PNet_.setInput(inputBlob, "data");
        const vector< String >  targets_node{"conv4-2","prob1"};
        vector< Mat > targets_blobs;
        PNet_.forward(targets_blobs,targets_node);

        Mat prob = targets_blobs[1]
        ;
        Mat reg = targets_blobs[0];
        GenerateBBox(&prob, &reg, scales[i], threshold);
        vector<FaceInfo> bboxes_nms = NMS(candidate_boxes_, 0.5, 'u');
        if (bboxes_nms.size()>0) {
            total_boxes_.insert(total_boxes_.end(), bboxes_nms.begin(), bboxes_nms.end());
        }
    }
    int num_box = (int)total_boxes_.size();

    vector<FaceInfo> res_boxes;
    if (num_box != 0) {
        res_boxes = NMS(total_boxes_, 0.7f, 'u');
        BBoxRegression(res_boxes);
        BBoxPadSquare(res_boxes, width, height);
    }
    return res_boxes;
}

vector<FaceInfo> CFaceDetectFMTCNN::DetectMTCNN(const Mat& image, const int minSize, const float* threshold, const float factor, const int stage) {
    vector<FaceInfo> pnet_res;
    vector<FaceInfo> rnet_res;
    vector<FaceInfo> onet_res;
    if (stage >= 1){
        pnet_res = ProposalNet(image, minSize, threshold[0], factor);
    }
    if (stage >= 2 && pnet_res.size()>0){
        if (pnet_max_detect_num < (int)pnet_res.size()){
            pnet_res.resize(pnet_max_detect_num);
        }
        int num = (int)pnet_res.size();
        int size = (int)ceil(1.f*num / step_size);
        for (int iter = 0; iter < size; ++iter){
            int start = iter*step_size;
            int end = min(start + step_size, num);
            vector<FaceInfo> input(pnet_res.begin() + start, pnet_res.begin() + end);
            vector<FaceInfo> res = NextStage(image, input, 24, 24, 2, threshold[1]);
            rnet_res.insert(rnet_res.end(), res.begin(), res.end());
        }
        rnet_res = NMS(rnet_res, 0.4f, 'm');
        BBoxRegression(rnet_res);
        BBoxPadSquare(rnet_res, image.cols, image.rows);

    }
    if (stage >= 3 && rnet_res.size()>0){
        int num = (int)rnet_res.size();
        int size = (int)ceil(1.f*num / step_size);
        for (int iter = 0; iter < size; ++iter){
            int start = iter*step_size;
            int end = min(start + step_size, num);
            vector<FaceInfo> input(rnet_res.begin() + start, rnet_res.begin() + end);
            vector<FaceInfo> res = NextStage(image, input, 48, 48, 3, threshold[2]);
            onet_res.insert(onet_res.end(), res.begin(), res.end());
        }
        BBoxRegression(onet_res);
        onet_res = NMS(onet_res, 0.4f, 'm');
        BBoxPad(onet_res, image.cols, image.rows);

    }
    if (stage == 1){
        return pnet_res;
    }
    else if (stage == 2){
        return rnet_res;
    }
    else if (stage == 3){
        return onet_res;
    }
    else{
        return onet_res;
    }
}

void CFaceDetectFMTCNN::RotateFace(Mat image, FaceInfo faceInfoVar, Mat &dstImage)
{
    float std_points[10] = {30.2946, 65.5318, 48.0252, 33.5493, 62.7299, 51.6963, 51.5014, 71.7366, 92.3655, 92.2041};

    float facial_points[10];
    for (int i = 0; i < 5; i++) {
        facial_points[2 * i] = faceInfoVar.landmark[2 * i];
        facial_points[2 * i + 1] = faceInfoVar.landmark[2 * i + 1];
    }
    Mat tform = getTformMatrix(std_points, facial_points);

    warpAffine(image, dstImage, tform, dstImage.size(), 1, 0, Scalar(0));
}

void CFaceDetectFMTCNN::NormalizeFace(Mat dstImage, Mat &normalizeImg)
{
    Mat subfactor = 127.5 * Mat(IMGNORHEIGHT, IMGNORWIDTH, CV_32FC3, Scalar(1, 1, 1));
    dstImage.convertTo(normalizeImg, CV_32FC3);
    normalizeImg = normalizeImg - subfactor;
    normalizeImg = normalizeImg / 128;
}

Mat CFaceDetectFMTCNN::getTformMatrix(float* std_points, float* feat_points)
{
    int points_num_ = 5;
	double sum_x = 0, sum_y = 0;
	double sum_u = 0, sum_v = 0;
	double sum_xx_yy = 0;
	double sum_ux_vy = 0;
	double sum_vx__uy = 0;
	for (int c = 0; c < points_num_; ++c) {
		int x_off = c * 2;
		int y_off = x_off + 1;
		sum_x += std_points[c * 2];
		sum_y += std_points[c * 2 + 1];
		sum_u += feat_points[x_off];
		sum_v += feat_points[y_off];
		sum_xx_yy += std_points[c * 2] * std_points[c * 2] +
			std_points[c * 2 + 1] * std_points[c * 2 + 1];
		sum_ux_vy += std_points[c * 2] * feat_points[x_off] +
			std_points[c * 2 + 1] * feat_points[y_off];
		sum_vx__uy += feat_points[y_off] * std_points[c * 2] -
			feat_points[x_off] * std_points[c * 2 + 1];
	}
	double q = sum_u - sum_x * sum_ux_vy / sum_xx_yy
		+ sum_y * sum_vx__uy / sum_xx_yy;
	double p = sum_v - sum_y * sum_ux_vy / sum_xx_yy
		- sum_x * sum_vx__uy / sum_xx_yy;
	double r = points_num_ - (sum_x * sum_x + sum_y * sum_y) / sum_xx_yy;
	double a = (sum_ux_vy - sum_x * q / r - sum_y * p / r) / sum_xx_yy;
	double b = (sum_vx__uy + sum_y * q / r - sum_x * p / r) / sum_xx_yy;
	double c = q / r;
	double d = p / r;


	Mat Tinv = (cv::Mat_<float>(3, 3) << a, b, 0, -b, a, 0, c, d, 1);
	Mat T = Tinv.inv();
	Mat res = T.colRange(0, 2).clone();
	return res.t();
}

void CFaceDetectFMTCNN::Detect(Mat image, vector<FaceInfo> &faceInfo)
{
    Mat copyImage;
    image.copyTo(copyImage);

    faceInfo = DetectMTCNN(image, minSize, threshold, factor, 3);
    
    for (int i = 0; i < faceInfo.size(); i++) {
        int x = (int) faceInfo[i].bbox.xmin;
        int y = (int) faceInfo[i].bbox.ymin;
        int w = (int) (faceInfo[i].bbox.xmax - faceInfo[i].bbox.xmin + 1);
        int h = (int) (faceInfo[i].bbox.ymax - faceInfo[i].bbox.ymin + 1);
        rectangle(copyImage, Rect(x, y, w, h), Scalar(255, 0, 0), 2);
    }

    imshow("人脸检测结果", copyImage);
    waitKey(0);
}

CFaceDetectFMTCNN::~CFaceDetectFMTCNN(void)
{
}