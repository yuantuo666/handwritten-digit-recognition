#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/ml/ml.hpp> // Include machine learning module
#include <opencv2/core/utils/logger.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

using namespace std;
using namespace cv;

bool showWrongImg = false; // set true to pop up windows when predict wrong number
int testNum = 0; // limit predict numbers to test different k value. Set 0 to test all data.

string basePath = "E:/CUHKSZ/2023 Fall/CSC3002/mnist/";

string trainImagesName = "train-images.idx3-ubyte";
string trainLabelsName = "train-labels.idx1-ubyte";
string testImagesName = "t10k-images.idx3-ubyte";
string testLabelsName = "t10k-labels.idx1-ubyte";

int reverseInt(int i) {
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

Mat readMnistLabel(const string fileName) {
    int magicNumber;
    int numberOfItems;

    Mat labelMat;

    ifstream file(fileName, ios::binary);

    // https://stackoverflow.com/questions/8286668/how-to-read-mnist-data-in-c
    // [offset] [type] [value] [description]
    // 0000     32 bit integer  0x00000801(2049) magic number(MSB first)
    // 0004     32 bit integer  60000            number of items
    // 0008     unsigned byte ? ? label
    // 0009     unsigned byte ? ? label
    // ........
    // xxxx     unsigned byte ? ? label
    // The labels values are 0 to 9.

    if (file.is_open()) {
        cout << "> Successfully opened: " << fileName << "\n";

        file.read((char*)&magicNumber, sizeof(magicNumber));
        file.read((char*)&numberOfItems, sizeof(numberOfItems));
        magicNumber = reverseInt(magicNumber);
        numberOfItems = reverseInt(numberOfItems);

        cout << "  Magic number = " << magicNumber << "; Total number = " << numberOfItems << "\n";

        labelMat = Mat::zeros(numberOfItems, 1, CV_32SC1); // init with zero
        for (int i = 0; i < numberOfItems; i++) {
            unsigned char temp = 0;
            file.read((char*)&temp, sizeof(temp));
            labelMat.at<int>(i, 0) = temp;
            if (i % 1000 == 0) {
                // update the process with \r
                cout << "* Reading Labels... [" << i << "/" << numberOfItems << "] " << "\r";
            }
        }
        cout << "> Read Label data complete: " << numberOfItems << "\n";
    } else {
        cout << "> Fail to open file: " << fileName << "\n";
    }
    file.close();
    return labelMat;
}

Mat readMnistImage(const string fileName, const int num = 0) {
    int magicNumber = 0;
    int numberOfImages = 0;
    int numRows = 0;
    int numCols = 0;

    Mat dataMat;

    ifstream file(fileName, ios::binary);
    if (file.is_open()) {
        cout << "> Successfully opened: " << fileName << "\n";

        file.read((char*)&magicNumber, sizeof(magicNumber)); // Magic number
        file.read((char*)&numberOfImages, sizeof(numberOfImages)); // Total number of images
        file.read((char*)&numRows, sizeof(numRows)); // Number of rows per image
        file.read((char*)&numCols, sizeof(numCols)); // Number of columns per image

        magicNumber = reverseInt(magicNumber);
        numberOfImages = reverseInt(numberOfImages);
        numRows = reverseInt(numRows);
        numCols = reverseInt(numCols);

        cout << "  Magic number = " << magicNumber << "; Total number = " << numberOfImages
            << "\n  Number of rows per image = " << numRows << "; Number of columns per image = " << numCols << "\n";

        if (num != 0 && numberOfImages > num) {
            numberOfImages = num; // limit num
        }

        // CV_32FC1 => bit_depth=32  F=float  number_of_channels=1
        dataMat = Mat::zeros(numberOfImages, numRows * numCols, CV_32FC1);
        for (int i = 0; i < numberOfImages; i++) {
            for (int j = 0; j < numRows * numCols; j++) {
                unsigned char temp = 0;
                file.read((char*)&temp, sizeof(temp));
                float pixelValue = float(temp); // save in float
                dataMat.at<float>(i, j) = pixelValue;
            }

            if (i % 1000 == 0) cout << "* Reading Images... [" << i << "/" << numberOfImages << "] " << "\r";
        }

        cout << "> Read Image data complete: " << numberOfImages << "\n";
    } else {
        cout << "> Fail to open file: " << fileName << "\n";
    }
    file.close();
    return dataMat;
}

void showImg(Mat labelData, Mat imageData, int index) {
    cout << "* Showing Picture: " << index << endl;

    int pixelNum = 28 * 28;

    labelData.convertTo(labelData, CV_8UC1);
    unsigned char label = labelData.at<uchar>(index, 0);

    cout << "  label = " << int(label) << endl;

    Mat img;
    img.create(28, 28, CV_32FC1);

    for (int i = 0; i < 28; i++)
    {
        for (int j = 0; j < 28; j++) {
            img.at<float>(i, j) = imageData.at<float>(index, i * 28 + j);
        }
    }

    cv::resize(img, img, cv::Size(200, 200));

    imshow("label = " + to_string(int(label)), img);
    int k = waitKey(0); // Wait for a keystroke in the window
}

cv::Ptr<cv::ml::KNearest> train(Mat trainLabels, Mat trainImages, int k = 10) {
    cv::Ptr<cv::ml::KNearest> knn = cv::ml::KNearest::create();
    
    knn->setDefaultK(k);// default k is 10
    knn->setIsClassifier(true); // true for classification

    // Start training
    cv::Ptr<cv::ml::TrainData> trainData = cv::ml::TrainData::create(trainImages, cv::ml::ROW_SAMPLE, trainLabels);
    knn->train(trainData);
    
    return knn;
}

float test(cv::Ptr<cv::ml::KNearest> model, Mat testLabels, Mat testImages) {
    Mat preOut;

    float accuracy = model->predict(testImages, preOut);

    // convert to same data type
    preOut.convertTo(preOut, CV_8UC1);
    testLabels.convertTo(testLabels, CV_8UC1);

    int equalNums = 0;
    for (int i = 0; i < preOut.rows; i++) {
        if (preOut.at<uchar>(i, 0) == testLabels.at<uchar>(i, 0)) {
            equalNums++;
        } else if (showWrongImg) {
            cout << "  Wrong Test: label = " << int(testLabels.at<uchar>(i, 0))
                << "; predict = " << int(preOut.at<uchar>(i, 0)) << endl;
            showImg(testLabels, testImages, i);
        }
    }

    return float(equalNums) / float(preOut.rows);
}

int main() {
    // disable annoying info
    utils::logging::setLogLevel(utils::logging::LOG_LEVEL_SILENT);

    // Read training data 
    Mat trainLabels = readMnistLabel(basePath + trainLabelsName); // (60000,1)
    Mat trainImages = readMnistImage(basePath + trainImagesName); // (60000,784)
    trainImages = trainImages / 255.0; // Normalize image data

    // Read test data 
    Mat testLabels = readMnistLabel(basePath + testLabelsName); // (10000,1)
    Mat testImages = readMnistImage(basePath + testImagesName, testNum); // (10000,784)
    testImages = testImages / 255.0;

    cout << "> Data loaded..." << "\n";

    // showImg(trainLabels, trainImages, 10);
    // return 0;

    int k = 3;
  
    // for (int k = 1; k <= 10; k++) {
        cout << "* Training started..." << "\r";
        cv::Ptr<cv::ml::KNearest> model = train(trainLabels, trainImages, k);
        cout << "> Training completed " << "\n";
        cout << "* Testing started..." << "\r";
        float accuracy = test(model, testLabels, testImages);
        cout << "> Testing completed " << "\n";

        cout << "K = " << k << ". Accuracy = " << setprecision(4) << accuracy * 100.0 << "%" << "\n";
    // }
    return 0;
}