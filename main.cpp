#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "Matrix.h"
#include <complex>

using namespace cv;
using namespace std;

int main(int argc, char *argv[]) {

//    Matrix<double> EigenTest2{{1, 1, 1},
//                              {1, 1, 1},
//                              {1, 2, 2}};

    Matrix<double> EigenTest2{{1, 2,  3, 2},
                              {5, -2, 4, 6},
                              {6, 2,  5, 5},
                              {4, 3,  2, 1}};


//    cout << EigenTest2.reshape(2, 8);
//    cout << EigenTest2.slicing(0, 2, 0, 2);

    cout << "Determinant: " << EigenTest2.det() << endl;
    cout << "Inverse: \n" << EigenTest2.inverse() << endl;
    cout << "Trace: " << EigenTest2.trace() << endl;
    cout << EigenTest2.eigenvector() << endl;





//  opencv & Matrix<int>
    //  opencv & Matrix<int>
    Mat mat = imread(R"(D:\Uni File\2021Spring\C&Cpp Programing\sustech.jpg)");
    cvtColor(mat, mat, COLOR_BGR2GRAY);

//    cout << mat << endl;

    imshow("test1", mat);
    waitKey(0);

    Matrix<int> temp(mat.rows, mat.cols);
    temp = temp.Transfer_From(mat);

//    cout << temp << endl;

//  deal with the image
    temp = temp.transpose();
    temp = temp * 2;

    Mat mat2;
    mat2 = Transfer_To(temp);
    imshow("test2", mat2);
    waitKey(0);

    return 0;
}
