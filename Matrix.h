#ifndef CS205PROJECT_MATRIX_H
#define CS205PROJECT_MATRIX_H
#include <vector>
#include <complex>
#include <algorithm>
#include <cstdint>
#include <functional>
#include <numeric>
#include <utility>
#include <valarray>
#include <cmath>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

//judge is complex type
template<typename T>
struct isComplex {
    operator bool() {
        return false;
    }
};

template<typename T>
struct isComplex<complex<T>> {
    operator bool() {
        return true;
    }
};

template<typename T>
class Matrix {
private:
    vector<vector<T>> matrix;
public:
    //Constructor:
    Matrix() : Matrix(0, 0) {};

    Matrix(int rows, int cols);

    Matrix(const Matrix<T> &other);

    Matrix<T>(const vector<vector<T>> &matrix);

    Matrix<T>(const initializer_list<initializer_list<T>> &list);

    Matrix<T> &operator=(const Matrix<T> &other);

    static Matrix<T> zeros(int rows, int cols);

    static Matrix<T> ones(int rows, int cols);

    static Matrix<T> eye(int d);

    //get member values
    inline int getRows() const;

    inline int getCols() const;

    inline T get(int col,int row) const;

    //judge is_empty or size_is_equal
    bool is_empty() const;

    bool size_is_equal(Matrix<T> other) const;

    bool is_square() const;

    //overwrite operator:
    friend ostream &operator<<(ostream &os, const Matrix<T> &other) {
        for (const auto &i : other.matrix) {
            for (const auto &j : i) {
                os<<setprecision(4)<<((j<0.000001&&j>-0.000001)?0:j)<<"\t";
            }
            os << endl;
        }
        return os;
    }

    Matrix<T> operator+(const Matrix<T> &other) const;

    Matrix<T> operator-(const Matrix<T> &other) const;

    Matrix<T> operator*(const Matrix<T> &other) const;

    Matrix<T> operator/(T constant) const;

    template<typename T2>
    Matrix<T> operator+(T2 constant) const;

    template<typename T2>
    Matrix<T> operator-(T2 constant) const;

    Matrix<T> operator*(T constant) const;

//    template<typename T1, typename T2>
//    friend Matrix<T> operator+(T2 constant, Matrix<T> &other);

//    template<typename T1, typename T2>
//    friend Matrix<T2> operator-(T1 constant, Matrix<T2> &other);

//    template<typename T1, typename T2>
//    friend Matrix<T2> operator*(T1 constant, Matrix<T2> &other);

    //transpose
    Matrix<T> transpose() const;
    //cross_mul
    Matrix<T> cross(const Matrix<T> &other);
    //dot_mul
    Matrix<T> dot(const Matrix<T> &other) const;
    //element_wise_mul
    Matrix<T> element_wise_mul(const Matrix<T> &other) const;
    //reshape
    Matrix<T> reshape(int row, int col);
    //slicing
    Matrix<T> slicing(int row0, int row1, int col0, int col1);
    //detemination
    T det();

    Matrix<T> row_exchange(int row0, int row1);

    Matrix<T> row_transformation(int row0, int row1, double time);
    //adjiont
    Matrix<T> adj();
    //Inverse
    Matrix<T> inverse();

    //get max, min, avg for all matrix or special line
    T col_max(int col) const;
    T row_max(int row) const;
    T col_min(int col) const;
    T row_min(int row) const;
    T col_avg(int col) const;
    T row_avg(int row) const;
    T all_max() const;
    T all_min() const;
    T all_avg() const;

    void QR(Matrix<T> &Q, Matrix<T> &R);

    void Multiplicate(Matrix<T> &Q, Matrix<T> &r);

    vector<T> eigenvalue();

    Matrix<T> eigenvector();

    T trace();

    Matrix<T> Transfer_From(Mat img);

    Matrix<T> convolution(Matrix<T> &core);

    Matrix<T> conjugation();

};

template<typename T>
Matrix<T>::Matrix(int rows, int cols) {
    this->matrix = vector<vector<T>>(max(rows, 0), vector<T>(max(cols, 0)));
}

template<typename T>
Matrix<T>::Matrix(const Matrix<T> &other) {
    this->matrix = vector<vector<T>>(other.matrix);
}

template<typename T>
Matrix<T>::Matrix(const vector<vector<T>> &matrix) {
    this->vec = vector<vector<T>>(matrix);
}

template<typename T>
Matrix<T>::Matrix(const initializer_list<initializer_list<T>> &list) {
    this->matrix = vector<vector<T >>(0);
    for (auto i = list.begin(); i != list.end() - 1; i++) {
        if ((*i).size() != (*(i + 1)).size()) {
            return;
        }
    }
    this->matrix = vector<vector<T >>(list.size());
    for (auto i = list.begin(); i != list.end(); i++) {
        this->matrix[i - list.begin()] = *i;
    }
}

template<typename T>
Matrix<T> &Matrix<T>::operator=(const Matrix<T> &other) {
    if(this == &other){
        return *this;
    }
    this->matrix = vector<vector<T>>(other.matrix);
    return *this;
}

template<typename T>
int Matrix<T>::getRows() const {
    return this->matrix.size();
}

template<typename T>
int Matrix<T>::getCols() const {
    if (this->getRows() != 0){
        return this->matrix.front().size();
    }
    return 0;
}

template<typename T>
T Matrix<T>::get(int row, int col) const {
    if(col<0 || col>=getCols() || row<0 || row>=getRows()){
        throw domain_error("Element out of range of matrix!");
    }
    return this->matrix[row][col];
}

template<typename T>
bool Matrix<T>::is_empty() const {
    return (this->matrix.size() <= 0);
}

template<typename T>
bool Matrix<T>::size_is_equal(Matrix<T> other) const {
    return ((this->is_empty() && other.is_empty()) ||
            ((this->getRows() == other.getRows()) && this->getCols() == other.getCols()));
}

template<typename T>
bool Matrix<T>::is_square() const {
    return (this->getRows() == this->getCols());
}

template<typename T>
Matrix<T> Matrix<T>::operator+(const Matrix<T> &other) const {
    Matrix<T> result(this->getRows(), this->getCols());
    if (!this->size_is_equal(other)){
        throw domain_error("The Shape of two matrices is not the same!");
    }
    for (int i = 0; i < this->getRows(); i++) {
        for (int j = 0; j < this->getCols(); j++) {
            result.matrix[i][j] = this->get(i,j) + other.get(i, j);
        }
    }
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::operator-(const Matrix<T> &other) const {
    Matrix<T> result(this->getRows(), this->getCols());
    if (!this->size_is_equal(other)){
        throw domain_error("The Shape of two matrices is not the same!");
    }
    for (int i = 0; i < this->getRows(); i++) {
        for (int j = 0; j < this->getCols(); j++) {
            result.matrix[i][j] = this->get(i,j) - other.get(i, j);
        }
    }
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::operator*(const Matrix<T> &other) const {
    Matrix<T> result(this->getRows(), other.getCols());
    if (this->getCols()!=other.getRows()){
        throw domain_error("The colume of the first maxtrix"
                           " is not euqal to the row of the second one!");
    }
    for (int i = 0; i < this->getRows(); i++) {
        for (int j = 0; j < other.getCols(); j++) {
            for (int k = 0; k < this->getCols(); k++) {
                result.matrix[i][j] += this->get(i,k) * other.get(k,j);
            }
        }
    }
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::operator/(T constant) const {
    Matrix<T> result(this->getRows(), this->getCols());
    for (int i = 0; i < this->getRows(); i++) {
        for (int j = 0; j < this->getCols(); j++) {
            result.matrix[i][j] = this->get(i,j)/constant;
        }
    }
    return result;
}

template<typename T>
template<typename T2>
Matrix<T> Matrix<T>::operator+(T2 constant) const {
    Matrix<T> result(this->getRows(), this->getCols());
    for (int i = 0; i < this->getRows(); i++) {
        for (int j = 0; j < this->getCols(); j++) {
            result.matrix[i][j] = this->get(i,j) + (T)constant;
        }
    }
    return result;
}

template<typename T>
template<typename T2>
Matrix<T> Matrix<T>::operator-(T2 constant) const {
    Matrix<T> result(this->getRows(), this->getCols());
    for (int i = 0; i < this->getRows(); i++) {
        for (int j = 0; j < this->getCols(); j++) {
            result.matrix[i][j] = this->get(i,j) - (T)constant;
        }
    }
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::operator*(T constant) const {
    Matrix<T> result(this->getRows(), this->getCols());
    for (int i = 0; i < this->getRows(); i++) {
        for (int j = 0; j < this->getCols(); j++) {
            result.matrix[i][j] = this->get(i,j) * (T)constant;
        }
    }
    return result;
}

//template<typename T2,typename T>
//Matrix<T> operator+(T2 constant, Matrix<T> &other) {
//    return other+constant;
//}
//
//template<typename T1, typename T2>
//Matrix<T2> operator-(T1 constant, Matrix<T2> &other) {
//    return other*-1+constant;
//}

//template<typename T1, typename T2>
//Matrix<T2> operator*(T1 constant, Matrix<T2> &other) {
//    return other*constant;
//}


template<typename T>
Matrix<T> Matrix<T>::transpose() const {
    if (this->is_empty()) {
        return *this;
    }
    Matrix<T> result(this->getCols(), this->getRows());
    for (int i = 0; i < this->getRows(); ++i) {
        for (int j = 0; j < this->getCols(); ++j) {
            result.matrix[j][i] = this->matrix[i][j];
        }
    }
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::dot(const Matrix<T> &other) const {
    Matrix<T> result(1, getCols());
    if (!size_is_equal(other)) {
        throw domain_error("The Shape is not the same!");
    }
    T sum = 0;
    for (int i = 0; i < this->getCols(); ++i) {
        for (int j = 0; j < this->getRows(); ++j) {
            sum += this->matrix[j][i] * other.matrix[j][i];
        }
        result.matrix[0][i] = sum;
        sum = 0;
    }
    return result;
}



template<typename T>
Matrix<T> Matrix<T>::element_wise_mul(const Matrix<T> &other) const {
    Matrix<T> result(this->getRows(), this->getCols());
    if (!size_is_equal(other)) {
        throw domain_error("The Shape is not the same!");
    }
    for (int i = 0; i < this->getRows(); ++i) {
        for (int j = 0; j < this->getCols(); ++j) {
            result.matrix[i][j] = this->get(i,j) * other.get(i,j);
        }
    }
    return result;
}


template<typename T>
Matrix<T> Matrix<T>::cross(const Matrix<T> &other) {
    Matrix<T> result(3, this->getCols());
    if(!(this->size_is_equal(other) && (this->getRows()==3 && other.getRows()==3))){
        throw domain_error("The two matrices must be the same shape and degree 3!");
    }
    for(int j=0; j<this->getCols(); j++){
        result.matrix[0][j] = matrix[1][j] * other.matrix[2][j] - matrix[2][j] * other.matrix[1][j];
        result.matrix[1][j] = matrix[0][j] * other.matrix[2][j] - matrix[2][j] * other.matrix[0][j];
        result.matrix[2][j] = matrix[0][j] * other.matrix[1][j] - matrix[1][j] * other.matrix[0][j];
    }
    return result;
}


//reshape and slicing
template<typename T>
Matrix<T> Matrix<T>::reshape(int row, int col) {
    Matrix<T> result(row, col);
    int num=0;
    if (row * col != getRows() * getCols()) {
        throw domain_error("Number of elements of the"
                           " new matrix are not equal to the original matrix!");
    }
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            num = i * col + j;
            result.matrix[i][j] = this->matrix[num / getCols()][num % getCols()];
        }
    }
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::slicing(int row0, int row1, int col0, int col1) {
    if (!(row1 > row0 || row1 > getRows() ||
          row0 < 0 || col1 > col0 || col1 > getCols() || col0 < 0)) {
        throw domain_error("Out of range of matrix!");
    }
    Matrix<T> result(row1 - row0 + 1, col1 - col0 + 1);
    for (int i = row0; i <= row1; ++i) {
        for (int j = col0; j <= col1; ++j) {
            result.matrix[i - row0][j - col0] = this->matrix[i][j];
        }
    }
    return result;
}

//function used in determinant
template<typename T>
Matrix<T> Matrix<T>::row_exchange(int row0, int row1) {
    Matrix<T> result(getRows(), getCols());
    for (int i = 0; i < getRows(); ++i) {
        for (int j = 0; j < getCols(); ++j) {
            if (i == row0) {
                i = row1;
                result.matrix[row0][j] = matrix[i][j];
                i = row0;
            } else if (i == row1) {
                i = row0;
                result.matrix[row1][j] = matrix[i][j];
                i = row1;
            } else {
                result.matrix[i][j] = matrix[i][j];
            }
        }
    }
    return result;
}

//  function used in determinant
template<typename T>
Matrix<T> Matrix<T>::row_transformation(int row0, int row1, double time) {
    Matrix<T> result = *this;
    for (int i = 0; i < getCols(); ++i) {
        result.matrix[row0][i] = result.matrix[row0][i] + time * this->matrix[row1][i];
    }
    return result;
}

//  transform to gauss standardized form to calculate determinant
template<typename T>
T Matrix<T>::det() {
    T result(0);
    int row_exchange_time = 0;
    if (!is_square()) {
        throw domain_error("Not square!");
    }
    Matrix<T> temp = *this;

    for (int i = 0; i < getRows(); ++i) {
        if (temp.matrix[i][i] == 0) {
            for (int j = i + 1; j < getRows(); ++j) {
                if (temp.matrix[j][i] != 0) {
                    temp = temp.row_exchange(i, j);
                    row_exchange_time++;
                    break;
                }
            }
        }
        if (temp.matrix[i][i] == 0) {
            return (T) 0;
        }

        for (int j = i + 1; j < getRows(); ++j) {
            temp = temp.row_transformation(j, i, -1 *(temp.matrix[j][i] / (double)temp.matrix[i][i]));
        }
    }
    result = pow(-1, row_exchange_time);
    for (int i = 0; i < getRows(); ++i) {
        result *= temp.matrix[i][i];
    }
    return result;
}

//  cofactor to find adjoint matrix
template<typename T>
Matrix<T> Matrix<T>::adj() {
    Matrix<T> result (getRows(), getCols());
    Matrix<T> temp (getRows() - 1, getCols() - 1);
    for (int i = 0; i < getRows(); ++i) {
        for (int j = 0; j < getCols(); ++j) {
            for (int k = 0; k < getRows()-1; ++k) {
                for (int l = 0; l < getCols()-1; ++l) {
                    temp.matrix[k][l] = matrix[k>=i?k+1:k][l>=j?l+1:l];
                }
            }
            result.matrix[i][j] = (temp.det()==0)?0:pow(-1,i+j)*temp.det();

        }
    }
    return result.transpose();
}

//  Use adjoint matrix and determinant to calculate inverse
template<typename T>
Matrix<T> Matrix<T>::inverse() {
    if (det()==0)
        throw invalid_argument("The matrix is not invertible.");
    Matrix<T> result = this->adj();
    result = result/this->det();
    return result;
}

template<typename T>
T Matrix<T>::col_max(int col) const {
    if (col >= this->getCols() || col < 0) {
        throw domain_error("Out of range of colume!");
    }
    T result = this->matrix[0][col];
    for (int i = 0; i < this->getRows(); ++i) {
        result = std::max(result, this->matrix[i][col]);
    }
    return result;
}

template<typename T>
T Matrix<T>::row_max(int row) const {
    if (row >= this->getRows() || row < 0) {
        throw domain_error("Out of range of row!");
    }
    T result = this->matrix[row][0];
    for (int i = 0; i < this->getCols(); ++i) {
        result = std::max(result, this->matrix[row][0]);
    }
    return result;
}

template<typename T>
T Matrix<T>::col_min(int col) const {
    if (col >= this->getCols() || col < 0) {
        throw domain_error("Out of range of colume!");
    }
    T result = this->matrix[0][col];
    for (int i = 0; i < this->getRows(); ++i) {
        result = std::min(result, this->matrix[i][col]);
    }
    return result;
}

template<typename T>
T Matrix<T>::row_min(int row) const {
    if (row >= this->getRows() || row < 0) {
        throw domain_error("Out of range of row!");
    }
    T result = this->matrix[row][0];
    for (int i = 0; i < this->getCols(); ++i) {
        result = std::min(result, this->matrix[row][0]);
    }
    return result;
}

template<typename T>
T Matrix<T>::all_max() const {
    T result = this->col_max(0);
    for (int i = 0; i < this->getCols(); ++i) {
        result = std::max(result, this->col_max(i));
    }
    return result;
}

template<typename T>
T Matrix<T>::all_min() const {
    T result = this->col_min(0);
    for (int i = 0; i < this->getCols(); ++i) {
        result = std::min(result, this->col_min(i));
    }
    return result;
}

template<typename T>
T Matrix<T>::col_avg(int col) const {
    if (col >= this->getCols() || col < 0) {
        throw domain_error("Out of range of colume!");
    }
    T result = this->matrix[0][col];
    for (int i = 0; i < this->getRows(); ++i) {
        result += matrix[i][col];
    }
    return result / (T)this->getRows();
}

template<typename T>
T Matrix<T>::row_avg(int row) const {
    if (row > this->getRows() || row < 0) {
        throw domain_error("Out of range of row!");
    }
    T result = this->matrix[row - 1][0];
    for (int i = 0; i < this->getRows(); ++i) {
        result += matrix[row][i];
    }
    return result / (T)this->getCols();
}

template<typename T>
T Matrix<T>::all_avg() const {
    T result(0);
    for (int i = 0; i < this->getRows(); ++i) {
        for (int j = 0; j < this->getCols(); ++j) {
            result += this->matrix[i][j];
        }
    }
    return result / (T)(getRows() * getCols());
}

template<typename T>
Matrix<T> Matrix<T>::zeros(int rows, int cols) {
    return Matrix<T>(rows,cols);
}

template<typename T>
Matrix<T> Matrix<T>::ones(int rows, int cols) {
    Matrix<T> result(rows,cols);
    for(int i=0; i<rows; i++){
        for(int j=0; j<cols; j++){
            result.matrix[i][j]=1;
        }
    }
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::eye(int d) {
    Matrix<T> result(d,d);
    for(int i=0; i<d; i++){
        result.matrix[i][i]=1;
    }
    return result;
}

// Use QR factorization to calculate eigenvalue
template<typename T>
vector<T> Matrix<T>::eigenvalue() {
    vector<T> result;
    Matrix<T> result_matrix(getRows(), getCols());
    Matrix<T> Q(getRows(), getCols());
    Matrix<T> R(getRows(), getCols());

    // 100
    for (int i = 0; i < 100; ++i) {
        QR(Q, R);
        Multiplicate(Q, R);
    }
//    show R matrix
    for (int i = 0; i < getRows(); ++i) {
        if (R.matrix[i][i] != 0)
            if (Q.matrix[i][i]<0)R.matrix[i][i] *=-1;
        result.push_back(R.matrix[i][i]);
    }
    return result;
}

//  function used in calculate QR factoeization
template<typename T>
void Matrix<T>::QR(Matrix<T> &Q, Matrix<T> &R) {
    T temp;
    T a[Q.getRows()], b[Q.getRows()];
    for (int j = 0; j < Q.getRows(); ++j) {
        for (int i = 0; i < getRows(); ++i) {
            a[i] = this->matrix[i][j];
            b[i] = this->matrix[i][j];
        }
        for (int k = 0; k < j; ++k) {
            R.matrix[k][j] = 0;
            for (int m = 0; m < getRows(); ++m) {
                R.matrix[k][j] += a[m] * Q.matrix[m][k];
            }
            for (int m = 0; m < getRows(); ++m) {
                b[m] -= R.matrix[k][j] * Q.matrix[m][k];
            }
        }
        temp = 0;
        for (int i = 0; i < getRows(); ++i) {
            temp += b[i] * b[i];
        }
        R.matrix[j][j] = sqrt(temp);
        for (int i = 0; i < getRows(); ++i) {
            Q.matrix[i][j] = b[i] / sqrt(temp);
        }
    }
}

//  function used in calculate QR factoeization
template<typename T>
void Matrix<T>::Multiplicate(Matrix<T> &Q, Matrix<T> &R) {
    T temp = -1;
    temp += 1;
    for (int i = 0; i < getRows(); ++i) {
        for (int j = 0; j < getRows(); ++j) {
            temp = 0;
            for (int k = 0; k < getRows(); ++k) {
                temp += R.matrix[i][k] * Q.matrix[k][j];
            }
            this->matrix[i][j] = temp;
        }
    }
}

template<typename T>
Matrix<T> Matrix<T>::Transfer_From(Mat img) {
    Matrix<T> result(img.rows, img.cols);
    for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.cols; ++j) {
            result.matrix[i][j] = img.at<uchar>(i, j);
//            cout<<result.matrix[i][j]<<"\t";
        }
//        cout<<endl;
    }
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::eigenvector() {
    Matrix<T> result(getRows(),getCols());
    int i,j,q;
    int count;
    int m;
    T eValue;
    T sum, midSum, mid;
    Matrix<T> content = *this;
    vector<T> eigenvalue = this->eigenvalue();
    Matrix<T> temp(getRows(),getCols());
    cout<<"Eigenvalues are:"<<endl;
    for (int k = 0; k < eigenvalue.size(); ++k) {
        cout<<eigenvalue[k]<<" ";
    }
    cout<<"\n"<<"Eigenvectors are:"<<endl;

    for (count = 0; count < getRows(); count++) {
        temp =content;
//        cout<<temp<<endl;
        eValue = eigenvalue[count];

        for (i = 0; i < getCols(); i++) {
            temp.matrix[i][i] -= eValue;
        }
//        cout<<temp<<endl;

        for (i = 0; i < getRows()-1; i++) {
            mid = temp.matrix[i][i];
            for (j = i; j < getCols() ; j++) {
                temp.matrix[i][j] /=mid;
            }

            for(j = i+1;j<getRows();j++){
                mid =  temp.matrix[j][i];
                for(q = i;q<getCols();q++){
                    temp.matrix[j][q] -= mid*temp.matrix[i][q];
                }
            }
        }

        midSum = result.matrix[getRows()-1][count] = 1;
        for(m = getRows()-2;m>=0;--m){
            sum = 0;
            for(j = m+1;j<getCols();j++){
                sum+=temp.matrix[m][j]*result.matrix[j][count];
            }
            sum = -sum/temp.matrix[m][m];
            midSum += sum*sum;
            result.matrix[m][count] = sum;
        }

        midSum = sqrt(midSum);
        for (i = 0; i < result.getRows() ; i++) {
            result.matrix[i][count] /=midSum;
        }
    }
    return result;
}

template<typename T>
Mat Transfer_To(Matrix<T> other) {
    Mat result(other.getRows(), other.getCols(), CV_8UC1);
//    cout<<result.rows<<" "<<result.cols<<endl;
    for (int i = 0; i < result.rows; ++i) {
        for (int j = 0; j < result.cols; ++j) {
            result.at<uchar>(i, j) = (uchar) other.get(i, j);
        }
    }
    return result;
}

template<typename T>
T Matrix<T>::trace() {
    if (!is_square())
        throw invalid_argument("The matrix is not square.");
    T result = matrix[0][0];
    for (int i = 1; i < getRows(); ++i) {
        result += matrix[i][i];
    }
    return result;
}

//only for odd core
//return the Matrix after convolution without changing itself
template<typename T>
Matrix<T> Matrix<T>::convolution(Matrix<T> &core) {
    T temp;
    //rotate the core by 180 degree
    int cr = core.getRows();
    int cc = core.getCols();
    for (int i = 0; i < cr / 2; ++i) {
        for (int j = 0; j < cc; ++j) {
            temp = core.matrix[i][j];
            core.matrix[i][j] = core.matrix[cr - 1 - i][cc - 1 - j];
            core.matrix[cr - 1 - i][cc - 1 - j] = temp;
        }
    }
    for (int j = 0; j < cc / 2; ++j) {
        temp = core.matrix[cr / 2][j];
        core.matrix[cr / 2][j] = core.matrix[cr - 1 - cr / 2][cc - 1 - j];
        core.matrix[cr - 1 - cr / 2][cc - 1 - j] = temp;
    }
    cout << core;


    //multiply every element of this matrix with the core matrix
    Matrix<T> result(this->getRows(), this->getCols());

    for (int i = 0; i < getRows(); ++i) {
        for (int j = 0; j < getCols(); ++j) {
            temp = 0;
            for (int k = 0; k < cr; ++k) {
                for (int l = 0; l < cc; ++l) {
                    if (i - cr / 2 + k >= 0     &&
                        j - cc / 2 + l >= 0     &&
                        i - cr / 2 + k < getRows() &&
                        j - cc / 2 + l < getCols()       ) {

                        temp += matrix[i - cr / 2 + k][j - cc / 2 + l] * core.matrix[k][l];
                    }
                }
            }
            result.matrix[i][j] = temp;
        }
    }
    return result;
}
template<typename T>
Matrix<T> Matrix<T>::conjugation(){
    Matrix<T> result(getRows(), getCols());
    for(int i=0; i<getRows(); i++){
        for(int j=0; j<getCols(); j++){
            result.matrix[i][j]=conj(this->get(i,j));
        }
    }
    return result;
}



#endif //CS205PROJECT_MATRIX_H