#include<iostream>
#include <eigen3/Eigen/Eigen>

int main(){
    Eigen::MatrixXd x1(3,4);
    x1 << 1,2,3,4,
          5,6,7,8,
          3,22,6,5;
    Eigen::MatrixXd x2(3,1);
    x2 << 1,2,3;

    Eigen::Vector3d x4;
    x4 << 1,2,3;

    std::cout <<x1<<std::endl;
    std::cout <<x2<<std::endl;
    std::cout << (x2/x2.sum());
    Eigen::MatrixXd x3 = (x1.colwise() - x1.rowwise().maxCoeff()).array().colwise()/x1.array().rowwise().sum();
    std::cout << x3 << std::endl;
    return 0;
}