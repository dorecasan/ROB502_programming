#include <fstream>
#include <iomanip>
#include <iostream>
#include <eigen3/Eigen/Eigen>

Eigen::Matrix3d transform_mat(double q, double l)
{
    // --- Your code here
    Eigen::Matrix3d transformation_matrix;
    transformation_matrix << std::cos(q), -std::sin(q), l*std::cos(q),
                             std::sin(q), std::cos(q), l*std::sin(q),
                             0,0,1;
    return transformation_matrix;
    // ---
}

int main(int argc, char* argv[])
{
    // by defult, use the file joint_angles.txt, but if you supply a command line argument, use that instead
    std::string input_filename{"joint_angles.txt"};
    if (argc >= 2) {
        input_filename = argv[1];
    }
    std::ifstream ifs(input_filename);

    if (!ifs.good()) {
        std::cerr << "Failed to open file " << input_filename << std::endl;
        return 0;
    }

    double l1{1.0};
    double l2{0.5};
    double l3{0.25};

    while (ifs.peek() != -1)
    {
        double q1, q2, q3;
        ifs >> q1 >> q2 >> q3;

        // --- Your code here
        Eigen::Matrix3d T1 = transform_mat(q1,l1);
        Eigen::Matrix3d T2 = transform_mat(q2,l2);
        Eigen::Matrix3d T3 = transform_mat(q3,l3);

        Eigen::Matrix3d T = T1*T2*T3;

        std::cout << "Transformation matrix for q1 = "<<q1<<", q2 = " <<q2<<", q3 = "<<q3<<":\n";
        std::cout << std::setprecision(3) << T <<std::endl; 
        // ---
    }
    return 0;
}