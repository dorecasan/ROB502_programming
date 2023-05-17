#include <chrono>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <eigen3/Eigen/Eigen>

Eigen::MatrixXd csv2mat(std::ifstream &ifs)
{
    if (!ifs.good())
    {
        throw std::runtime_error("failed to open file!");
    }

    int rows, cols;
    ifs >> rows;
    ifs >> cols;
    Eigen::MatrixXd mat(rows, cols);

    int row = 0;
    int col = 0;
    while (ifs.peek() != ifs.eof())
    {
        double x;
        ifs >> x;
        mat(row, col) = x;
        ++col;
        if (col == cols)
        {
            col = 0;
            ++row;
        }
        if (row == rows)
        {
            break;
        }
    }
    return mat;
}

class Layer
{
public:
    Layer() {};

    virtual Eigen::MatrixXd forward(Eigen::MatrixXd const &x) const = 0;
};

class Linear : public Layer
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
public:
    Linear(std::string const &A_filename, std::string const &b_filename)
    {
        std::ifstream A_file(A_filename);
        std::ifstream b_file(b_filename);
        A = csv2mat(A_file);
        b = csv2mat(b_file);
    }

    Eigen::MatrixXd forward(Eigen::MatrixXd const &x) const override
    {
        // --- Your code here

        Eigen::MatrixXd out = A*x + b;
        return out;

        // ---
    };

private:
    Eigen::MatrixXd A;
    Eigen::MatrixXd b;
};

// Create and implement the ReLU and Softmax classes here
// --- Your code here
class ReLU: public Layer{
public:
    Eigen::MatrixXd forward(const Eigen::MatrixXd &x) const override{
        return x.cwiseMax(0);
    }
};

class Softmax: public Layer{
public:
    Eigen::MatrixXd forward(const Eigen::MatrixXd &x) const override{
        Eigen::MatrixXd x_exp = x.array().exp();
        if(x.cols() == 1){
            x_exp /= x_exp.sum();
        }
        else{
            x_exp = (x_exp.colwise()-x_exp.rowwise().maxCoeff()).array().colwise()/x_exp.array().rowwise().sum();
        }
        return x_exp; 
    }
};

// ---

int main(int argc, char* argv[])
{
    std::string prefix_path{"../network/"};
    const Eigen::IOFormat vec_csv_format(3, Eigen::DontAlignCols, ", ", ", ");
    std::ofstream ofs(prefix_path+"output.csv");

    // load in the weights, biases, and the data from files
    std::vector<std::string> data_filenames{prefix_path+"data1.csv", prefix_path+"data2.csv", prefix_path+"data3.csv", prefix_path+"data4.csv"};
    if (argc >= 2) {
        data_filenames.clear();
        for (int i{1}; i < argc; ++i) {
            data_filenames.push_back(argv[i]);
        }
    }

    Linear l1(prefix_path+"A1.csv", prefix_path+"b1.csv");
    ReLU r;
    Linear l2(prefix_path+"A2.csv", prefix_path+"b2.csv");
    Softmax s;

    for (std::string const &data_filename : data_filenames)
    {
        std::cout << "Evaluating " << data_filename;
        std::ifstream ifs{data_filename};
        Eigen::MatrixXd X = csv2mat(ifs);

        // now call your layers
        // --- Your code here
        Eigen::MatrixXd X1 = l1.forward(X);
        X1 = r.forward(X1);
        X1 = l2.forward(X1);
        X1 = s.forward(X1);
        // ---
        ofs << X1.format(vec_csv_format) << std::endl;
        
    }
}