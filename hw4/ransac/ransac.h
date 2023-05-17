#include <random>
#include <iomanip>
#include <iostream>
#include <eigen3/Eigen/Eigen>
#include <vector>

struct Plane
{
    double a;
    double b;
    double c;
    double d;
};

struct FitResult
{
    Plane plane;
    int n_inliers = -1;
};

void pretty_print(std::string const &input_filename, Plane const &plane)
{
    std::cout << std::setprecision(3) << "--infile " << input_filename << " -a " << plane.a << " -b " << plane.b << " -c " << plane.c << " -d " << plane.d << '\n';
}

std::ostream &operator<<(std::ostream &os, Plane const &plane)
{
    os << std::setprecision(3) << plane.a << " " << plane.b << " " << plane.c << " " << plane.d;
    return os;
}

Plane compute_plane_from_points(Eigen::Vector3d const &p0,Eigen::Vector3d const &p1,Eigen::Vector3d const &p2)
{
    // 1. given p0, p1, and p2 form two vectors v1 and v2 which lie on the plane
    // 2. use v1 and v2 to find the normal vector of the plane `n`
    // 3. set a,b,c from the normal vector `n`
    // 4. set `d = -n.dot(p0)`
    // --- Your code here
    Eigen::Vector3d  v1 = p1 - p0;
    Eigen::Vector3d  v2 = p2 - p0;
    Eigen::Vector3d n = v1.cross(v2).normalized();

    return {n[0],n[1],n[2],-n.dot(p0)};


    // ---
}

class BaseFitter
{
public:
    BaseFitter(int num_points) : mt(rd()), dist(0, num_points - 1)
    {
        mt.seed(0);
    }

    /**
     * Given all of the data `points`, select a random subset and fit a plane to that subset.
     * the parameter points is all of the points
     * the return value is the FitResult which contains the parameters of the plane (a,b,c,d) and the number of inliers
     */
    virtual FitResult fit(Eigen::MatrixXd const &points) = 0;

    int get_random_point_idx()
    {
        return dist(mt);
    };

    double const inlier_threshold_{0.02};

private:
    // These are for generating random indices, you don't need to know how they work.
    // Just use `get_random_point_idx()` and `points.row(rand_idx)`
    std::random_device rd;
    std::mt19937 mt;
    std::uniform_int_distribution<int> dist;
};

class AnalyticFitter : public BaseFitter
{
public:
    using PointType_ = Eigen::Vector3d;
    AnalyticFitter(int num_points) : BaseFitter(num_points) {}

    // by writing `override`, the compiler will check that this function actually overrides something 
    // in the base class. Always use this to prevent mistakes in the function signature!
    FitResult fit(Eigen::MatrixXd const &points) override
    {
        // 1. select 3 points from `points` randomly 
        // 2. compute the equation of the plane (HINT: use compute_plane_from_points)
        // 3. compute the `n_inliers` given that plane equation
        // (HINT: multiply the points matrix by the normal vector)
        // --- Your code here
        int idx = get_random_point_idx();
        PointType_ p1 = points.row(idx);
        PointType_ p2;
        do{
            idx = get_random_point_idx();
            p2 = points.row(idx);
        }
        while(p1 == p2);
        PointType_ p3;
        do{
            idx = get_random_point_idx();
            p3 = points.row(idx);
        }
        while(p1 == p3 || p2 == p3 );

        Plane analytic_plane = compute_plane_from_points(p1,p2,p3);
        int n_inliers = 0;

        for (int i=0;i<points.rows();i++){
            PointType_ p_test = points.row(i);
            double dis = std::abs(analytic_plane.a*p_test[0]+analytic_plane.b*p_test[1]+analytic_plane.c*p_test[2]+analytic_plane.d);
            if (dis < inlier_threshold_) n_inliers++;
        }
        // ---

        return {analytic_plane, n_inliers};
    }
};

class LeastSquaresFitter : public BaseFitter
{
public:
    using PointType_ = Eigen::Vector3d;
    LeastSquaresFitter(int num_points, int n_sample_points) : BaseFitter(num_points), n_sample_points_(n_sample_points) {

    }
    FitResult fit(Eigen::MatrixXd const &points) override{
        std::vector<int> indexList;
        int idx;
        Eigen::MatrixXd A(n_sample_points_,3);
        Eigen::VectorXd b = -Eigen::VectorXd::Ones(n_sample_points_);
        while(indexList.size() != n_sample_points_){
            idx = get_random_point_idx();
            if(std::find(indexList.begin(), indexList.end(), idx) == indexList.end()){
                indexList.push_back(idx);
            }
        }
        for(int i=0;i<n_sample_points_;i++){
            PointType_ p = points.row(indexList[i]);
            A.row(i) << p[0], p[1], p[2]; 
        }

        Eigen::Vector3d coeff = A.fullPivHouseholderQr().solve(b);
        Plane p;
        p.a = coeff[0];
        p.b = coeff[1];
        p.c = coeff[2];
        p.d = 1;

        int n_inliers = 0;

        for(int i=0;i<points.rows();i++){
            PointType_ pTest = points.row(i); 
            double dis = std::abs(p.a*pTest[0]+p.b*pTest[1]+p.c*pTest[2]+p.d)/std::sqrt(p.a*p.a+p.b*p.b+p.c*p.c);
            if(dis < inlier_threshold_) n_inliers++;
        }

        return {p,n_inliers};
        
    }

    // You should override the `fit` method here
    // --- Your code here



    // ---
private:
    const int n_sample_points_;
};

Plane ransac(BaseFitter &fitter, Eigen::MatrixXd const &points)
{
    // --- Your code here
    int num_iterations = 100;
    int best_num_inliers = 0;
    FitResult best_result;
    for(int i=0;i<points.rows();i++){
        FitResult result =  fitter.fit(points);
        if(result.n_inliers > best_num_inliers){
            best_num_inliers = result.n_inliers;
            best_result = result;
        }
    }
    // HINT: the number of inliers should be between 20 and 80 if you did everything correctly
    std::cout << best_result.n_inliers << std::endl;
    return best_result.plane;
}