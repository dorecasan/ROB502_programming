#include <eigen3/Eigen/Eigen>
#include <fstream>
#include <iostream>
#include <vector>

using Edge = std::pair<Eigen::Vector2d, Eigen::Vector2d>;

// returns true if the equation a*t^2+b*t+c=0 has a solution between 0 and 1 (inclusive)
// HINT: recall your solution to `quadratic` in HW1 
bool quadratic_has_valid_solution(double const a, double const b,  double const c) {
    // --- Your code here



    // ---
}

class Disc; // forward-declare the type Disc, otherwise we can't use it in the Obstacle base class

class Obstacle
{
public:
    // returns true if the robot is collides with the obstacle
    virtual bool check_collision(Disc const &robot) const = 0;

    // returns true if whether the point p is within this disc
    virtual bool contains(Eigen::Vector2d const &p) const = 0;
};

class Disc : public Obstacle
{
public:
    Disc(double x, double y, double radius) : pos_(x, y), radius_(radius) {}

    // override check_collision (HINT: use the `norm` function!)
    bool check_collision(Disc const &robot) const override
    {
        // --- Your code here



        // ---
    }

    // returns true if the point p is within this disc
    bool contains(Eigen::Vector2d const &p) const override
    {
        // --- Your code here



        // ---
    }


    Eigen::Vector2d pos_;
    double radius_;
};
class Rectangle : public Obstacle
{
public:
    Rectangle(double x1, double y1, double x2, double y2) : bottom_left_(x1, y1), top_right_(x2, y2),
                                                            edges_{{{x1, y1}, {x2, y1}}, {{x2, y1}, {x2, y2}}, {{x2, y2}, {x1, y2}}, {{x1, y2}, {x1, y1}}},
                                                            corners_{{x1, y1}, {x2, y1}, {x2,y2}, {x1,y2}}
    {
    }

    // override check_collision
    // HINT: use the `Rectangle::check_intersection_with_edge`, `Rectangle::contains`, and `Disc::contains` functions
    // --- Your code here



    // ---

    // Override the `contains` function
    // --- Your code here



    // ---

    // (HINT: use the `quadratic_has_valid_solution` function)
    bool check_intersection_with_edge(Edge const &e, Disc const& disc) const
    {
        // --- Your code here



        // ---
    }

    Eigen::Vector2d bottom_left_;
    Eigen::Vector2d top_right_;
    std::vector<Edge> edges_;
    std::vector<Eigen::Vector2d> corners_;
};

bool check_collisions(std::vector<Obstacle const *> const &obstacles, Disc const &robot)
{
    // --- Your code here



    // ---
}