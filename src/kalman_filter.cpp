#include "kalman_filter.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
    /* predict the state */
    x_ = F_ * x_;
    P_ = F_ * P_ * F_.transpose() + Q_;
    
    // std::cout << "Prediction x: " << x_ << std::endl;
    // std::cout << "Prediction P: " << P_ << std::endl;
}

void KalmanFilter::Update(const VectorXd &z) {
    /* update the state by using Kalman Filter equations */
    
    MatrixXd Ht = H_.transpose();
    
    long x_size = x_.size();
    MatrixXd I = MatrixXd::Identity(x_size, x_size);
    
    VectorXd y = z - H_ * x_; // 2 element vector
    MatrixXd S = (H_ * P_ * Ht) + R_; // 2x2 matrix
    MatrixXd K = P_ * Ht * S.inverse(); // 4x2 matrix
    
    x_ = x_ + K * y;
    P_ = (I - K * H_) * P_;
    
    // std::cout << "Update after Laser measurement x: " << x_ << std::endl;
    // std::cout << "Update after Laser measurement P: " << P_ << std::endl;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
    /* update the state by using Extended Kalman Filter equations */
    
    Tools tools = Tools();
    
    MatrixXd Hj = tools.CalculateJacobian(x_);
    MatrixXd Hjt = Hj.transpose();
    
    long x_size = x_.size();
    MatrixXd I = MatrixXd::Identity(x_size, x_size);
    
    VectorXd hx = tools.CalculateNonLinearMeasurementVector(x_);
    VectorXd y = z - hx; // 2 element vector
    MatrixXd S = (Hj * P_ * Hjt) + R_; // 3x3 matrix
    MatrixXd K = P_ * Hjt * S.inverse(); // 3x3 matrix
    
    x_ = x_ + K * y;
    P_ = (I - K * Hj) * P_;
    
    // std::cout << "Update after Radar measurement x: " << x_ << std::endl;
    // std::cout << "Update after Radar measurement P: " << P_ << std::endl;
}
