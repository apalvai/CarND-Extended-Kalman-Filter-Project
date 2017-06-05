#include <iostream>
#include "tools.h"
#include <math.h>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
    VectorXd rmse(4);
    rmse << 0,0,0,0;
    
    // check the validity of the following inputs:
    //  * the estimation vector size should not be zero
    //  * the estimation vector size should equal ground truth vector size
    if(estimations.size() != ground_truth.size()
       || estimations.size() == 0){
        cout << "Invalid estimation or ground_truth data" << endl;
        return rmse;
    }
    
    //accumulate squared residuals
    for(unsigned int i=0; i < estimations.size(); ++i){
        
        VectorXd residual = estimations[i] - ground_truth[i];
        
        //coefficient-wise multiplication
        residual = residual.array()*residual.array();
        rmse += residual;
    }
    
    //calculate the mean
    rmse = rmse/estimations.size();
    
    //calculate the squared root
    rmse = rmse.array().sqrt();
    
    //return the result
    return rmse;
}

VectorXd Tools::ConvertPolarToCartesianVector(const VectorXd& x_polar) {
    
    VectorXd x_cart(4);
    
    //recover state parameters
    float range = x_polar(0);
    float bearing = x_polar(1);
    float range_rate = x_polar(2);
    
    //pre-compute a set of terms to avoid repeated calculation
    float px = range * cos(bearing);
    float py = range * sin(bearing);
    float vx = range_rate * cos(bearing);
    float vy = range_rate * sin(bearing);
    
    x_cart <<   px,
                py,
                vx,
                vy;
    
    return x_cart;
}

VectorXd Tools::CalculateNonLinearMeasurementVector(const VectorXd& x_state) {
    
    VectorXd hx(3);
    
    //recover state parameters
    float px = x_state(0);
    float py = x_state(1);
    float vx = x_state(2);
    float vy = x_state(3);
    
    //pre-compute a set of terms to avoid repeated calculation
    float c1 = px*px+py*py;
    float c2 = sqrt(c1);
    float pxvx = px*vx;
    float pyvy = py*vy;
    float bearing = atan2(py, px);
    
    //check division by zero
    if(fabs(c1) < 0.0001){
        cout << "CalculateNonLinearMeasurementVector () - Error - Division by Zero" << endl;
        return hx;
    }
    
    if (fabs(px) < 0.0001) {
        cout << "CalculateNonLinearMeasurementVector () - Error - Division by Zero" << endl;
        return hx;
    }
    
    // normalize bearing to be in the range of [-M_PI, M_PI]
    while (bearing > M_PI) {
        cout << "CalculateNonLinearMeasurementVector () - Error - Bearing out of range (over)" << bearing << endl;
        bearing -= 2*M_PI;
    }
    
    while (bearing < -M_PI) {
        cout << "CalculateNonLinearMeasurementVector () - Error - Bearing out of range (under)" << bearing << endl;
        bearing += 2*M_PI;
    }
    
    //compute the hx vector
    hx <<   c2,
            bearing,
            (pxvx+pyvy)/c2;
    
    return hx;
}


MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
    
    MatrixXd Hj(3,4);
    //recover state parameters
    float px = x_state(0);
    float py = x_state(1);
    float vx = x_state(2);
    float vy = x_state(3);
    
    //pre-compute a set of terms to avoid repeated calculation
    float c1 = px*px+py*py;
    float c2 = sqrt(c1);
    float c3 = (c1*c2);
    
    //check division by zero
    if(fabs(c1) < 0.0001){
        cout << "CalculateJacobian () - Error - Division by Zero" << endl;
        return Hj;
    }
    
    //compute the Jacobian matrix
    Hj << (px/c2), (py/c2), 0, 0,
		  -(py/c1), (px/c1), 0, 0,
		  py*(vx*py - vy*px)/c3, px*(px*vy - py*vx)/c3, px/c2, py/c2;
    
    return Hj;
}
