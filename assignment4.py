import numpy as np
from sim.sim2d_prediction import sim_run

# Simulator options.
options = {}
options['FIG_SIZE'] = [8,8]
options['ALLOW_SPEEDING'] = True

class KalmanFilter:
    def __init__(self):
        # Initial State
        # Size: n_states x 1
        # Populate with starting condition
        self.x = np.matrix([[55.],
                            [3.],
                            [5.],
                            [0.]])

        # Uncertainity Matrix
        # Size: n_states x n_states
        # Populate with initial uncertainty of each state
        self.P = np.matrix([[0., 0., 0., 0.],
                            [0., 0,  0., 0.],
                            [0., 0., 0., 0.],
                            [0., 0., 0., 0.]])

        # Next State Function
        # Size: n_states x n_states
        # Populate the matrix so that x_t_plus_one = F.x_t
        # You will get these from the linear algebra equations
        # Note: some of the values will get updated in measure_and_update()
        self.F = np.matrix([[1., 0., 0., 0.],
                            [0., 1., 0., 0.],
                            [0., 0., 1., 0.],
                            [0., 0., 0., 1.]])
        
	# Measurement Function
        # Size: n_measurements x n_states
        # Populate the matrix so that x_measured = HZ
        self.H = np.matrix([[1., 0., 0., 0.],
                            [0., 1., 0., 0.]])
        
        # Measurement Uncertainty
        # Size: n_measurements x n_measurements
        # How accurate each of your measurements are
        # 0 = very certain that they are accurate 100%  
        # Values only go along the diagonal
        self.R = np.matrix([[1.0, 0.0],
                            [0.0, 1.0]])
        
        # Identity Matrix
        self.I = np.matrix([[1., 0., 0., 0.],
                            [0., 1., 0., 0.],
                            [0., 0., 1., 0.],
                            [0., 0., 0., 1.]])
    def predict(self, dt):
        # Put dt into the state transition matrix.
        self.F[0,2] = dt
        self.F[1,3] = dt
        self.x = self.F * self.x
        self.P = self.F * self.P * np.transpose(self.F)
        return
    def measure_and_update(self,measurements, dt):
        self.F[0,2] = dt
        self.F[1,3] = dt
        Z = np.matrix(measurements)
        y = np.transpose(Z) - (self.H * self.x)
        S = self.H * self.P * np.transpose(self.H) + self.R
        K = self.P * np.transpose(self.H) * np.linalg.inv(S)
        self.x = self.x + K * y
        self.P = (self.I - K * self.H) * self.P

        self.P[0,0] = self.P[0,0] + 0.1
        self.P[1,1] = self.P[1,1] + 0.1
        self.P[2,2] = self.P[2,2] + 0.1
        self.P[3,3] = self.P[3,3] + 0.1
 
        return [self.x[0], self.x[1]]

    def predict_red_light(self,light_location):
        light_duration = 3
        F_new = np.copy(self.F)
        F_new[0,2] = light_duration
        F_new[1,3] = light_duration
        x_new = F_new * self.x
        if x_new[0] < light_location:
            return [False, x_new[0]]
        else:
            return [True, x_new[0]]

    def predict_red_light_speed(self, light_location):
        light_duration = 3
        F_new = np.copy(self.F)
        # Accelerate
        duration = 1
        u_pedal = 1.5
        U = np.matrix([[0.],
                       [0.],
                       [u_pedal],
                       [0.]])
        F_new[0,2] = duration 
        F_new[1,3] = duration 
        
        x_new = F_new * self.x + U
        # Constant speed
        duration = 2
        F_new[0,2] = duration 
        F_new[1,3] = duration 
        x_new = F_new * x_new
        if x_new[0] < light_location:
            return [False, x_new[0]]
        else:
            return [True, x_new[0]]


for i in range(0,5):
    sim_run(options,KalmanFilter,i)
