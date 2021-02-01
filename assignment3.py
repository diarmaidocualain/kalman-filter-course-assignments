import numpy as np
from sim.sim2d import sim_run

# Simulator options.
options = {}
options['FIG_SIZE'] = [8,8]

options['DRIVE_IN_CIRCLE'] = True
# If False, measurements will be x,y.
# If True, measurements will be x,y, and current angle of the car.
# Required if you want to pass the driving in circle.
options['MEASURE_ANGLE'] = False
options['RECIEVE_INPUTS'] = False

class KalmanFilter:
    def __init__(self):
        # Initial State. 
        # Size: n_states x 1
        # Populate with starting condition
        self.x = np.matrix([[0.],
                            [0.],
                            [0.],
                            [0.]])

        # Uncertainity Matrix
        # Size: n_states x n_states
        # Polupate with initial uncertainty of each state
        self.P = np.matrix([[100, 0., 1., 0.],
                            [0., 100, 0., 1.],
                            [0., 0., 100, 0.],
                            [0., 0., 0., 100]])

        # Next State Function (State transition matrix)
        # Size: n_states x n_states
        # Populate the matrix so that x_t_plus_one = F.x_t
        # You will get these from the linear algebra equations
        # Note: some of the values will get updated in measure_and_update()
        self.F = np.matrix([[1., 0., 0., 0.],
                            [0., 1., 0., 0.],
                            [0., 0., 1., 0.],
                            [0., 0., 0., 1.]])

        # Measurement Function (Measurement matrix)
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
        # Calculate dt.
        # Put dt into the state transition matrix.
        self.F[0,2] = dt
        self.F[1,3] = dt
        self.x = self.F * self.x
        self.P = self.F * self.P * np.transpose(self.F)
        return
    def measure_and_update(self, measurements, dt):
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
        
        return [self.x[0], self.x[1]]

    def recieve_inputs(self, u_steer, u_pedal):
        return

sim_run(options,KalmanFilter)
