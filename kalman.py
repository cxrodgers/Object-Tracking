import numpy as np
from scipy.optimize import linear_sum_assignment

## Factories
def create_sensor_functions(L, folx, foly):
    """Return sensor function and its jacobian for a follicle and length"""
    def sensor_function(state):
        """Return prediction of tip from state"""
        theta = state[0][0]
        return np.array([
            [folx - np.abs(L * np.cos(theta)), L * np.sin(theta) + foly, folx, foly]
        ]).T
    
    def sensor_function_jacobian(state):
        """Return jacobian of prediction of tip from state"""
        theta = state[0][0]
        return np.array([
          [L * np.sin(theta), 0],
          [L * np.cos(theta), 0],
          [0, 0],
          [0, 0],
        ])

    return sensor_function, sensor_function_jacobian

def create_state_functions(dt, period):
    """Return state function and its jacobian for a certain period"""
    def state_function(state):
        """Return new angle and velocity from current state and period"""
        theta = state[0][0]
        omega = state[1][0]
        theta_new = theta + omega * dt


        omega_new = omega + (-1 * (2 * np.pi / period) ** 2) * dt * theta
        if np.abs(omega_new) > np.pi:
            omega_new = np.pi

        return np.array([[theta_new, omega_new]]).T

    def state_function_jacobian(state):
        """Return jacobian of new angle and velocity"""
        A = np.array([
          [1,       dt], 
          [(-1 * (2 * np.pi / period) ** 2) * dt, 1 ],
        ])
        return A

    return state_function, state_function_jacobian


## Kalman filter objects
class ExtendedKalmanThread(object):
    """An implemention of the Kalman algorithm.
    
    An estimation is a juxtaposed with an observation at each time step.
    """
    def __init__(self, initial_state, 
        state_function, state_function_jacobian,
        sensor_function, sensor_function_jacobian,
        initial_cov_estimation=0, cov_process_noise=0, cov_sensors=0,
        ):
        """Initialize a new ExtendedKalmanThread.
        
        x0, f, F, h, H, P0=0, Q=0, R=0):

        initial_state (x0): column of initial state values
        state_function, state_function_jacobian (f, F)
        sensor_function, sensor_function_jacobian (h, H)
        initial_cov_estimation (P0): initial covariance matrix of 
            estimation process
        cov_process_noise (Q): covariance matrix of process noise
        cov_sensors (R): covariances matrix of sensors
        """
        ## Store variables
        # Sensor functions
        self.sensor_function = sensor_function
        self.sensor_function_jacobian = sensor_function_jacobian
        
        # State functions
        self.state_function = state_function
        self.state_function_jacobian = state_function_jacobian
        
        # Covariance matrices
        self.cov_process_noise = cov_process_noise
        self.cov_sensors = cov_sensors
        
        # Initialize the state
        self.current_state = initial_state
        
        # track the covariance matrix of estimation process
        self.initial_cov_estimation = initial_cov_estimation
        self.current_cov_estimation = initial_cov_estimation
        self.det_cov_estimation = np.linalg.det(initial_cov_estimation)
        
        # Number of states and sensors
        nstates = initial_state.size
        nsensors = cov_sensors.shape[0]

        # more stuff
        self.observation_solution = self.sensor_function(self.current_state) # z
        self.n_steps = 1 # k

    def update_preview(self, current_observation):
        """Provides a snapshot of one timestep of the algorithm
        
        z: current observation values    
        
        self.state_function is used to predict the next state from the
        current state (self.x)
        
        
        """
        ## Prediction Step
        # Predict the new state
        pred_state = self.state_function(self.current_state)
        F_res = self.state_function_jacobian(self.current_state)
        P_new = np.dot(np.dot(F_res, self.current_cov_estimation), 
            F_res.T) + self.cov_process_noise

        ## Update Step
        H_res = self.sensor_function_jacobian(pred_state)
        h_res = self.sensor_function(pred_state)
        
        G = np.dot(
                    np.dot(P_new, H_res.T),
                    np.linalg.pinv(
                        np.dot(
                            np.dot(H_res, P_new), 
                            H_res.T
                        ) + self.cov_sensors
                    )
                )


        pred_state = pred_state + np.dot(G, current_observation - h_res)
        P_new = np.dot(np.eye(pred_state.size) - np.dot(G, H_res), P_new)
        soln = h_res
        detP = np.linalg.det(P_new)

        return pred_state, P_new, detP, soln

    def update(self, z):
        """Runs one time step of the algorithm
        
        z: current observation values
        """
        x, P, detP, soln = self.update_preview(z)
        self.current_state = x
        self.current_cov_estimation = P
        self.det_cov_estimation = detP
        self.observation_solution = soln
        self.n_steps += 1

        return soln

    def set_state_functions(self, f, F):
        """Sets state-defining functions
            
        setter: Two item tuple with the state function and its Jacobian
        """
        self.state_function = f
        self.state_function_jacobian = F

class KalmanTracker(object):
    """
    Uses Kalman Filters and assignments with a Hungarian algorithm to keep track
    of observed objects
    """
    def __init__(self, initial_cov_estimation, cov_process_noise, cov_sensors, 
        state_factory=create_state_functions, 
        sensor_factory=create_sensor_functions):
        """Init a new KalmanTracker
        
        """
        self.predictors = [] #List of KalmanThreads
        self.current_predictor_label = 1
        self.strikes = []

        # Covariance matrices
        self.initial_cov_estimation = initial_cov_estimation
        self.cov_process_noise = cov_process_noise
        self.cov_sensors = cov_sensors
        self.state_factory = state_factory
        self.sensor_factory = sensor_factory
        self.k = 0

    def detect(self, observations):
        """Label each observation.

        observations : list of dicts
        Each dict has the following keys:
            'x0': 2d array of [theta, omega]
            'z': 2d array [xtip, ytip, xfol, yfol] (the observation itself)
            'state_factory_args': list of dt, period
            'sensor_factory_args': list of length, xfol, yfol
        
        We first calculate the "cost matrix", the difference between prediction
        and data for every possible assignment of prediction to data. Then
        we use linear_sum_assignment to choose the best possible assignment.
        Finally we update each predictor with these assignments.
        
        """
        ## Generate the cost matrix
        # This will store the cost associated with each possible assignment
        # of observation to predictor
        #rows of cost matrix represent observations, columns represent predictions
        cost_matrix = np.zeros((len(observations), len(self.predictors)))

        # Calculate every entry in the cost_matrix
        for i, observation_dict in enumerate(observations):
            for j, predictor in enumerate(self.predictors):
                z = observation_dict['z']

                # Generate state_function, state_function_jacobian using the
                # data from this observation
                state_function, state_function_jacobian = self.state_factory(
                    *observation_dict["state_factory_args"])

                # Set the state functions for this predictor
                self.predictors[j]['predictor'].set_state_functions(
                    state_function, state_function_jacobian)

                # Get the prediction for this predictor
                _, _, detP, prediction = self.predictors[j][
                    'predictor'].update_preview(z)

                # Store the prediction minus the observation in the cost_matrix
                cost_matrix[i, j] = np.linalg.norm(prediction - z)

        ## Hungarian algorithm
        observation_indices, prediction_indices = linear_sum_assignment(
            cost_matrix)
        
        ## Update
        for observation_index, prediction_index in zip(
            observation_indices, prediction_indices):
            # Get the observation
            z = observations[observation_index]['z']

            # Update the associated predictor
            self.predictors[prediction_index]['predictor'].update(z)

        # Prepare to add new observation if number exceeds predictions
        if len(observations) > len(self.predictors):
            mask = np.in1d(np.arange(len(observations)), observation_indices)

            unused_indices = np.where(~mask)[0]
            for i in unused_indices:  
                observation_dict = observations[i]
                x0 = observation_dict["x"]
                self.addPredictor(
                    x0,
                    observation_dict["state_factory_args"],
                    observation_dict["sensor_factory_args"],
                )
                prediction_indices = np.append(prediction_indices, i)

        ## Count strikes
        # If cost of any assignment is too high, increment strikes...threshold is arbitrary 
        for i, j in zip(observation_indices, prediction_indices):
            cost = cost_matrix[i, j]
            if cost > 50:
                self.strikes[j] += 1
            else:
                self.strikes[j] = 0

        ## Return the label (numerical) of each assignment
        labels = [self.predictors[i]['label'] for i in prediction_indices]

        return labels

    def addPredictor(self, initial_state, state_factory_args=[], sensor_factory_args=[]):
        # Generate state and sensor functions
        state_function, state_function_jacobian = self.state_factory(
            *state_factory_args)
        sensor_function, sensor_function_jacobian = self.sensor_factory(
            *sensor_factory_args)

        # Init a new KalmanThread
        new_kalman_thread = ExtendedKalmanThread(
            initial_state, 
            state_function, state_function_jacobian,
            sensor_function, sensor_function_jacobian,
            initial_cov_estimation=self.initial_cov_estimation, 
            cov_process_noise=self.cov_process_noise, 
            cov_sensors=self.cov_sensors,
        )        
        
        # Store it
        self.predictors.append(
            {
            'predictor' : new_kalman_thread,
            'label' : self.current_predictor_label
            }
        )

        self.strikes.append(0)
        self.current_predictor_label += 1
