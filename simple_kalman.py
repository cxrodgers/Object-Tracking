import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import pandas
import pdb
from filterpy.kalman import KalmanFilter, UnscentedKalmanFilter, MerweScaledSigmaPoints, unscented_transform, JulierSigmaPoints
from filterpy.common import Q_discrete_white_noise

class DerivedUnscentedKalmanFilter(UnscentedKalmanFilter):
    def __init__(self, dim_x, dim_z, dt, hx, fx, points):
        UnscentedKalmanFilter.__init__(
            self, dim_x=dim_x, dim_z=dim_z, dt=dt, hx=hx, fx=fx, points=points, 
        )

    def get_prediction(self, dt=None,  UT=None, fx_args=()):
        if dt is None:
            dt = self._dt

        if not isinstance(fx_args, tuple):
            fx_args = (fx_args,)

        if UT is None:
            UT = unscented_transform

        # calculate sigma points for given mean and covariance
        sigmas = self.points_fn.sigma_points(self.x, self.P)

        for i in range(self._num_sigmas):
            self.sigmas_f[i] = self.fx(sigmas[i], dt, *fx_args)

        x, P = UT(self.sigmas_f, self.Wm, self.Wc, self.Q,
                            self.x_mean, self.residual_x)
        return x, P

class UnscentedKalmanTracker(object):
    """
    Uses Kalman Filters and assignments with a Hungarian algorithm to keep track
    of observed objects
    """
    def __init__(self, P0, Q, R, f, h, dt):
        self.predictors = [] #List of KalmanThreads
        self.current_predictor_label = 1
        # {'label': 1, 'predictor': KalmanThread}
        self.strikes = []

        self.P0, self.Q, self.R, self.f, self.h, self.dt = P0, Q, R, f, h, dt

        self.k = 0


    def detect(self, observations):
        # print [predictor['label'] for predictor in self.predictors]
        #Remove a predictor if it has had too many erroneous walks
        # for i, strikes in enumerate(self.strikes):
        #   if strikes > 5:
        #     del self.strikes[i]
        #     del self.predictors[i]
        #     self.current_predictor_label -= 1

        #update each prediction and form a cost matrix


        cost_matrix = np.zeros((len(observations), len(self.predictors)))

        #rows of cost matrix represent observations, columns represent predictions
        for i, observation_dict in enumerate(observations):
            for j, predictor in enumerate(self.predictors):
                z = observation_dict['z']
                fx_args = observation_dict['fx_args']
                # print 'here'
                x0 = predictor['predictor'].x
                x, P = predictor['predictor'].get_prediction(fx_args=fx_args)
                # print "xtip:{}\tytip:{}\tpixlen:{}\tw:{}".format(x0[0], x0[1], x0[4], x0[5])
                prediction = predictor['predictor'].hx(x)
                # print prediction
                # print 'there'
                dist = np.linalg.norm(prediction - z)
                detP = np.linalg.det(P)
                cost = dist
                # cost = dist * detP
                # print cost
                cost_matrix[i, j] = cost
                # print detP_weight
        # print 'here'
        observation_indices, prediction_indices = linear_sum_assignment(cost_matrix)
        # print 'there'
        preds = []
        for i in range(len(observation_indices)):
            observation_index = observation_indices[i]
            prediction_index = prediction_indices[i]

            cost = cost_matrix[observation_index, prediction_index]
            # print cost
            # if cost < 30:
            observation_dict = observations[observation_index]
            z = observation_dict['z']
            fx_args = observation_dict['fx_args']

            self.predictors[prediction_index]['predictor'].predict(fx_args=fx_args)
            
            self.predictors[prediction_index]['predictor'].update(z)
            # preds.append(self.predictors[prediction_index]['predictor'].hx(self.predictors[i]['predictor'].x))


        # Prepare to add new observation if number exceeds predictions
        if len(observations) > len(self.predictors):
            mask = np.in1d(np.arange(len(observations)), observation_indices)

            unused_indices = np.where(~mask)[0]
            for i in unused_indices:  
                observation_dict = observations[i]
                x0 = observation_dict["x"]
                self.addPredictor(x0)
                prediction_indices = np.append(prediction_indices, i)

        # if len(self.predictors) > len(observations):
        #   mask = np.in1d(np.arange(len(self.predictors)), prediction_indices)

        #   unused_indices = np.where(~mask)[0]

        #   # for i in unused_indices:
        #   #   self.predictors[i]['predictor'].P



        #If cost of any assignment is too high, increment strikes...threshold is arbitrary 
        for i, j in zip(observation_indices, prediction_indices):
            cost = cost_matrix[i, j]
            # print "i:{}\t j:{}\t cost:{}".format(i, j, cost)

            if cost > 50:
                self.strikes[j] += 1
            else:
                self.strikes[j] = 0

        # Return the label (numerical) of each assignment
        labels = [self.predictors[i]['label'] for i in prediction_indices]
        preds = [self.predictors[i]['predictor'].hx(self.predictors[i]['predictor'].x) for i in prediction_indices]
        return labels, preds

    def addPredictor(self, x0):
        dim_x = self.Q.shape[0]
        dim_z = self.R.shape[0]
        points_class = JulierSigmaPoints(dim_x, 0)

        kf = DerivedUnscentedKalmanFilter(
            dim_x=dim_x, dim_z=dim_z,
            dt=self.dt, hx=self.h, fx=self.f,
            points=points_class,
        )
        kf.x = x0
        kf.P = self.P0
        kf.R = self.R
        kf.Q = self.Q

        self.predictors.append(
          {
            'predictor' : kf,
            'label' : self.current_predictor_label
          }
        )

        self.strikes.append(0)

        self.current_predictor_label += 1

class KalmanTracker(object):
    """
    Uses Kalman Filters and assignments with a Hungarian algorithm to keep track
    of observed objects
    """
    def __init__(self, P0, F, H, Q, R):
        self.predictors = [] #List of KalmanThreads
        self.current_predictor_label = 1
        # {'label': 1, 'predictor': KalmanThread}
        self.strikes = []

        self.P0, self.Q, self.R, self.F, self.H = P0, Q, R, F, H

        self.k = 0


    def detect(self, observations):
        # print [predictor['label'] for predictor in self.predictors]
        #Remove a predictor if it has had too many erroneous walks
        # for i, strikes in enumerate(self.strikes):
        #   if strikes > 5:
        #     del self.strikes[i]
        #     del self.predictors[i]
        #     self.current_predictor_label -= 1

        #update each prediction and form a cost matrix


        cost_matrix = np.zeros((len(observations), len(self.predictors)))

        #rows of cost matrix represent observations, columns represent predictions
        for i, observation_dict in enumerate(observations):
            for j, predictor in enumerate(self.predictors):
                z = observation_dict['z']

                x, P = predictor['predictor'].get_prediction()

                dist = np.linalg.norm(x - z)
                detP = np.linalg.det(P)
                cost = dist
                # cost = dist * detP
                # print cost
                cost_matrix[i, j] = cost
                # print detP_weight
        # print 'here'
        observation_indices, prediction_indices = linear_sum_assignment(cost_matrix)
        # print 'there'

        for i in range(len(observation_indices)):
            observation_index = observation_indices[i]
            prediction_index = prediction_indices[i]

            cost = cost_matrix[observation_index, prediction_index]
            # print cost
            # if cost < 30:
            observation_dict = observations[observation_index]
            z = observation_dict['z']

            self.predictors[prediction_index]['predictor'].predict()
            self.predictors[prediction_index]['predictor'].update(z)


        # Prepare to add new observation if number exceeds predictions
        if len(observations) > len(self.predictors):
            mask = np.in1d(np.arange(len(observations)), observation_indices)

            unused_indices = np.where(~mask)[0]
            for i in unused_indices:  
                observation_dict = observations[i]
                x0 = observation_dict["x"]
                self.addPredictor(x0)
                prediction_indices = np.append(prediction_indices, i)

        # if len(self.predictors) > len(observations):
        #   mask = np.in1d(np.arange(len(self.predictors)), prediction_indices)

        #   unused_indices = np.where(~mask)[0]

        #   # for i in unused_indices:
        #   #   self.predictors[i]['predictor'].P



        #If cost of any assignment is too high, increment strikes...threshold is arbitrary 
        for i, j in zip(observation_indices, prediction_indices):
            cost = cost_matrix[i, j]
            # print "i:{}\t j:{}\t cost:{}".format(i, j, cost)

            if cost > 50:
                self.strikes[j] += 1
            else:
                self.strikes[j] = 0

        # Return the label (numerical) of each assignment
        labels = [self.predictors[i]['label'] for i in prediction_indices]
        preds = [self.predictors[i]['predictor'].x for i in prediction_indices]

        return labels, preds

    def addPredictor(self, x0):
        dim_x = self.Q.shape[0]
        dim_z = self.R.shape[0]

        kf = KalmanFilter(dim_x=dim_x, dim_z=dim_z)
        kf.x = x0
        kf.F = self.F
        kf.H = self.H
        kf.P = self.P0
        kf.R = self.R
        kf.Q = self.Q

        self.predictors.append(
          {
            'predictor' : kf,
            'label' : self.current_predictor_label
          }
        )

        self.strikes.append(0)

        self.current_predictor_label += 1

def state_function(state, dt, omega, angle):
    tipx, tipy, folx, foly, pixlen = state[0], state[1], state[2], state[3], state[4]

    theta = -1 * np.arctan((tipy - foly) / (tipx - folx))
    # print theta * 180 / np.pi
    # theta_new = theta + omega * dt
    # v = omega * pixlen
    omega = (angle - theta) / dt
    v = omega * pixlen

    # print np.cos(theta_new - theta)
    # tipx = tipx + np.cos(theta_new - theta) * pixlen
    # tipy = tipy + np.sin(theta_new - theta) * pixlen
    # print -v * np.cos(theta) * dt, -v * np.sin(theta) * dt
    tipx = tipx - v * np.cos(theta + np.pi / 2) * dt
    tipy = tipy + v * np.sin(theta) * dt


    # print midpoint * 180 / np.pi
    # print (theta - midpoint) * 180 / np.pi
    # print ((2 * np.pi / T) ** 2) * (theta - midpoint) * dt
    # omega = omega 
    # omega = omega - ((2 * np.pi / T) ** 2) * (theta - midpoint) * dt

    return np.array([tipx, tipy, folx, foly, pixlen])

def measurement_function(state):
    tipx, tipy, folx, foly, pixlen = state[0], state[1], state[2], state[3], state[4]

    return np.array([tipx, tipy, folx, foly, pixlen])



#State is [tipx, tipy, folx, foly, pixlen, omega]
dim_x = 5
dim_z = 5

# F = np.eye(dim_x)
# F = np.array([
#     [1, 0,  0,  0,  ]
# ])
# H = np.eye(dim_z)

# P0 = np.eye(dim_x) * 2
# R = np.eye(dim_z)
# Q = np.eye(dim_x)

P0 = np.eye(dim_x) * np.array([1, 1, 1, 0.5, 0.0001,])
R = np.eye(dim_z) * np.array([1, 1, 0.0001, 0.0001, 5]) * 200
Q = np.eye(dim_x) * np.array([1, 1, 0.0001, 0.0001, 0.0001,])

dt = 30 ** -1

# tracker = KalmanTracker(
#     P0=P0, Q=Q, R=R, F=F, H=H,
# )
tracker = UnscentedKalmanTracker(P0, Q, R, state_function, measurement_function, dt)
whisker_colors = ['k', 'b', 'g', 'r', 'c', 'm', 'y', 'pink', 'orange']
predictions = {}



data = pandas.read_pickle('15000_frames_revised.pickle')
data = data[(data.frame > 10000) & (data.frame < 10500) ]

data_filtered = data[data.pixlen > 20].groupby('frame')
angles = data_filtered['angle'].apply(lambda x: x.mean()) * np.pi / 180
diffs = angles.diff()


# To estimate the period, find time it takes for change in mean angle to change signs

# start_frame = data_filtered.frame[0]
# end_frame = start_frame + 1
# initial_direction = diffs[start_frame]
# while not (np.sign(diffs[end_frame]) == np.sign(initial_direction)):
#     end_frame += 1
#     print end_frame

# period = dt * (end_frame - start_frame) * 2
# disp = (angles[end_frame] - angles[start_frame]) / 2
# start_frame, end_frame = end_frame, end_frame + 1
# initial_direction = diffs[start_frame]

for frame, observations in data_filtered:
    if frame == 0:
        continue
    if frame % 1000 == 0:
        print frame
    
    indices = observations.index.values
    observation_dicts = []

    for j, observation in observations.iterrows():
        pixlen, tipx, tipy, folx, foly, angle = observation.pixlen, observation.tip_x, observation.tip_y, observation.fol_x, observation.fol_y, observation.angle
        angle *= np.pi / 180
        z = np.array([tipx, tipy, folx, foly, pixlen])
        omega = ((diffs[frame]) / dt)

        # print"Calculated angle:{}\tMeasured angle:{}".format(np.arctan((tipy - foly) / (tipx - folx)), angle)

        x0 = np.array(
            [tipx, tipy, folx, foly, pixlen]
        )

        observation_dicts.append({
            "x" : x0,
            "z" : z,
            "fx_args" : (omega, angle)
        })

    labels, preds = tracker.detect(observation_dicts)
    predictions[frame] = dict([(labels[i], preds[i]) for i in range(len(preds))])
    data.loc[indices, 'color_group'] = labels

    # Alter period if a sign change is observed
    # if frame <= end_frame:
    #     if not (np.sign(diffs[end_frame]) == np.sign(initial_direction)):
    #         print "changed signs!"
    #         period = dt * (end_frame - start_frame) * 2
    #         disp = (angles[end_frame] - angles[start_frame]) / 2
    #         start_frame, end_frame = end_frame, end_frame + 1
    #         initial_direction = diffs[start_frame]
    #     else:
    #         end_frame = frame


subset = data[ (data.frame > 10000) & (data.pixlen < 13000)].groupby('frame')

plt.ion()
for frame, whiskers in subset:
    if frame == 0:
        continue
    plt.clf()
    plt.xlim(0, 640)
    plt.ylim(0, 640)
    plt.gca().invert_yaxis()
    plt.title('Frame: {}'.format(frame))

    preds = predictions[frame]

    for key in preds.keys():
        z = preds[key]
        color = whisker_colors[int(key)]
        tipx, tipy, foly  = z[0], z[1], z[2]
        plt.plot(tipx, tipy, marker='o', color=color, markersize=15)


    for i, whisker in whiskers.iterrows():
        folx, foly = whisker['fol_x'], whisker['fol_y']
        tipx, tipy = whisker['tip_x'], whisker['tip_y']
        # angle = whisker['angle'] * np.pi / 180
        # print"Calculated angle:{}\tMeasured angle:{}".format(np.arctan((tipy - foly) / (tipx - folx)), angle)

        color = whisker_colors[int(whisker['color_group'])]

        plt.plot([folx, tipx], [foly, tipy], color=color)

    plt.figtext(0.4, 0.3, angles[frame] * 180 / np.pi)

    plt.pause(0.05)


