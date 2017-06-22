# Run the kalman filter algorithm on whisker data
import argparse
import os
import pandas
import matplotlib.pyplot as plt
import whiskvid
import numpy as np
import tables
import my
import numpy as np
from kalman import *

## Parse arguments
parser = argparse.ArgumentParser(description="""
    Run the kalman filter on an example whiskers dataset.
""")
parser.add_argument("output", 
    help='Name of the output file containing labeled whiskers')
args = parser.parse_args()
output_filename = args.output

## Factories
def create_sensor_functions(L, folx, foly):
    """Return sensor function and its jacobian for a follicle and length"""
    def sensor_function(state):
        """Return prediction of tip from state"""
        theta = state[0][0]
        return np.array([[
            folx - np.abs(L * np.cos(theta)), 
            L * np.sin(theta) + foly, folx, foly]
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

# Load the data to classify
mwe = pandas.read_pickle(os.path.expanduser(
    '~/mnt/nas2_home/whisker/test_bed/161215_KM91/masked_whisker_ends'))

# Just run on the first N frames
mwe = mwe[mwe.frame < 1000].copy()

# Set of uncertainty matrices for kalman filter
P0 = np.eye(2) * 10
Q = np.eye(2) * 0.1
R = np.eye(4) * 0.1

#Initialize tracker for observed whiskers
tracker = KalmanTracker(
    initial_cov_estimation=P0, 
    cov_process_noise=Q, 
    cov_sensors=R,
    state_factory=create_state_functions, 
    sensor_factory=create_sensor_functions,
)

mwe['ordinal'] = 0
# Group data by frames and limit to largest n whiskers in frame that are above a certain length threshold
print "filtering whiskers of interest"
# mwe_filtered = mwe[mwe.pixlen > 40].groupby('frame', as_index=False).apply(lambda x: x.nlargest(10, 'pixlen')).groupby('frame')
mwe_filtered = mwe[mwe.pixlen > 40].groupby('frame')

#Use frames as timepoints
dt = 30 ** -1

#Find mean angle difference between each frame to guess changes in motion direction
#diffs = mwe_filtered['angle'].apply(lambda x: x.mean()).diff()
diffs = mwe_filtered['angle'].mean().diff()

# To estimate the period, find time it takes for change in mean angle to change signs
start_frame = 1
end_frame = 2
initial_direction = diffs[start_frame]
while not (np.sign(diffs[end_frame]) == np.sign(initial_direction)):
    end_frame += 1

period = dt * (end_frame - start_frame) * 2
start_frame, end_frame = end_frame, end_frame + 1
initial_direction = diffs[start_frame]

# Run one iteration of the Kalman Tracker for each frame
print "Shifting through frames and classifying"
for frame, observations_in_frame in mwe_filtered:
    # Announce
    if frame == 0:
        continue
    if frame % 100 == 0:
        print frame
    
    # indices = [idx[1] for idx in observations_in_frame.index.values]
    indices = observations_in_frame.index.values

    # Extract state and parameters from each observation
    observation_dicts = []
    for j, o in observations_in_frame.iterrows():
        # angular velocity (omega) is set on a frame to frame basis
        omega = (diffs[frame]) / dt 

        # Pull theta from the observed angle
        theta, omega = o.angle * np.pi / 180, omega * np.pi / 180

        length, xtip, ytip, xfol, yfol = o.pixlen, o.tip_x, o.tip_y, o.fol_x, o.fol_y

        x0 = np.array([[theta, omega]]).T
        # z is an array of the x and y coordinates of the tip. 
        # Can calculate as shown or use xtip, ytip
        z = np.array([[xtip, ytip, xfol, yfol]]).T

        #Parameters that will be passed into the Kalman Filter each iteration
        sensor_factory_args = [length, xfol, yfol]
        state_factory_args = [dt, period]

        #Keep track of observation's features in an organized way for the filter
        observation_dicts.append({
            "x" : x0,
            "z" : z,
            "sensor_factory_args": sensor_factory_args,
            "state_factory_args" : state_factory_args,
        })

    # Run the filter, and generate labels for each observation 
    labels = tracker.detect(observation_dicts)

    mwe.loc[indices, 'ordinal'] = labels

    # Alter period if a sign change is observed
    if frame <= end_frame:
        if not (np.sign(diffs[end_frame]) == np.sign(initial_direction)):
            period = dt * (end_frame - start_frame) * 2
            start_frame, end_frame = end_frame, end_frame + 1
            initial_direction = diffs[start_frame]
        else:
            end_frame = frame

mwe['color_group'] = mwe['ordinal']

# Now pickle mwe and run validate_classification_results on it
mwe.to_pickle(output_filename)