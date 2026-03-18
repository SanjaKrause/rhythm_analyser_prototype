# encoding: utf-8
# cython: embedsignature=True 
from __future__ import absolute_import, division, print_function
import warnings
from typing import List

import numpy as np
from scipy.stats import entropy
cimport numpy as np
cimport cython
import madmom

from numpy.math cimport INFINITY


ctypedef np.uint32_t uint32_t


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def viterbi_matrix(transition_model: madmom.ml.hmm.TransitionModel,
				   observation_model: madmom.ml.hmm.ObservationModel,
				   observations: np.ndarray):
	"""
	Best path matrix for the viterbi algorithm
	Args:
		transition_model: DBN transition model
		observation_model: DBN observation model
		observations: Observations to decode the optimal path for

	Returns: The matrix result of the viterbi algorithm

	"""
	# transition model stuff
	tm = transition_model
	cdef uint32_t [::1] tm_states = tm.states
	cdef uint32_t [::1] tm_pointers = tm.pointers
	cdef double [::1] tm_probabilities = tm.log_probabilities
	cdef unsigned int num_states = tm.num_states

	# observation model stuff
	om = observation_model
	cdef unsigned int num_observations = len(observations)
	cdef uint32_t [::1] om_pointers = om.pointers
	cdef double [:, ::1] om_densities = om.log_densities(observations)

	# current viterbi variables
	cdef double [::1] current_viterbi = np.empty(num_states,
												 dtype=np.float)

	# previous viterbi variables, init with the initial state distribution
	initial_distribution = (np.ones(transition_model.num_states,
                                           dtype=np.float) / 
                                   transition_model.num_states)
	cdef double [::1] previous_viterbi = np.log(initial_distribution)

	# back-tracking pointers
	cdef uint32_t [:, ::1] bt_pointers = np.empty((num_observations,
												   num_states),
												  dtype=np.uint32)
	# define counters etc.
	cdef unsigned int state, frame, prev_state, pointer
	cdef double density, transition_prob

	matrix = np.empty((num_observations,num_states),
												  dtype=np.float32)

	# iterate over all observations
	for frame in range(num_observations):
		# search for the best transition
		for state in range(num_states):
			# reset the current viterbi variable
			current_viterbi[state] = -INFINITY
			# get the observation model probability density value
			# the om_pointers array holds pointers to the correct
			# observation probability density value for the actual state
			# (i.e. column in the om_densities array)
			# Note: defining density here gives a 5% speed-up!?
			density = om_densities[frame, om_pointers[state]]
			# iterate over all possible previous states
			# the tm_pointers array holds pointers to the states which are
			# stored in the tm_states array
			for pointer in range(tm_pointers[state],
								 tm_pointers[state + 1]):
				# get the previous state
				prev_state = tm_states[pointer]
				# weight the previous state with the transition probability
				# and the current observation probability density
				transition_prob = previous_viterbi[prev_state] + \
								  tm_probabilities[pointer] + density
				# if this transition probability is greater than the
				# current one, overwrite it and save the previous state
				# in the back tracking pointers
				if transition_prob > current_viterbi[state]:
					# update the transition probability
					current_viterbi[state] = transition_prob
					# update the back tracking pointers
					bt_pointers[frame, state] = prev_state

		# overwrite the old states with the current ones
		previous_viterbi[:] = current_viterbi
		matrix[frame,:] = (np.copy(np.asarray(current_viterbi)))

	
	state = np.asarray(current_viterbi).argmax()
	# set the path's probability to that of the best state
	log_probability = current_viterbi[state]


	return  matrix, log_probability
