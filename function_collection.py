import numpy as np


'''
A collection of example functions for the technique
'''

def f1(x = np.linspace(-2,2,50)):
	'''
	polynomial function
	'''
	np.random.seed(0)
	X = np.linspace(-2, 2, 50).reshape(-1, 1)
	y = 2 * X**3 - 3 * X + 2 * X + np.random.normal(0, 0.5, X.shape[0]).reshape(-1, 1)
	return X, y

def f2(x = np.linspace(-2,2,50), sin_range: float = 0.8):
	'''
	sinusoid function for a specific range, followed by a polynomial function for the rest
	'''
	np.random.seed(0)
	# set sine range for sin_range % of the data
	sin_range = int(sin_range * len(x))
	# poly range is the rest of the data
	poly_range = len(x) - sin_range
	# create sine data
	sine_data = np.sin(x[:sin_range] * 2 * np.pi)
	# create polynomial data
	poly_data = 2 * x[sin_range:]**3 - 3 * x[sin_range:] + 2 * x[sin_range:] + np.random.normal(0, 0.5, poly_range).reshape(-1, 1)
	# combine data
	y = np.concatenate((sine_data, poly_data), axis = 0)
	X = np.concatenate((x[:sin_range], x[sin_range:]), axis = 0).reshape(-1, 1)
	return X, y

	
def mask(X,y, method = 'random', test_size = 0.2):
	'''
	mask function for training and testing
	'''
	# We can either create a test set in between the training set or at the end of the training set
	
	# method should be either 'random', 'end', 'start' or 'middle'
	if method == 'random':
		# Randomly select the test set
		test_index = np.random.choice(X.shape[0], int(X.shape[0] * test_size), replace = False)
		train_index = np.setdiff1d(np.arange(X.shape[0]), test_index)
		X_train, y_train = X[train_index], y[train_index]
		X_test, y_test = X[test_index], y[test_index]
	elif method == 'end':
		raise NotImplementedError
	elif method == 'start':
		raise NotImplementedError
	elif method == 'middle':
		raise NotImplementedError
	else:  
		raise ValueError('method should be either random, end, start or middle')
	
	return X_train, y_train, X_test, y_test
