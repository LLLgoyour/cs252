'''data_transformations.py
Jack Dai
Performs translation, scaling, and rotation transformations on data
CS 251 / 252: Data Analysis and Visualization
Spring 2026

NOTE: All functions should be implemented from scratch using basic NumPy WITHOUT loops and high-level library calls.
'''
import numpy as np


def normalize(data):
    '''Perform min-max normalization of each variable in a dataset.

    Parameters:
    -----------
    data: ndarray. shape=(N, M). The dataset to be normalized.

    Returns:
    -----------
    ndarray. shape=(N, M). The min-max normalized dataset.
    '''
    mins = np.min(data, axis=0)
    maxs = np.max(data, axis=0)
    ranges = maxs - mins
    return (data - mins) / ranges


def center(data):
    '''Center the dataset.

    Parameters:
    -----------
    data: ndarray. shape=(N, M). The dataset to be centered.

    Returns:
    -----------
    ndarray. shape=(N, M). The centered dataset.
    '''
    means = np.mean(data, axis=0)
    return data - means


def rotation_matrix_3d(degrees, axis='x'):
    '''Make a 3D rotation matrix for rotating the dataset about ONE variable ("axis").

    Parameters:
    -----------
    degrees: float. Angle (in degrees) by which the dataset should be rotated.
    axis: str. Specifies the variable about which the dataset should be rotated. Assumed to be either 'x', 'y', or 'z'.

    Returns:
    -----------
    ndarray. shape=(3, 3). The 3D rotation matrix.

    NOTE: This method just CREATES and RETURNS the rotation matrix. It does NOT actually PERFORM the rotation!
    '''
    radians = np.deg2rad(degrees)
    cos = np.cos(radians)
    sin = np.sin(radians)

    if axis == 'x':
        return np.array([[1, 0, 0],
                         [0, cos, -sin],
                         [0, sin, cos]])
    if axis == 'y':
        return np.array([[cos, 0, sin],
                         [0, 1, 0],
                         [-sin, 0, cos]])
    if axis == 'z':
        return np.array([[cos, -sin, 0],
                         [sin, cos, 0],
                         [0, 0, 1]])

    raise ValueError("axis must be 'x', 'y', or 'z'")
