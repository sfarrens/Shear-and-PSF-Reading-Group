import numpy as np


def get_centroid(data):

    # Sum flux over x and y individually
    sum_i = np.array([np.sum(data, axis=i) for i in (1, 0)])

    # Get range of x and y values
    ranges = np.array([np.arange(i) for i in data.shape])

    # Calculate centroids
    cents = np.sum(sum_i * ranges, axis=1) / np.sum(data)

    return cents.astype(int)


def get_moments(data):

    centroid = get_centroid(data)
    ranges = np.array([np.arange(i) for i in data.shape])

    x = np.outer(ranges[0] - centroid[0], np.ones(data.shape[1]))
    y = np.outer(np.ones(data.shape[0]), ranges[1] - centroid[1])

    q = np.array([np.sum(data * xi * xj) for xi in (x, y) for xj in (x, y)])
    q = (q / np.sum(data)).reshape(2, 2).astype('complex')

    return q


def get_ellipticity(data, method='chi'):

    # Calculate moments
    q = get_moments(data)

    # Calculate the image size.
    r2 = q[0, 0] + q[1, 1]

    # Calculate the numerator
    num = (q[0, 0] - q[1, 1] + 2 * np.complex(0, q[0, 1]))

    # Calculate the denominator
    den = r2

    if method == 'epsilon':
        den += 2 * np.sqrt(q[0, 0] * q[1, 1] - q[0, 1] ** 2)

    # Calculate the ellipticity/polarisation
    ellip = num / den

    return np.around([ellip.real, ellip.imag], 3)


def get_abt(data):

    q = get_moments(data)

    qq_plus = q[0, 0] + q[1, 1]
    qq_minus = q[0, 0] - q[1, 1]
    root = np.sqrt(qq_minus ** 2 + 4 * q[0, 1] ** 2)

    a = np.around(np.real(np.sqrt(0.5 * (qq_plus + root))), 3)
    b = np.around(np.real(np.sqrt(0.5 * (qq_plus - root))), 3)
    if qq_minus == 0.0:
        theta = -45.0 * np.sign(np.real(q[0, 1]))
    else:
        theta = (np.around(np.real(0.5 * np.arctan(2 * q[0, 1] / qq_minus)) *
                 180. / np.pi, 3))

    if qq_minus > 0.0 and q[0, 1] == 0.0:
        theta += 90.0
    elif qq_minus > 0.0:
        theta -= 90.0 * np.sign(np.real(q[0, 1]))

    return a, b, theta
