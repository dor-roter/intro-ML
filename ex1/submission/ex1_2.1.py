import gausssian_helper
import numpy as np
import matplotlib.pyplot as plt

# make prints a little easier to read
np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)

# - constants -
# -------------

SIZE = 50000
DIM = 3
X = 0
Y = 1
Z = 2

# data's covariance
cov_matrix = np.eye(DIM)
# data's mean
mean_matrix = np.zeros(DIM)

# create the required scaling matrix
S = np.diag([.1, .5, 2])
# generate a random orthogonal matrix
R = gausssian_helper.get_orthogonal_matrix(3)

# conditional projections range
Z_MIN = -.4
Z_MAX = .1

def plot_xy_histogram(data):
    """
    plot the data as X axis and Y axis histograms
    :param data: dataset with x and y axis
    """
    fig, axs = plt.subplots(1, 2, figsize=(10, 3))
    axs[0].hist(data[X], bins='auto')
    axs[0].set_title("X probability density")
    axs[0].set_xlim(-6, 6)
    axs[1].hist(data[Y], bins='auto')
    axs[1].set_title("Y probability density")
    axs[1].set_xlim(-6, 6)
    fig.suptitle("PDF of X,Y")


def generate_dataset(cov, mean, size):
    """
    generate a gaussian distributed dataset using the passed
    parameters, the plot it and print its covariance.
    :param cov: covariance matrix to use
    :param mean: mean matrix to use
    :param size: size of the dataset
    :return: the generated dataset
    """
    data = np.random.multivariate_normal(mean, cov, size).T

    # plot the dataset for visualization
    gausssian_helper.plot_3d(data, "Dataset Plot")

    # make sure covariance matrices are ok
    print("Analytical covariance matrix:")
    print(cov_matrix)

    print()
    print("Numerical covariance matrix:")
    print(np.cov(data))

    return data


def apply_scaling_transformation(data):
    """
    apply the scaling matrix S to the dataset, plot it and print its covariance.
    :param data: the dataset to transform
    :return: the transformed dataset
    """
    global cov_matrix

    print("transformation: ")
    print(S)
    print()

    # apply the transformation
    data = S @ data

    # plot
    gausssian_helper.plot_3d(data, "Scaled dataset")

    # looking at the covariance matrix:
    print("Analytical covariance matrix:")
    cov_matrix = S @ cov_matrix @ S.T
    print(cov_matrix)

    print()
    print("Numerical covariance matrix:")
    print(np.cov(data))

    return data


def apply_random_rotation_transformation(data):
    """
    apply the rotation matrix R to the dataset, plot it and print its covariance.
    :param data: the dataset to transform
    :return: the transformed dataset
    """
    global cov_matrix

    print("Rotation matrix:")
    print(R)
    print()

    # apply the transformation
    data = R @ data

    # plot
    gausssian_helper.plot_3d(data, "Randomly rotated dataset")

    # look at the new samples' variance matrix:
    print("Analytical covariance matrix:")
    cov_matrix = S @ cov_matrix @ S.T
    print(cov_matrix)

    print()
    print("Numerical covariance matrix:")
    print(np.cov(data))

    return data


def plot_xy_projection(data):
    """
    plot the XY projection of the data
    :param data: the dataset to plot
    """
    # x,y projection for original data
    gausssian_helper.plot_2d(data, "x,y projection for marginal distribution")
    plot_xy_histogram(data)


def plot_xy_conditional_projection(data):
    """
    plot the XY projection of the data under a condition
    :param data: the dataset to plot
    """
    # get the relevant data points from the dataset
    conditional_points = np.reshape(data[:, np.where(np.logical_and(Z_MIN < data[Z], data[Z] < Z_MAX))], (3, -1))

    # x,y conditional projection
    gausssian_helper.plot_2d(conditional_points, "x,y conditional projection")
    plot_xy_histogram(data)


def main():
    """
    run and generate the plots
    """
    # (11)
    data = generate_dataset(cov_matrix, mean_matrix, SIZE)

    # (12)
    data = apply_scaling_transformation(data)
    # (13)
    data = apply_random_rotation_transformation(data)
    # (14)
    plot_xy_projection(data)
    # (15)
    plot_xy_conditional_projection(data)

    plt.show()


if __name__ == '__main__':
    main()
