import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import qr

# make prints a little easier to read
np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)

# matplotlib settings
plt.rcParams.update({
    'font.size': 12,
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

# -------------------------- helpers -------------------------
# ------------------------------------------------------------

def get_orthogonal_matrix(dim):
    H = np.random.randn(dim, dim)
    Q, R = qr(H)
    return Q


def plot_3d(x_y_z, title=""):
    """
    plot points in 3D
    :param title:
    :param x_y_z: the points. numpy array with shape: 3 X num_samples (first dimension for x, y, z
    coordinate)
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_y_z[0], x_y_z[1], x_y_z[2], s=1, marker='.', depthshade=False)

    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-5, 5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(title)


def plot_2d(x_y, title=""):
    """
    plot points in 2D
    :param title:
    :param x_y: the points. numpy array with shape: 2 X num_samples (first dimension for x, y
    coordinate)
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x_y[0], x_y[1], s=1, marker='.')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)

# ------------------------- constants ------------------------
# ------------------------------------------------------------

# ========================
# ---------- q1 ----------
# ========================

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
R = get_orthogonal_matrix(3)

# conditional projections range
Z_MIN = -.4
Z_MAX = .1

# ========================
# ---------- q2 ----------
# ========================

# the dataset: 100000 sequences of 1000 coin tosses
SEQUENCES = 100000
TOSSES = 1000

# p-coin
P = 0.25

# epsilon values for the next questions
EPSILONS = [0.5, 0.25, 0.1, 0.01, 0.001]
EPS_LEN = len(EPSILONS)

# used as the x axis for the plots (1,...,TOSSES)
M = np.arange(TOSSES) + 1

# first figure size
ESTIMATION_SET_SIZE = 5


# ------------------------- functions ------------------------
# ------------------------------------------------------------

# ========================
# ---------- q1 ----------
# ========================

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


def generate_dataset1(cov, mean, size):
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
    plot_3d(data, "Dataset Plot")

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
    plot_3d(data, "Scaled dataset")

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
    plot_3d(data, "Randomly rotated dataset")

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
    plot_2d(data, "x,y projection for marginal distribution")
    plot_xy_histogram(data)


def plot_xy_conditional_projection(data):
    """
    plot the XY projection of the data under a condition
    :param data: the dataset to plot
    """
    # get the relevant data points from the dataset
    conditional_points = np.reshape(data[:, np.where(np.logical_and(Z_MIN < data[Z], data[Z] < Z_MAX))], (3, -1))

    # x,y conditional projection
    plot_2d(conditional_points, "x,y conditional projection")
    plot_xy_histogram(data)


# ========================
# ---------- q2 ----------
# ========================
def generate_data2(p, seq, tosses):
    """
    generate a coin toss dataset as described in q16 using the passed parameters
    :param p: the coin's bias
    :param seq: number of sequences
    :param tosses: number of tosses in each sequence
    :return: the generated dataset
    """
    data = np.random.binomial(1, p, (seq, tosses))
    print(f"shape: {data.shape}")
    print(data)
    return data


def plot_estimation(data, size):
    """
    plot the mean unbiased estimation: Xm
    :param data: the dataset
    :param size: number of sequences to plot
    """
    estimations = data[:size, :].cumsum(axis=1) / M

    # plot estimations for every sequence of the first ESTIMATION_SIZE sequences
    plt.figure()
    for estimation in estimations:
        plt.plot(M, estimation)

    plt.title(r'$\overline{X}_m$ estimate')
    plt.xlabel('m')
    plt.ylabel(r'$\overline{X}_m$')


def plot_bounds(data, show_percentage=True):
    """
    plot upper bound using chebyshev and hoeffding, and the actual satisfied sequences count
    over the entire dataset
    :param show_percentage: add (c) or show just (b)
    :param data: the dataset
    """

    def chebyshev(n, e):
        return min(1 / (4 * n * (e ** 2)), 1)

    def hoeffding(n, e):
        return min(2 * np.exp(-2 * n * (e ** 2)), 1)

    deltas = np.abs((data.cumsum(axis=1) / M) - P)

    fig, axs = plt.subplots(3, 2, figsize=(15, 15), edgecolor="#e4e6e8", constrained_layout=True)
    for j, eps in enumerate(EPSILONS):
        ax = axs[int(j / 2), (j % 2)]

        ax.plot(M, [chebyshev(i, eps) for i in M], label='chebyshev')
        ax.plot(M, [hoeffding(i, eps) for i in M], label='hoeffding')

        if show_percentage:
            percentage = np.sum((deltas >= eps), axis=0) / SEQUENCES
            ax.scatter(M, percentage, marker='.', label="percentage", edgecolors='red', alpha=0.5)

        ax.legend()
        ax.set_xlabel("m")
        ax.set_ylabel(r"satisfied sequences (\%) / upper bound ")
        ax.set_title(r'$|\overline{X}_m-\mathbb{E}[X]|\geq ' + str(eps) + '$', fontsize=16)

    fig.delaxes(axs[2, 1])
    fig.suptitle(r"Upper\&actual bounds", fontsize=16)


def main():
    """
    run and generate the plots
    """
    # (11)
    data = generate_dataset1(cov_matrix, mean_matrix, SIZE)

    # (12)
    data = apply_scaling_transformation(data)
    # (13)
    data = apply_random_rotation_transformation(data)
    # (14)
    plot_xy_projection(data)
    # (15)
    plot_xy_conditional_projection(data)

    # (16)
    data = generate_data2(P, SEQUENCES, TOSSES)
    # (16a)
    plot_estimation(data, ESTIMATION_SET_SIZE)
    # (16b & 16c)
    plot_bounds(data)

    plt.show()


if __name__ == '__main__':
    main()
