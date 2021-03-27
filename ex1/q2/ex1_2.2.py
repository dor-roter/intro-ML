import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({
    'font.size': 12,
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

# --------------------------------------------

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


def generate_data(p, seq, tosses):
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
    plt.show()


def plot_bounds(data):
    """
    plot upper bound using chebyshev and hoeffding, and the actual satisfied sequences count
    over the entire dataset
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
        percentage = np.sum((deltas >= eps), axis=0) / SEQUENCES

        ax.plot(M, [chebyshev(i, eps) for i in M], label='chebyshev')
        ax.plot(M, [hoeffding(i, eps) for i in M], label='hoeffding')
        ax.scatter(M, percentage, marker='.', label="percentage", edgecolors='red', alpha=0.5)
        ax.legend()
        ax.set_xlabel("m")
        ax.set_ylabel(r"satisfied sequences (\%) / upper bound ")
        ax.set_title(r'$|\overline{X}_m-\mathbb{E}[X]|\geq ' + str(eps) + '$', fontsize=16)

    fig.delaxes(axs[2, 1])
    fig.suptitle(r"Upper\&actual bounds", fontsize=16)
    plt.show()


def main():
    """
    run and generate plots
    """
    data = generate_data(P, SEQUENCES, TOSSES)
    plot_estimation(data, ESTIMATION_SET_SIZE)
    plot_bounds(data)


if __name__ == '__main__':
    main()
