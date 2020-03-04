import matplotlib.pyplot as plt
import numpy as np


def main():
    n=1
    norm_reward_hill = lambda x, k: x**n / (k**n + x**n)


    k_d = 500

    vals_1 = []
    for i in np.linspace(0, 2000):
        vals_1.append(norm_reward_hill(i, k_d))

    plt.plot(np.linspace(0, 2000), vals_1)
    plt.show()


if __name__ == "__main__":
    main()
