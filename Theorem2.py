import copy
import random
import statistics
import matplotlib.pyplot as plt

import numpy as np
from sklearn.linear_model import LinearRegression


def simulate_data(mean_x=0, mean_y=0, mean_z=0, std_x=1, std_y=1, std_z=1, data_size=10000, beta=2, alpha=0.5):
    ux = np.random.normal(mean_x, std_x, size=data_size)
    uy = np.random.normal(mean_y, std_y, size=data_size)
    uz = np.random.normal(mean_z, std_z, size=data_size)

    x = ux
    y = beta * x + uy
    z = alpha * y + uz

    return x, y, z


"""  Missingness in X and Y caused by Z  """


def create_missingness(x, y, z, missing_pct=0.4, Rx_z=-0.25):
    missing_count = 0
    Rx_val = np.ones(len(x))
    Ry_val = np.ones(len(x))
    for i in range(len(x)):
        if z[i] <= Rx_z:
            x[i] = np.nan
            y[i] = np.nan
            missing_count += 1
            Rx_val[i] = 0
            Ry_val[i] = 0
    # print(missing_pct, missing_count / len(x), Rx_z)
    return x, y, z


""" Data is created above. Now, given data, how to find MAR 2 Beta"""


def compute_Betayx(x, y, z):
    y_Ry0 = []
    x_Ry0 = []
    z_Ry0 = []
    for i in range(len(x)):
        if np.isnan(x[i]) or np.isnan(y[i]):
            pass
        else:
            y_Ry0.append(y[i])
            x_Ry0.append(x[i])
            z_Ry0.append(z[i])

    X_temp_np = np.array(x_Ry0)
    XX = X_temp_np.reshape(-1, 1)
    Beta_YX_CCA = LinearRegression().fit(XX, y_Ry0).coef_[0]
    Beta_ZX_CCA = LinearRegression().fit(XX, z_Ry0).coef_[0]

    Z_temp_np = np.array(z_Ry0)
    ZZ = Z_temp_np.reshape(-1, 1)
    Beta_YZ_CCA = LinearRegression().fit(ZZ, y_Ry0).coef_[0]
    Beta_XZ_CCA = LinearRegression().fit(ZZ, y_Ry0).coef_[0]

    Beta_YX_givenZ = (Beta_YX_CCA - (Beta_YZ_CCA * Beta_ZX_CCA)) / (1 - (Beta_ZX_CCA * Beta_XZ_CCA))

    return Beta_YX_givenZ, Beta_YX_CCA


"""Compute bias for both MAR and MCAR"""


def compute_bias(beta_cca, beta=2, alpha=0.5):
    cca_bias = np.abs(beta_cca - beta)
    return cca_bias


def main():
    mean_x, std_x = 0, 1
    mean_y, std_y = 0, 1
    # data_size = [100, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500,
    # 8000, 8500, 9000, 9500, 10000, 10500, 11000, 11500, 12000, 12500, 13000, 13500, 14000, 14500, 15000]
    data_size = [100, 500, 1000, 2000, 5000, 7000,
                 10000]  # , 20000, 30000, 40000, 50000]  # , 60000, 70000, 80000, 90000,
    # 100000]
    beta = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 30, 32, 34, 36, 38, 40]
    # missing_pct = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24, 0.26, 0.30, 0.32, 0.34,
    # 0.36, 0.38, 0.40, 0.42, 0.44, 0.46, 0.48, 0.5, 0.52, 0.54, 0.56, 0.58, 0.6, 0.62, 0.64, 0.66, 0.68, 0.70]
    missing_pct = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.5, 0.55, 0.6, 0.65, 0.70, 0.75, 0.8, 0.85,
                   0.9, 0.95]
    Rx = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    Rx_intensity = [0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
    # missing_pct = [0.24, 0.26, 0.30, 0.32, 0.34, 0.36, 0.38, 0.40, 0.42, 0.44, 0.46, 0.48, 0.5, 0.52, 0.54, 0.56,
    # 0.58, 0.6, 0.62, 0.64, 0.66, 0.68, 0.70]

    """ Missing percentage vs Bias """

    missing_pct_rx = [-1.64, -1.28, -1.04, -0.84, -0.67, -0.52, -0.39, -0.25, -0.13, 0, 0.13, 0.26, 0.39, 0.53, 0.68,
                      0.85, 1.04, 1.29, 1.65]
    # missing_pct_rx = [-16.4, -12.8, -10.4, -8.4, -6.7, -5.2, -3.9, -2.5, -1.3, 0, 1.3, 2.6, 3.9, 5.3, 6.8,
    #                   8.5, 10.4, 12.9, 16.5]


    avg_runs = 100
    bias_avg_cal = []
    cca_beta_avg_cal = []

    for j in range(avg_runs):
        x, y, z = simulate_data()
        missing_pct_rx_j = np.array(missing_pct_rx) * np.std(z)
        cca_bias_missing_pct = []
        Beta_YX_CCA_missing_pct = []
        for i in range(len(missing_pct)):
            x_dash = copy.deepcopy(x)
            y_dash = copy.deepcopy(y)
            z_dash = copy.deepcopy(z)

            x_m, y_m, z_m = create_missingness(x_dash, y_dash, z_dash, Rx_z=missing_pct_rx_j[i],
                                               missing_pct=missing_pct[i])
            Beta_YX_givenZ, Beta_YX_CCA = compute_Betayx(x_m, y_m, z_m)
            cca_bias = compute_bias(Beta_YX_CCA)

            cca_bias_missing_pct.append(cca_bias)
            Beta_YX_CCA_missing_pct.append(Beta_YX_CCA)
        if len(bias_avg_cal) >= 1:
            # print(bias_avg_cal)
            bias_avg_cal += np.array(cca_bias_missing_pct)
            cca_beta_avg_cal += np.array(Beta_YX_CCA)
            # print(bias_avg_cal)
            # print(cca_bias_missing_pct)
        else:
            # print(cca_bias_missing_pct)
            bias_avg_cal = np.array(cca_bias_missing_pct)
            cca_beta_avg_cal = np.array(Beta_YX_CCA_missing_pct)

    bias_avg_cal /= avg_runs
    cca_beta_avg_cal /= avg_runs

    fig1, ax1 = plt.subplots()
    ax1.plot(missing_pct, bias_avg_cal, label='CCA')
    ax1.set_title('Bias vs Missing Percent')
    ax1.set_xlabel('Missing Percentage')
    ax1.set_ylabel('Bias')

    fig2, ax2 = plt.subplots()
    ax2.plot(missing_pct, cca_beta_avg_cal, label='CCA')
    ax2.set_title('Beta_yx vs Missing Percent')
    ax2.set_xlabel('Missing Percent')
    ax2.set_ylabel('Beta_yx')

    plt.show()


if __name__ == "__main__":
    main()
