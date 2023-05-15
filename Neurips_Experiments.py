import copy
import random
import statistics
import matplotlib.pyplot as plt

import numpy as np
from sklearn.linear_model import LinearRegression

default_beta = 0.5


def simulate_data(mean_x=0, mean_y=0, std_x=1, std_y=1, data_size=10000, beta=default_beta):
    ux = np.random.normal(mean_x, std_x, size=data_size)
    uy = np.random.normal(mean_y, std_y, size=data_size)

    x = ux
    y = beta * x + uy

    return x, y


"""  Missingness in X caused by Y  """


def create_MAR2_missingness(x, y, missing_pct=0.4, Rx=-0.25):
    missing_count = 0
    Rx_val = np.ones(len(x))
    for i in range(len(x)):
        if y[i] <= Rx:
            x[i] = np.nan
            missing_count += 1
            Rx_val[i] = 0
    return x, y


""" MNAR Missingness """


def create_MNAR_missingness(x, y, Rx=0.25):
    missing_count = 0
    mar_count = 0
    Rx_val = np.ones(len(x))
    for i in range(len(x)):
        if x[i] <= Rx:
            x[i] = np.nan
            missing_count += 1
            Rx_val[i] = 0
            mar_count += 1
    return x, y


""" Data is created above. Now, given data, how to find MAR 2 Beta"""


def compute_beta_cca(x, y):
    y_Ry0 = []
    x_Ry0 = []
    for i in range(len(x)):
        if np.isnan(x[i]):
            pass
        else:
            y_Ry0.append(y[i])
            x_Ry0.append(x[i])
    X_temp_np = np.array(x_Ry0)
    XX = X_temp_np.reshape(-1, 1)
    Beta_YX_CCA = LinearRegression().fit(XX, y_Ry0).coef_[0]
    return Beta_YX_CCA


"""Compute bias"""


def compute_bias(beta_computed, beta=default_beta):
    bias = np.abs(beta_computed - beta)
    return bias


def main():
    beta = []
    beta1 = [-45, -35, -25, -15, -5, 0, 5, 15, 25, 35, 45]
    beta2 = [-5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
    beta3 = [-0.5, -0.45, -0.4, -0.35, -0.3, -0.25, -0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35,
               0.4, 0.45, 0.5]
    beta4 = [-0.05, -0.04, -0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03, 0.04, 0.05]
    beta.append(beta1)
    beta.append(beta2)
    beta.append(beta3)
    beta.append(beta4)

    avg_runs = 100

    cca_beta_beta = []
    bias_beta = []

    """ Beta vs Bias """
    for i in range(len(beta)):
        print(i)
        cca_beta = []
        bias = []
        for b in beta[i]:
            bias_avg_cal = 0
            cca_beta_avg_cal = 0
            for j in range(avg_runs):
                x, y = simulate_data(beta=b)
                x_mar2, y_mar2 = create_MAR2_missingness(x, y)
                cca_beta_j = compute_beta_cca(x_mar2, y_mar2)
                bias_avg_cal += compute_bias(cca_beta_j, b)
                cca_beta_avg_cal += cca_beta_j
            cca_beta.append(cca_beta_avg_cal / avg_runs)
            bias.append(bias_avg_cal / avg_runs)
        cca_beta_beta.append(cca_beta)
        bias_beta.append(bias)

    fig1, ax1 = plt.subplots()
    ax1.plot(beta[0], cca_beta_beta[0])
    ax1.set_title('Beta_CCA vs TACE')
    ax1.set_xlabel('TACE')
    ax1.set_ylabel('Beta')
    ax1.legend()
    fig2, ax2 = plt.subplots()
    ax2.plot(beta[0], bias_beta[0])
    ax2.set_title('Bias vs Beta Value')
    ax2.set_xlabel('TACE')
    ax2.set_ylabel('Bias')
    ax2.legend()
    fig3, ax3 = plt.subplots()
    ax3.plot(beta[1], cca_beta_beta[1])
    ax3.set_title('Beta_CCA vs TACE')
    ax3.set_xlabel('TACE')
    ax3.set_ylabel('Beta')
    ax3.legend()
    fig4, ax4 = plt.subplots()
    ax4.plot(beta[1], bias_beta[1])
    ax4.set_title('Bias vs Beta Value')
    ax4.set_xlabel('TACE')
    ax4.set_ylabel('Bias')
    ax4.legend()
    fig5, ax5 = plt.subplots()
    ax5.plot(beta[2], cca_beta_beta[2])
    ax5.set_title('Beta_CCA vs TACE')
    ax5.set_xlabel('TACE')
    ax5.set_ylabel('Beta')
    ax5.legend()
    fig6, ax6 = plt.subplots()
    ax6.plot(beta[2], bias_beta[2])
    ax6.set_title('Bias vs Beta Value')
    ax6.set_xlabel('TACE')
    ax6.set_ylabel('Bias')
    ax6.legend()
    fig7, ax7 = plt.subplots()
    ax7.plot(beta[3], cca_beta_beta[3])
    ax7.set_title('Beta_CCA vs TACE')
    ax7.set_xlabel('TACE')
    ax7.set_ylabel('Beta')
    ax7.legend()
    fig8, ax8 = plt.subplots()
    ax8.plot(beta[3], bias_beta[3])
    ax8.set_title('Bias vs Beta Value')
    ax8.set_xlabel('TACE')
    ax8.set_ylabel('Bias')
    ax8.legend()
    plt.show()

    # """ Missing percentage vs Bias """
    # missing_pct_rx = [-1.64, -1.28, -1.04, -0.84, -0.67, -0.52, -0.39, -0.25, -0.13, 0, 0.13, 0.26, 0.39, 0.53, 0.68,
    #                   0.85, 1.04, 1.29, 1.65]
    # # missing_pct_rx = np.array(missing_pct_rx) * default_beta
    # missing_pct = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.5, 0.55, 0.6, 0.65, 0.70, 0.75, 0.8, 0.85,
    #                0.9, 0.95]
    # avg_runs = 100
    # bias_avg_cal = []
    # cca_beta_avg_cal = []
    # for j in range(avg_runs):
    #     x, y = simulate_data(data_size=10000)
    #     missing_pct_rx_j = np.array(missing_pct_rx) * np.std(y)
    #     bias_missing_pct = []
    #     cca_beta_missing_pct = []
    #     for i in range(len(missing_pct_rx_j)):
    #         x_dash = copy.deepcopy(x)
    #         y_dash = copy.deepcopy(y)
    #         x_mar2, y_mar2 = create_MAR2_missingness(x_dash, y_dash, Rx=missing_pct_rx_j[i])
    #         beta_i = compute_beta_cca(x_mar2, y_mar2)
    #         bias_i = compute_bias(beta_i)
    #         bias_missing_pct.append(bias_i)
    #         cca_beta_missing_pct.append(beta_i)
    #     if len(bias_avg_cal) >= 1:
    #         bias_avg_cal += np.array(bias_missing_pct)
    #         cca_beta_avg_cal += np.array(cca_beta_missing_pct)
    #     else:
    #         bias_avg_cal = np.array(bias_missing_pct)
    #         cca_beta_avg_cal = np.array(cca_beta_missing_pct)
    # bias_avg_cal /= avg_runs
    # cca_beta_avg_cal /= avg_runs
    #
    # fig3, ax3 = plt.subplots()
    # ax3.plot(missing_pct, bias_avg_cal)
    # ax3.set_title('Bias vs Missing Percent')
    # ax3.set_xlabel('Missing Percentage')
    # ax3.set_ylabel('Bias')
    # # ax3.legend()
    # fig4, ax4 = plt.subplots()
    # ax4.plot(missing_pct, cca_beta_avg_cal)
    # ax4.set_title('Beta_CCA vs Missing Percent')
    # ax4.set_xlabel('Missing Percentage')
    # ax4.set_ylabel('Beta_CCA')
    # # ax4.legend()
    # plt.show()


if __name__ == "__main__":
    main()
