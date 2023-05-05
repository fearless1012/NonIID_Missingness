import copy
import random
import statistics
import matplotlib.pyplot as plt

import numpy as np
from sklearn.linear_model import LinearRegression


def simulate_data(mean_x=0, mean_y=0, std_x=1, std_y=1, data_size=10000, beta=10):
    ux = np.random.normal(mean_x, std_x, size=data_size)
    uy = np.random.normal(mean_y, std_y, size=data_size)

    x = ux
    y = beta * x + uy

    # print(ux, uy, X, Y)
    # print(str(statistics.variance(X)) + " " + str(statistics.variance(Y)) + " " + str(statistics.mean(X)) + " " + str(statistics.mean(Y)))

    # print("Simulated data created")
    # x_np = np.array(x)
    # xx = x_np.reshape(-1, 1)
    # regr_yx = LinearRegression().fit(xx, y)
    # print("Beta_YX from full data : " + str(regr_yx.coef_[0]))
    # y_np = np.array(y)
    # yy = y_np.reshape(-1, 1)
    # regr_xy = LinearRegression().fit(yy, x)
    # print("Beta_XY from full data : " + str(regr_xy.coef_[0]))

    return x, y


"""  Missingness in X caused by Y  """


def create_MAR2_missingness(x, y, missing_pct=0.4, Rx=0.4, Rx_intensity=1):
    missing_count = 0
    mar_count = 0
    missing_pct_track = 0
    count = 0
    Rx_val = np.ones(len(x))
    for i in range(len(x)):
        if np.abs(y[i]) < Rx:
            if random.random() < Rx_intensity:
                x[i] = np.nan
                missing_count += 1
                Rx_val[i] = 0
            mar_count += 1
        # else:
        #     if random.random() < (1 - Rx_intensity):
        #         x[i] = np.nan
        #         missing_count += 1
        #         Rx_val[i] = 0
        missing_pct_track = missing_count / len(x)
        # if missing_pct_track >= missing_pct:
        #     break
    print(missing_count)
    print(mar_count)

    Y_temp_np = np.array(y)
    YY = Y_temp_np.reshape(-1, 1)
    Beta_RxY = LinearRegression().fit(YY, Rx_val)

    return x, y, Beta_RxY.coef_[0]


""" Data is created above. Now, given data, how to find MAR 2 Beta"""


def compute_Betayx(x, y):
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
    Beta_YX_CCA = LinearRegression().fit(XX, y_Ry0)
    # print("Beta_YX_CCA: " + str(Beta_YX_CCA.coef_[0]))

    Y_temp_np = np.array(y_Ry0)
    YY = Y_temp_np.reshape(-1, 1)
    Beta_XY_Rx0 = LinearRegression().fit(YY, x_Ry0)
    # print("Beta_XY_Rx0: " + str(Beta_XY_Rx0.coef_[0]))

    var_Y = statistics.variance(y)
    # print("Var(Y) : " + str(var_Y))

    var_X_Rx0 = statistics.variance(x_Ry0)
    # print("Var(X_Rx0) : " + str(var_X_Rx0))

    Rho_Rx0 = statistics.correlation(x_Ry0, y_Ry0)
    # print("Rho_Rx0 : " + str(var_X_Rx0))

    var_X = (var_X_Rx0 * (1 - Rho_Rx0 ** 2)) + ((Beta_XY_Rx0.coef_[0] ** 2) * var_Y)
    # print("var_X : " + str(var_X))

    cov_XY = Beta_XY_Rx0.coef_[0] * var_Y
    # print("cov_YX : " + str(cov_XY))

    Beta_YX = cov_XY / var_X
    # print("Beta_YX : " + str(Beta_YX))

    return Beta_YX, Beta_YX_CCA.coef_[0]


"""Compute bias for both MAR and MCAR"""


def compute_bias(mar_beta, mcar_beta, beta=10):
    # bias = np.abs(beta_yx - Beta_YX_CCA.coef_[0])
    mar_bias = np.abs(mar_beta - beta)
    mcar_bias = np.abs(mcar_beta - beta)
    # print("Bias : ", bias)
    return mar_bias, mcar_bias


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
    missing_pct = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.5, 0.55, 0.6, 0.65, 0.70, 0.75, 0.8]
    Rx = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    Rx_intensity = [0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
    # missing_pct = [0.24, 0.26, 0.30, 0.32, 0.34, 0.36, 0.38, 0.40, 0.42, 0.44, 0.46, 0.48, 0.5, 0.52, 0.54, 0.56,
    # 0.58, 0.6, 0.62, 0.64, 0.66, 0.68, 0.70]

    mar_bias_samplesize = []
    mar_bias_beta = []
    mar_bias_missing_pct = []
    mar_bias_Rx = []
    mar_bias_Rx_intensity = []

    mcar_bias_samplesize = []
    mcar_bias_beta = []
    mcar_bias_missing_pct = []
    mcar_bias_Rx = []
    mcar_bias_Rx_intensity = []

    mar_beta_yx_samplesize = []
    mar_beta_yx_beta = []
    mar_beta_yx_missing_pct = []
    mar_beta_yx_Rx = []
    mar_beta_yx_Rx_intensity = []

    mcar_beta_yx_samplesize = []
    mcar_beta_yx_beta = []
    mcar_beta_yx_missing_pct = []
    mcar_beta_yx_Rx = []
    mcar_beta_yx_Rx_intensity = []

    m_avg = 0
    for j in range(100):
        x, y = simulate_data()
        x_mar2, y_mar2, m = create_MAR2_missingness(x, y)
        m_avg += m

    print(m_avg/100)

    """ Data Size vs Bias """
    # for i in range(len(data_size)):
    #     print(data_size[i])
    #     mar_bias_avg_cal = 0
    #     mar_beta_yx_avg_cal = 0
    #     mcar_beta_yx_avg_cal = 0
    #     mcar_bias_avg_cal = 0
    #     for j in range(100):
    #         x, y = simulate_data(mean_x, mean_y, std_x, std_y, data_size[i])
    #         x_mar2, y_mar2 = create_MAR2_missingness(x, y)
    #         mar_beta_yx, mcar_beta_yx = compute_Betayx(x_mar2, y_mar2)
    #         mar_bias, mcar_bias = compute_bias(mar_beta=mar_beta_yx, mcar_beta=mcar_beta_yx)
    #         mar_beta_yx_avg_cal += mar_beta_yx
    #         mar_bias_avg_cal += mar_bias
    #         mcar_beta_yx_avg_cal += mcar_beta_yx
    #         mcar_bias_avg_cal += mcar_bias
    #     avg_bias_mar = mar_bias_avg_cal / 100
    #     avg_beta_yx_mar = mar_beta_yx_avg_cal / 100
    #     avg_bias_mcar = mcar_bias_avg_cal / 100
    #     avg_beta_yx_mcar = mcar_beta_yx_avg_cal / 100
    #     mar_bias_samplesize.append(avg_bias_mar)
    #     mar_beta_yx_samplesize.append(avg_beta_yx_mar)
    #     mcar_bias_samplesize.append(avg_bias_mcar)
    #     mcar_beta_yx_samplesize.append(avg_beta_yx_mcar)
    #
    # """ Beta vs Bias """
    # for i in range(len(beta)):
    #     print(beta[i])
    #     mar_bias_avg_cal = 0
    #     mar_beta_yx_avg_cal = 0
    #     mcar_beta_yx_avg_cal = 0
    #     mcar_bias_avg_cal = 0
    #     for j in range(100):
    #         x, y = simulate_data(beta=beta[i])
    #         x_mar2, y_mar2 = create_MAR2_missingness(x, y)
    #         mar_beta_yx, mcar_beta_yx = compute_Betayx(x_mar2, y_mar2)
    #         mar_bias, mcar_bias = compute_bias(mar_beta=mar_beta_yx, mcar_beta=mcar_beta_yx)
    #         mar_beta_yx_avg_cal += mar_beta_yx
    #         mar_bias_avg_cal += mar_bias
    #         mcar_beta_yx_avg_cal += mcar_beta_yx
    #         mcar_bias_avg_cal += mcar_bias
    #     avg_bias_mar = mar_bias_avg_cal / 100
    #     avg_beta_yx_mar = mar_beta_yx_avg_cal / 100
    #     avg_bias_mcar = mcar_bias_avg_cal / 100
    #     avg_beta_yx_mcar = mcar_beta_yx_avg_cal / 100
    #     mar_bias_beta.append(avg_bias_mar)
    #     mar_beta_yx_beta.append(avg_beta_yx_mar)
    #     mcar_bias_beta.append(avg_bias_mcar)
    #     mcar_beta_yx_beta.append(avg_beta_yx_mcar)
    #
    # """ Missing percentage vs Bias """
    # x, y = simulate_data()
    # for i in range(len(missing_pct)):
    #     print(missing_pct[i])
    #     mar_bias_avg_cal = 0
    #     mar_beta_yx_avg_cal = 0
    #     mcar_beta_yx_avg_cal = 0
    #     mcar_bias_avg_cal = 0
    #     for j in range(100):
    #         x_dash = copy.deepcopy(x)
    #         y_dash = copy.deepcopy(y)
    #         x_mar2, y_mar2 = create_MAR2_missingness(x_dash, y_dash, missing_pct[i])
    #         # x_mar2, y_mar2 = create_MAR2_missingness(x, y, missing_pct[i])
    #         mar_beta_yx, mcar_beta_yx = compute_Betayx(x_mar2, y_mar2)
    #         mar_bias, mcar_bias = compute_bias(mar_beta=mar_beta_yx, mcar_beta=mcar_beta_yx)
    #         mar_beta_yx_avg_cal += mar_beta_yx
    #         mar_bias_avg_cal += mar_bias
    #         mcar_beta_yx_avg_cal += mcar_beta_yx
    #         mcar_bias_avg_cal += mcar_bias
    #     avg_bias_mar = mar_bias_avg_cal / 100
    #     avg_beta_yx_mar = mar_beta_yx_avg_cal / 100
    #     avg_bias_mcar = mcar_bias_avg_cal / 100
    #     avg_beta_yx_mcar = mcar_beta_yx_avg_cal / 100
    #     mar_bias_missing_pct.append(avg_bias_mar)
    #     mar_beta_yx_missing_pct.append(avg_beta_yx_mar)
    #     mcar_bias_missing_pct.append(avg_bias_mcar)
    #     mcar_beta_yx_missing_pct.append(avg_beta_yx_mcar)
    #
    # """ Rx vs Bias """
    # x, y = simulate_data()
    # for i in range(len(Rx)):
    #     print(Rx[i])
    #     mar_bias_avg_cal = 0
    #     mar_beta_yx_avg_cal = 0
    #     mcar_beta_yx_avg_cal = 0
    #     mcar_bias_avg_cal = 0
    #     for j in range(100):
    #         x_dash = copy.deepcopy(x)
    #         y_dash = copy.deepcopy(y)
    #         x_mar2, y_mar2 = create_MAR2_missingness(x_dash, y_dash, missing_pct=0.1, Rx=Rx[i])
    #         # x_mar2, y_mar2 = create_MAR2_missingness(x, y,  missing_pct=0.1, Rx=Rx[i])
    #         mar_beta_yx, mcar_beta_yx = compute_Betayx(x_mar2, y_mar2)
    #         mar_bias, mcar_bias = compute_bias(mar_beta=mar_beta_yx, mcar_beta=mcar_beta_yx)
    #         mar_beta_yx_avg_cal += mar_beta_yx
    #         mar_bias_avg_cal += mar_bias
    #         mcar_beta_yx_avg_cal += mcar_beta_yx
    #         mcar_bias_avg_cal += mcar_bias
    #     avg_bias_mar = mar_bias_avg_cal / 100
    #     avg_beta_yx_mar = mar_beta_yx_avg_cal / 100
    #     avg_bias_mcar = mcar_bias_avg_cal / 100
    #     avg_beta_yx_mcar = mcar_beta_yx_avg_cal / 100
    #     mar_bias_Rx.append(avg_bias_mar)
    #     mar_beta_yx_Rx.append(avg_beta_yx_mar)
    #     mcar_bias_Rx.append(avg_bias_mcar)
    #     mcar_beta_yx_Rx.append(avg_beta_yx_mcar)
    #
    # """ Rx effect(m) vs Bias """
    # x, y = simulate_data()
    # for i in range(len(Rx_intensity)):
    #     print(Rx_intensity[i])
    #     mar_bias_avg_cal = 0
    #     mar_beta_yx_avg_cal = 0
    #     mcar_beta_yx_avg_cal = 0
    #     mcar_bias_avg_cal = 0
    #     for j in range(100):
    #         x_dash = copy.deepcopy(x)
    #         y_dash = copy.deepcopy(y)
    #         x_mar2, y_mar2 = create_MAR2_missingness(x_dash, y_dash, missing_pct=0.1, Rx=0.4,
    #                                                  Rx_intensity=Rx_intensity[i])
    #         # x_mar2, y_mar2 = create_MAR2_missingness(x, y, missing_pct=0.1, Rx=0.4,
    #         #                                                      Rx_intensity=Rx_intensity[i])
    #         mar_beta_yx, mcar_beta_yx = compute_Betayx(x_mar2, y_mar2)
    #         mar_bias, mcar_bias = compute_bias(mar_beta=mar_beta_yx, mcar_beta=mcar_beta_yx)
    #         mar_beta_yx_avg_cal += mar_beta_yx
    #         mar_bias_avg_cal += mar_bias
    #         mcar_beta_yx_avg_cal += mcar_beta_yx
    #         mcar_bias_avg_cal += mcar_bias
    #     avg_bias_mar = mar_bias_avg_cal / 100
    #     avg_beta_yx_mar = mar_beta_yx_avg_cal / 100
    #     avg_bias_mcar = mcar_bias_avg_cal / 100
    #     avg_beta_yx_mcar = mcar_beta_yx_avg_cal / 100
    #     mar_bias_Rx_intensity.append(avg_bias_mar)
    #     mar_beta_yx_Rx_intensity.append(avg_beta_yx_mar)
    #     mcar_bias_Rx_intensity.append(avg_bias_mcar)
    #     mcar_beta_yx_Rx_intensity.append(avg_beta_yx_mcar)

    # fig1, ax1 = plt.subplots()
    # ax1.plot(data_size, mar_bias_samplesize, label='MAR')
    # ax1.plot(data_size, mcar_bias_samplesize, label='MCAR')
    # ax1.set_title('Bias Vs Sample Size')
    # ax1.set_xlabel('Sample Size')
    # ax1.set_ylabel('Bias')
    # ax1.legend()
    # fig2, ax2 = plt.subplots()
    # ax2.plot(beta, mar_bias_beta, label='MAR')
    # ax2.plot(beta, mcar_bias_beta, label='MCAR')
    # ax2.set_title('Bias Vs Beta Value')
    # ax2.set_xlabel('Beta Value')
    # ax2.set_ylabel('Bias')
    # ax2.legend()
    # fig3, ax3 = plt.subplots()
    # ax3.plot(missing_pct, mar_bias_missing_pct, label='MAR')
    # ax3.plot(missing_pct, mcar_bias_missing_pct, label='MCAR')
    # ax3.set_title('Bias Vs Missing Percent vs ')
    # ax3.set_xlabel('Missing Percentage')
    # ax3.set_ylabel('Bias')
    # ax3.legend()
    # fig4, ax4 = plt.subplots()
    # ax4.plot(Rx, mar_bias_Rx, label='MAR')
    # ax4.plot(Rx, mcar_bias_Rx, label='MCAR')
    # ax4.set_title('Rx vs Bias')
    # ax4.set_xlabel('Rx')
    # ax4.set_ylabel('Bias')
    # ax4.legend()
    # fig5, ax5 = plt.subplots()
    # ax5.plot(Rx_intensity, mar_bias_Rx_intensity, label='MAR')
    # ax5.plot(Rx_intensity, mcar_bias_Rx_intensity, label='MCAR')
    # ax5.set_title('Rx Effect vs Bias')
    # ax5.set_xlabel('Y effect on RX')
    # ax5.set_ylabel('Bias')
    # ax5.legend()
    # fig6, ax6 = plt.subplots()
    # ax6.plot(data_size, mar_beta_yx_samplesize, label='MAR')
    # ax6.plot(data_size, mcar_beta_yx_samplesize, label='MCAR')
    # ax6.set_title('Sample Size vs Beta_yx')
    # ax6.set_xlabel('Sample Size')
    # ax6.set_ylabel('Beta_yx')
    # ax6.legend()
    # fig7, ax7 = plt.subplots()
    # ax7.plot(beta, mar_beta_yx_beta, label='MAR')
    # ax7.plot(beta, mcar_beta_yx_beta, label='MCAR')
    # ax7.set_title('Beta Value vs Beta_yx')
    # ax7.set_xlabel('Beta Value')
    # ax7.set_ylabel('Beta_yx')
    # ax7.legend()
    # # ax7.set_xticks([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 30, 32, 34, 36, 38, 40])
    # fig8, ax8 = plt.subplots()
    # ax8.plot(missing_pct, mar_beta_yx_missing_pct, label='MAR')
    # ax8.plot(missing_pct, mcar_beta_yx_missing_pct, label='MCAR')
    # ax8.set_title('Missing Percent vs Beta_yx')
    # ax8.set_xlabel('Missing Percent')
    # ax8.set_ylabel('Beta_yx')
    # ax8.legend()
    # fig9, ax9 = plt.subplots()
    # ax9.plot(Rx, mar_beta_yx_Rx, label='MAR')
    # ax9.plot(Rx, mcar_beta_yx_Rx, label='MCAR')
    # ax9.set_title('Rx vs Beta_yx')
    # ax9.set_xlabel('Rx')
    # ax9.set_ylabel('Beta_yx')
    # ax9.legend()
    # fig10, ax10 = plt.subplots()
    # ax10.plot(Rx_intensity, mar_beta_yx_Rx_intensity, label='MAR')
    # ax10.plot(Rx_intensity, mcar_beta_yx_Rx_intensity, label='MCAR')
    # ax10.set_title('Rx Effect vs Beta_yx')
    # ax10.set_xlabel('Y effect on Rx')
    # ax10.set_ylabel('Beta_yx')
    # ax10.legend()
    # plt.show()


if __name__ == "__main__":
    main()
