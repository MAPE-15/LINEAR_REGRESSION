# MULTIPLE LINEAR REGRESSION EVALUATOR WITHOUT USING SCIKIT

# FINISHED !!!

# THIS WILL MAKE EVERYTHING, YOU CAN CHOOSE IF YOU WANT TO MAKE AN EVALUATION OR TO BUILD A MODEL


# importing this because I want my line to be a little bit more smoother when making CI and PI Bands plots
from scipy.interpolate import make_interp_spline

# I know that those calculation would be quicker calculated with this module in just few lines
# But I wanted to understand those calculations a little more so I did it more manually
# I only used this when I wanted to make A regression line in 3D PLOT that's all
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use('ggplot')


# EXAMPLE NUMBERS AND VALUES !!!

# x1 = 89, 66, 78, 111, 44, 77, 80, 66, 109, 76
# x2 = 4, 1, 3, 6, 1, 3, 3, 2, 5, 3
# x3 = 3.84, 3.19, 3.78, 3.89, 3.57, 3.57, 3.03, 3.51, 3.54, 3.25
# y1 = 7, 5.4, 6.6, 7.4, 4.8, 6.4, 7, 5.6, 7.3, 6.4

# x1 = 4, 7, 2, 1, 3, 7, 8
# x2 = 5, 2, 6, 9, 4, 3, 2
# x3 = 4, 3, 4, 6, 5, 4, 5
# y2 = 6, 11, 4, 3, 5, 9, 10

# x1 = 4, 4, 4, 5, 5, 8, 9, 5, 14, 9
# x2 = 5, 5, 9, 8, 5, 10, 7, 14, 6, 9
# x3 = 4, 3, 8, 7, 9, 8, 13, 14, 12, 9
# y3 = 5, 4, 9, 3, 5, 5, 8, 5, 5, 12

# x1 = 2, 3, 3, 4, 4, 6, 8, 9, 9, 9
# x2 = 2, 2, 3, 3, 2, 3, 5, 6, 6, 7
# x3 = 12, 14, 14, 13, 8, 8, 9, 14, 11, 7
# y4 = 23, 24, 15, 9, 14, 17, 22, 26, 34, 35


print('''
!!! WELCOME TO MULTIPLE LINEAR REGRESSION EVALUATION !!!

This program is for those who want to either evaluate their model and make a deep evaluation or to build a model right off the bat
You can pass at least 1 independent variable (x) and at most 3 independent variables (x's), because more than 3 IVs causes over-fitting and much more bigger problems

If you choose an evaluation, this program evaluates:

- 1. Regression Equation --> ŷ = b0 + b1 * x1 + ... + b3 * x3

- 2. Standard Error (S) --> The smaller the S the better, it's the average distance between regression line and observed values

- 3. RSquared (R ** 2) --> The higher the RSquared the better, what percentage of total variance (SST) covers our regression model

- 4. RSquared Adjusted (R ** 2 adj.) --> As closest to RSquared the better, and the higher the better

- 5. RSquared Predicted (R ** 2 pred.) --> As closest to RSquared Adjusted the better, and the higher the better (Works only with 1 IV yet)

- 6. Correlation (r) --> When IV vs DV Scatter Plot, the higher the correlation between independent variable and independent variable the better
                     --> When IV vs IV Scatter Plot, the smaller the correlation between independent variables the better
                     --> An interval is <-1.00; 1.00>, -1.00 indicates the worst correlation, and 1.00 indicates the best correlation
                     
                     
If you choose to build an model:

- 1. Regression Equation --> ŷ = b0 + b1 * x1 + ... + b3 * x3

- 2. Standard Error (S) --> The smaller the S the better, it's the average distance between regression line and observed values

- 3. RSquared (R ** 2) --> The higher the RSquared the better, what percentage of total variance (SST) covers our regression model

- 4. RSquared Adjusted (R ** 2 adj.) --> As closest to RSquared the better, and the higher the better

- 5. RSquared Predicted (R ** 2 pred.) --> As closest to RSquared Adjusted the better, and the higher the better (Works only with 1 IV yet)

- 6. SST, SSE, SSR --> SST = Sum of Squares Total,
                   --> SSE = Sum of Squared due to Error,
                   --> SSR = Sum of Squared due to Regression
                   --> As closest SSE is to SST the better
                   --> As lowest SSR is the better

- 7. MSE --> Mean Squared Error, the lower it is, the better

- 8. CI Bands --> Confidence Interval Mean Value

- 9. PI Bands --> Prediction Interval Individual Value

''')


def multiple_regression():

    print('')
    ask = input('So do you want to 1st make an Evaluation or just get right into the Linear Regression? evaluate/LR: ').upper()

    if ask == 'EVALUATE':

        def evaluation():
            global how_many_IVs, X_arr, Y_arr, y_values

            # !!! CHECK FOR ANY EXCEPTION OR ERROR !!!
            # ---------------------------------------------------------------------------------------------------------------------
            print('')
            try:
                how_many_IVs = int(input('How many independent variables you want to use for an evaluation? (max 3 allowed !!!): '))

            except ValueError:
                print('')
                print('TRY AGAIN !!!')
                evaluation()

            if (how_many_IVs >= 1) and (how_many_IVs <= 3):
                pass

            else:
                print('')
                print('Minimum 1 IV and maximum 3 IVs allowed, try again !!!')
                evaluation()


            try:

                IVs_values = []
                for i in range(1, how_many_IVs + 1):
                    IVs = input('Independent variable number ' + str(i) + ' (x' + str(i) + ' values): ').split(', ')
                    IVs_values.append(IVs)

                IVs_values.insert(0, [float(i) - (i - 1) for i in range(1, len(IVs_values[0]) + 1)])

                y_values = input('Dependent variable values (y values): ').split(', ')

                X_arr = np.array(IVs_values)
                X_arr = X_arr.astype(np.float)
                Y_arr = np.array([float(i) for i in y_values])


            except ValueError:
                print('')
                print('Oops something went wrong, try again !!!')
                evaluation()


            if how_many_IVs == 1:
                if len(X_arr[1, :]) != len(Y_arr):
                    print('')
                    print('For each variable must be same amount of numbers, try again !!!')
                    evaluation()
                else:
                    pass

                if len(Y_arr) <= 4:
                    print('')
                    print('You must pass more amount of values than 4, try again !!!')
                    evaluation()


            elif how_many_IVs == 2:
                if len(X_arr[1, :]) != len(Y_arr) or len(X_arr[2, :]) != len(Y_arr) or len(X_arr[1, :]) != len(X_arr[2, :]):
                    print('')
                    print('For each variable must be same amount of numbers, try again !!!')
                    evaluation()
                else:
                    pass

                if len(Y_arr) <= 4:
                    print('')
                    print('You must pass more amount of values than 4, try again !!!')
                    evaluation()

            elif how_many_IVs == 3:
                if len(X_arr[1, :]) != len(Y_arr) or len(X_arr[2, :]) != len(Y_arr) or len(X_arr[3, :]) != len(Y_arr) or len(X_arr[1, :]) != len(X_arr[2, :]) or len(X_arr[1, :]) != len(X_arr[3, :]) or len(X_arr[2, :]) != len(X_arr[3, :]):
                    print('')
                    print('For each variable must be same amount of numbers, try again !!!')
                    evaluation()
                else:
                    pass

                if len(Y_arr) <= 4:
                    print('')
                    print('You must pass more amount of values than 4, try again !!!')
                    evaluation()

            # -------------------------------------------------------------------------------------------------------------------------------



            # MORE INFO CALCULATIONS !!!

            # ------------------------START---------------------------- #
            x = X_arr[1:]

            x_mean = x.mean(axis=1).reshape(how_many_IVs, 1)
            y_mean = Y_arr.mean()

            # (Xi - X bar)
            x_residual = x - x_mean
            # (Yi - Y bar)
            y_residual = Y_arr - y_mean

            # (Xi - X bar) ** 2
            x_residual_squared = np.power(x_residual, 2)
            # (Yi - Y bar) ** 2
            y_residual_squared = y_residual ** 2

            # E (summation symbol) --> E(Xi - X bar) ** 2
            x_residual_squared_SUM = np.sum(x_residual_squared, axis=1).reshape(how_many_IVs, 1)
            # E(Yi - Y bar) ** 2 --> SST --> Sum of squares Total; Total Variation
            SST = np.round(np.sum(y_residual_squared), 5)


            # REGRESSION COEFFICIENTS EQUATION CALCULATIONS !!!

            # ------------------------START---------------------------- #


            # IV vs DV; coefficients
            # Each IV vs DV
            # ------------------------------------------------------------------------------------------------------------------

            print('')
            print('INDEPENDENT VARIABLE (X) VS DEPENDENT VARIABLE (Y) EQUATION(S)')
            print('↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓')


            which_variable = 1
            for i in range(how_many_IVs):

                # !!! COEFFICIENTS CALCULATIONS !!!
                # --------------------------------------------------------------------------------------------------------------
                X_arr_transposed = X_arr[[0, which_variable], :].transpose()

                X_arr_matrix_multiplication = np.matmul(X_arr[[0, which_variable], :], X_arr_transposed)
                X_arr_matrix_multiplication_inversion = np.linalg.pinv(X_arr_matrix_multiplication)

                X_arr_Y_arr_matrix_multiplication = np.matmul(X_arr[[0, which_variable], :], Y_arr).reshape(2, 1)

                coefficients = np.round(np.matmul(X_arr_matrix_multiplication_inversion, X_arr_Y_arr_matrix_multiplication), 5)

                b0_intercept = coefficients[0, 0]
                b1_slopes = coefficients[1, 0]
                # --------------------------------------------------------------------------------------------------------------

                # predicted values
                y_hat = b0_intercept + b1_slopes * X_arr[which_variable, :]
                # (Yi - ŷ)
                y_hat_residuals = Y_arr - y_hat
                # (Yi - ŷ) ** 2
                y_hat_residuals_squared = y_hat_residuals ** 2
                # E(Yi - ŷ) ** 2 --> SSE --> Sum of squares due to Error
                SSE = np.round(np.sum(y_hat_residuals_squared), 5)
                # SST = SSR + SSE --> SSR = SST - SSE --> Sum of Squares regression
                SSR = SST - SSE

                # variance (s ** 2)/MSE (Mean Squared Error)
                MSE = np.round(SSE / (len(y_values) - 2), 5)
                # Standard Error (s)/ sqrt(MSE)
                standard_error = np.round(np.sqrt(MSE), 5)

                # R**2 CORRELATION --> SSR / SST
                RSquared = np.round(SSR / SST, 5)

                # RSquared Adjusted --> 1 - (1 - RSquared)*(n - 1) / (n - p - 1) --> n = how many observations; p = how many IVs
                RSquared_Adj = np.round(1 - ((1 - RSquared) * (len(y_values) - 1)) / (len(y_values) - 2), 5)

                # R-sq_pred = 1 - PRESS / SST
                # PRESS --> predicted residual sum of squares --> Σ(ei / (1-hi))**2
                # ei == y - y_hat --> y_hat_residuals
                # hi == 1/n + x_residual_squared / x_residual_squared_SUM
                h_leverage = 1 / len(y_values) + (x_residual_squared[which_variable - 1, :] / x_residual_squared_SUM[which_variable - 1, :])
                PRESS = np.sum((y_hat_residuals / (1 - h_leverage)) ** 2)
                RSquared_Pred = np.round(1 - PRESS / SST, 5)

                # make a dictionary and then pandas DataFrame
                dictionary_pandas_multiple = {'Standard Error': standard_error,
                                              'RSquared': RSquared,
                                              'RSquared adj.': RSquared_Adj,
                                              'RSquared pred.': RSquared_Pred}

                df = pd.DataFrame(dictionary_pandas_multiple, index=[0])


                correlation_pandas_dictionary = {'X' + str(which_variable): X_arr[which_variable, :],
                                                 'Y': Y_arr}
                # dataframe for calculating correlation
                df_corr = pd.DataFrame(correlation_pandas_dictionary)
                # this is correlation value
                corr = np.round(df_corr.corr().iloc[0, 1], 4)

                # print out INFO
                print('')
                print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
                print('REGRESSION EQUATION !!! X' + str(i + 1) + ' (IV) vs Y (DV)')
                print('-----------------------------------------------------------')
                print('ŷ = ' + str(b0_intercept) + ' + ' + str(b1_slopes) + ' * X' + str(i + 1))
                print('')
                print('-----------------------------------------------------------')
                print('MODEL SUMMARY')
                print('-----------------------------------------------------------')
                print('')
                print(df.to_string(index=False))
                print('')
                print('-----------------------------------------------------------')
                print('CORRELATION')
                print('-----------------------------------------------------------')
                print(df_corr.corr())
                print('')
                print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')


                # MATPLOTLIB SCATTER PLOTS IV to DV (Relevancy Check) !!!
                # look for high correlation and significant linear relationship !!!

                # make a figure size (width, height) in inches
                fig = plt.figure(figsize=(6, 6))

                # make an ax which will have observed values, regression line and correlation value
                ax1 = plt.subplot2grid((6, 1), (2, 0), rowspan=5, colspan=1, fig=fig)

                ax1.scatter(X_arr[which_variable], Y_arr, c='#6f029e', marker='D',
                            label='Observed Values X' + str(which_variable))
                ax1.plot(X_arr[which_variable], y_hat, color='k', label='Regression Line')
                # make a text which will contain correlation value
                # bbox={'facecolor': 'blue', 'alpha': 0.2, 'pad': 10} --> will make in look like in square, in box
                ax1.text(0.15, 0.72, 'Correlation (r) = ' + str(corr),
                         fontsize=12, transform=plt.gcf().transFigure,
                         bbox={'facecolor': 'blue', 'alpha': 0.2, 'pad': 10})

                ax1.legend(loc='upper left')
                ax1.set_xlabel('X' + str(which_variable) + ' - axis')
                ax1.set_ylabel('Y - axis')
                ax1.set_title('Scatter Plot of Y vs X' + str(which_variable))

                fig.suptitle('\n\n\nIV to DV SCATTER PLOT(S)\n     Y vs X' + str(which_variable))

                plt.show()

                which_variable += 1


            print('')
            print('')
            # ----------------------------------------------------------------------------------------------------------------------------


            # IV pair vs DP; coefficients
            # In scatter plot IV vs IV, in regression model IV pair vs DV
            # ----------------------------------------------------------------------------------------------------------------------------

            # if there are 2 IV
            if how_many_IVs == 2:
                print('')
                print('INDEPENDENT VARIABLES (X1 & X2) VS DEPENDENT VARIABLE (Y) EQUATION(S)')
                print('↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓')


                # !!! COEFFICIENTS CALCULATIONS !!!
                # ---------------------------------------------------------------------------------------------------------------
                X_arr_transposed = X_arr[[0, 1, 2], :].transpose()

                X_arr_matrix_multiplication = np.matmul(X_arr[[0, 1, 2], :], X_arr_transposed)
                X_arr_matrix_multiplication_inversion = np.linalg.pinv(X_arr_matrix_multiplication)

                X_arr_Y_arr_matrix_multiplication = np.matmul(X_arr[[0, 1, 2], :], Y_arr).reshape(3, 1)

                coefficients = np.round(np.matmul(X_arr_matrix_multiplication_inversion, X_arr_Y_arr_matrix_multiplication), 5)

                b0_intercept = coefficients[0, 0]
                b1_slopes = coefficients[1:]
                # --------------------------------------------------------------------------------------------------------------

                # Predicted values
                y_hat = b0_intercept + b1_slopes[0][0] * X_arr[1, :] + b1_slopes[1][0] * X_arr[2, :]
                # (Yi - ŷ)
                y_hat_residuals = Y_arr - y_hat
                # (Yi - ŷ) ** 2
                y_hat_residuals_squared = y_hat_residuals ** 2
                # E(Yi - ŷ) ** 2 --> SSE --> Sum of squares due to Error
                SSE = np.round(np.sum(y_hat_residuals_squared), 5)
                # SST = SSR + SSE --> SSR = SST - SSE --> Sum of Squares regression
                SSR = SST - SSE

                # variance (s ** 2)/MSE (Mean Squared Error); MSE = SSE / (n - (k+1)) --> n = number of observations; k + 1 = number of coefficients
                MSE = np.round(SSE / (len(y_values) - 3), 5)
                # Standard Error (s)/ sqrt(MSE)
                standard_error = np.round(np.sqrt(MSE), 5)

                # R**2 CORRELATION --> SSR / SST
                RSquared = np.round(SSR / SST, 5)

                # RSquared Adjusted --> 1 - (1 - RSquared)*(n - 1) / (n - p - 1) --> n = how many observations; p = how many IVs
                RSquared_Adj = np.round(1 - ((1 - RSquared) * (len(y_values) - 1)) / (len(y_values) - 3), 5)


                # RSquared Predicted !!!
                # -------------------------------------------------------------------------------------------
                # R-sq_pred = 1 - PRESS / SST
                # PRESS --> predicted residual sum of squares --> Σ(ei / (1-hi))**2
                # ei == y - y_hat --> y_hat_residuals
                # hi == 1/n + x_residual_squared / x_residual_squared_SUM

                # H = X (X^TX)^-1 X^T --> take only diagonal values
                # PRESS --> Σ ( ei ** 2 / (1-hi) **2 )
                H = np.matmul(X_arr_matrix_multiplication_inversion, X_arr[[0, 1, 2], :])
                H2 = np.matmul(X_arr_transposed, H)
                Hii = H2.diagonal()
                PRESS = np.sum(y_hat_residuals_squared / (1 - Hii) ** 2)

                RSquared_Pred = np.round(1 - PRESS / SST, 5)
                # ------------------------------------------------------------------------------------------

                # make dictionary and then right after make a pandas DataFrame out of it
                dictionary_pandas_multiple = {'Standard Error': standard_error,
                                              'RSquared': RSquared,
                                              'RSquared adj.': RSquared_Adj,
                                              'RSquared pred.': RSquared_Pred}

                df = pd.DataFrame(dictionary_pandas_multiple, index=[0])

                correlation_pandas_dictionary = {'X1': X_arr[1, :],
                                                 'X2': X_arr[2, :],
                                                 'Y': Y_arr}

                # Correlation dataFrame
                df_corr = pd.DataFrame(correlation_pandas_dictionary)
                # correlation value
                corr = np.round(df_corr.corr().iloc[0, 1], 4)

                print('')
                print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
                print('REGRESSION EQUATION !!! X1 & X2 (IVs) vs Y (DV)')
                print('-----------------------------------------------')
                print('ŷ = ' + str(b0_intercept) + ' + ' + str(b1_slopes[0][0]) + ' * X1 + ' + str(b1_slopes[1][0]) + ' * X2')
                print('')
                print('-----------------------------------------------------------')
                print('MODEL SUMMARY')
                print('-----------------------------------------------------------')
                print('')
                print(df.to_string(index=False))
                print('')
                print('-----------------------------------------------------------')
                print('CORRELATION')
                print('-----------------------------------------------------------')
                print(df_corr.corr())
                print('')
                print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')


                # MATPLOTLIB SCATTER PLOTS IV to IV (MULTICOLLINEARITY Check) !!!
                # IV vs IV SCATTER PLOTS !!!

                fig = plt.figure(figsize=(6, 6))

                ax1 = plt.subplot2grid((6, 1), (2, 0), rowspan=5, colspan=1, fig=fig)

                ax1.scatter(X_arr[1], X_arr[2], c='#6f029e', marker='D', label='Observed Values\nX2 & X1')

                ax1.text(0.15, 0.72, 'Correlation (r) = ' + str(corr),
                         fontsize=12, transform=plt.gcf().transFigure, bbox={'facecolor': 'blue', 'alpha': 0.2, 'pad': 10})

                ax1.legend(loc='upper left')
                ax1.set_xlabel('X1 - axis')
                ax1.set_ylabel('X2 - axis')
                ax1.set_title('Scatter Plot of X2 vs X1')

                fig.suptitle('\n\n\nIV to IV SCATTER PLOT(S)\n    X2 VS X1')

                plt.show()



            # IV pairs vs DP; coefficients
            # when there are 3 IVs
            elif how_many_IVs == 3:
                print('')
                print('INDEPENDENT` VARIABLES (X1 & X2; X1 & X3; X2 & X3) VS DEPENDENT VARIABLE (Y) EQUATION(S)')
                print('↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓')

                # to orient around which variable to make calculations with
                variable_1 = 1
                variable_2 = 2

                for i in range(3):
                    # when x1 will be calculated with x2 and x3, then from x1 it will be x2 and then that x2 will be calculated with x3
                    if variable_2 > 3:
                        variable_1 = 2
                        variable_2 = 3

                    # !!! COEFFICIENTS CALCULATIONS !!!
                    # --------------------------------------------------------------------------------------------------------------
                    X_arr_transposed = X_arr[[0, variable_1, variable_2], :].transpose()

                    X_arr_matrix_multiplication = np.matmul(X_arr[[0, variable_1, variable_2], :], X_arr_transposed)
                    X_arr_matrix_multiplication_inversion = np.linalg.pinv(X_arr_matrix_multiplication)

                    X_arr_Y_arr_matrix_multiplication = np.matmul(X_arr[[0, variable_1, variable_2], :], Y_arr).reshape(3, 1)

                    coefficients = np.round(np.matmul(X_arr_matrix_multiplication_inversion, X_arr_Y_arr_matrix_multiplication), 5)

                    b0_intercept = coefficients[0, 0]
                    b1_slopes = coefficients[1:]
                    # ---------------------------------------------------------------------------------------------------------------


                    # Predicted Values
                    y_hat = b0_intercept + b1_slopes[0][0] * X_arr[variable_1, :] + b1_slopes[1][0] * X_arr[variable_2, :]
                    # (Yi - ŷ)
                    y_hat_residuals = Y_arr - y_hat
                    # (Yi - ŷ) ** 2
                    y_hat_residuals_squared = y_hat_residuals ** 2
                    # E(Yi - ŷ) ** 2 --> SSE --> Sum of squares due to Error
                    SSE = np.round(np.sum(y_hat_residuals_squared), 5)
                    # SST = SSR + SSE --> SSR = SST - SSE --> Sum of Squares regression
                    SSR = SST - SSE

                    # variance (s ** 2)/MSE (Mean Squared Error); MSE = SSE / (n - (k+1)) --> n = number of observations; k + 1 = number of coefficients
                    MSE = np.round(SSE / (len(y_values) - 3), 5)
                    # Standard Error (s)/ sqrt(MSE)
                    standard_error = np.round(np.sqrt(MSE), 5)

                    # R**2 CORRELATION --> SSR / SST
                    RSquared = np.round(SSR / SST, 5)

                    # RSquared Adjusted --> 1 - (1 - RSquared)*(n - 1) / (n - p - 1) --> n = how many observations; p = how many IVs
                    RSquared_Adj = np.round(1 - ((1 - RSquared) * (len(y_values) - 1)) / (len(y_values) - 3), 5)


                    # RSquared Predicted
                    # -------------------------------------------------------------------------------------------

                    # R-sq_pred = 1 - PRESS / SST
                    # PRESS --> predicted residual sum of squares --> Σ(ei / (1-hi))**2
                    # ei == y - y_hat --> y_hat_residuals
                    # hi == 1/n + x_residual_squared / x_residual_squared_SUM

                    # H = X (X^TX)^-1 X^T --> take only diagonal values
                    # PRESS --> Σ ( ei ** 2 / (1-hi) **2 )
                    H = np.matmul(X_arr_matrix_multiplication_inversion, X_arr[[0, variable_1, variable_2], :])
                    H2 = np.matmul(X_arr_transposed, H)
                    Hii = H2.diagonal()
                    PRESS = np.sum(y_hat_residuals_squared / (1 - Hii) ** 2)

                    RSquared_Pred = np.round(1 - PRESS / SST, 5)

                    # ------------------------------------------------------------------------------------------

                    # make a dictionary and then make a pandas Dataframe
                    dictionary_pandas_multiple = {'Standard Error': standard_error,
                                                  'RSquared': RSquared,
                                                  'RSquared adj.': RSquared_Adj,
                                                  'RSquared pred.': RSquared_Pred}

                    df = pd.DataFrame(dictionary_pandas_multiple, index=[0])

                    # dictionary with them values
                    correlation_pandas_dictionary = {'X' + str(variable_1): X_arr[variable_1],
                                                     'X' + str(variable_2): X_arr[variable_2],
                                                     'Y': Y_arr}
                    # Correlation Dataframe
                    df_corr = pd.DataFrame(correlation_pandas_dictionary)
                    # Correlation value
                    corr = np.round(df_corr.corr().iloc[0, 1], 4)


                    print('')
                    print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
                    print('REGRESSION EQUATION !!! X' + str(variable_1) + ' & X' + str(variable_2) + ' vs Y (DV)')
                    print('-------------------------------------------------------------------------------------')
                    print('ŷ = ' + str(b0_intercept) + ' + ' + str(b1_slopes[0][0]) + ' * X' + str(variable_1) + ' + ' + str(b1_slopes[1][0]) + ' * X' + str(variable_2))
                    print('')
                    print('-------------------------------------------------------------------------------------')
                    print('MODEL SUMMARY')
                    print('-------------------------------------------------------------------------------------')
                    print('')
                    print(df.to_string(index=False))
                    print('')
                    print('-------------------------------------------------------------------------------------')
                    print('CORRELATION')
                    print('-------------------------------------------------------------------------------------')
                    print(df_corr.corr())
                    print('')
                    print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')


                    # MATPLOTLIB SCATTER PLOTS IV to IV (MULTICOLLINEARITY Check) !!!
                    # IV vs IV Scatter Plots and Correlation values

                    fig = plt.figure(figsize=(6, 6))

                    ax1 = plt.subplot2grid((6, 1), (2, 0), rowspan=5, colspan=1, fig=fig)

                    ax1.scatter(X_arr[variable_1], X_arr[variable_2], c='#6f029e', marker='D', label='Observed Values\nX' + str(variable_2) + ' & X' + str(variable_1))

                    ax1.text(0.15, 0.72, 'Correlation (r) = ' + str(corr),
                             fontsize=12, transform=plt.gcf().transFigure, bbox={'facecolor': 'blue', 'alpha': 0.2, 'pad': 10})

                    ax1.legend(loc='upper left')
                    ax1.set_xlabel('X' + str(variable_1) + ' - axis')
                    ax1.set_ylabel('X' + str(variable_2) + ' - axis')
                    ax1.set_title('Scatter Plot of X' + str(variable_2) + ' vs X' + str(variable_1))

                    fig.suptitle('\n\n\nIV to IV SCATTER PLOT(S)\n    X' + str(variable_2) + ' vs X' + str(variable_1))

                    plt.show()


                    variable_2 += 1

                # -----------------------------------------------------------------------------------------------------------------------------------


                # X1 X2 X3 and DV calculation
                print('')
                print('')
                print('INDEPENDENT` VARIABLES (X1 & X2 & X3) VS DEPENDENT VARIABLE (Y) EQUATION(S)')
                print('↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓')

                # !!! COEFFICIENTS CALCULATIONS !!!
                # -------------------------------------------------------------------------------------------------------------
                X_arr_transposed = X_arr[[0, 1, 2, 3], :].transpose()

                X_arr_matrix_multiplication = np.matmul(X_arr[[0, 1, 2, 3], :], X_arr_transposed)
                X_arr_matrix_multiplication_inversion = np.linalg.pinv(X_arr_matrix_multiplication)

                X_arr_Y_arr_matrix_multiplication = np.matmul(X_arr[[0, 1, 2, 3], :], Y_arr).reshape(4, 1)

                coefficients = np.round(np.matmul(X_arr_matrix_multiplication_inversion, X_arr_Y_arr_matrix_multiplication), 5)

                b0_intercept = coefficients[0, 0]
                b1_slopes = coefficients[1:]
                # ------------------------------------------------------------------------------------------------------------

                # Predicted Values
                y_hat = b0_intercept + b1_slopes[0][0] * X_arr[1, :] + b1_slopes[1][0] * X_arr[2, :] + b1_slopes[2][0] * X_arr[3, :]
                # (Yi - ŷ)
                y_hat_residuals = Y_arr - y_hat
                # (Yi - ŷ) ** 2
                y_hat_residuals_squared = y_hat_residuals ** 2
                # E(Yi - ŷ) ** 2 --> SSE --> Sum of squares due to Error
                SSE = np.round(np.sum(y_hat_residuals_squared), 5)
                # SST = SSR + SSE --> SSR = SST - SSE --> Sum of Squares regression
                SSR = SST - SSE

                # variance (s ** 2)/MSE (Mean Squared Error); MSE = SSE / (n - (k+1)) --> n = number of observations; k + 1 = number of coefficients
                MSE = np.round(SSE / (len(y_values) - 4), 5)
                # Standard Error (s)/ sqrt(MSE)
                standard_error = np.round(np.sqrt(MSE), 5)

                # R**2 CORRELATION --> SSR / SST
                RSquared = np.round(SSR / SST, 5)

                # RSquared Adjusted --> 1 - (1 - RSquared)*(n - 1) / (n - p - 1) --> n = how many observations; p = how many IVs
                RSquared_Adj = np.round(1 - ((1 - RSquared) * (len(y_values) - 1)) / (len(y_values) - 4), 5)


                # RSquared Predicted
                # -------------------------------------------------------------------------------------------

                # R-sq_pred = 1 - PRESS / SST
                # PRESS --> predicted residual sum of squares --> Σ(ei / (1-hi))**2
                # ei == y - y_hat --> y_hat_residuals
                # hi == 1/n + x_residual_squared / x_residual_squared_SUM

                # H = X (X^TX)^-1 X^T --> take only diagonal values
                # PRESS --> Σ ( ei ** 2 / (1-hi) **2 )
                H = np.matmul(X_arr_matrix_multiplication_inversion, X_arr[[0, 1, 2, 3], :])
                H2 = np.matmul(X_arr_transposed, H)
                Hii = H2.diagonal()
                PRESS = np.sum(y_hat_residuals_squared / (1 - Hii) ** 2)

                RSquared_Pred = np.round(1 - PRESS / SST, 5)
                # ------------------------------------------------------------------------------------------

                # this will not include any plots cause we have more than 2 values at the same time !!!


                dictionary_pandas_multiple = {'Standard Error': standard_error,
                                              'RSquared': RSquared,
                                              'RSquared adj.': RSquared_Adj,
                                              'RSquared pred.': RSquared_Pred}

                df = pd.DataFrame(dictionary_pandas_multiple, index=[0])

                correlation_pandas_dictionary = {'X1': X_arr[1, :],
                                                 'X2': X_arr[2, :],
                                                 'X3': X_arr[3, :],
                                                 'Y': Y_arr}
                df_corr = pd.DataFrame(correlation_pandas_dictionary)


                print('')
                print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
                print('REGRESSION EQUATION !!! X1 & X2 & X3 vs Y (DV)')
                print('----------------------------------------------')
                print('ŷ = ' + str(b0_intercept) + ' + ' + str(b1_slopes[0][0]) + ' * X1 + ' + str(b1_slopes[1][0]) + ' * X2 + ' + str(b1_slopes[2][0]) + ' * X3')
                print('')
                print('----------------------------------------------')
                print('MODEL SUMMARY')
                print('----------------------------------------------')
                print('')
                print(df.to_string(index=False))
                print('')
                print('-------------------------------------------------------------------------------------')
                print('CORRELATION')
                print('-------------------------------------------------------------------------------------')
                print(df_corr.corr())
                print('')
                print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')

            # --------------------------------END-------------------------------#

            exit()

        evaluation()


    # when want to make a linear regression
    elif ask == 'LR':

        def linear_regression():
            global how_many_IVs, X_arr, Y_arr, y_values, SST, x_residual_squared, x_residual_squared_SUM

            # !!! CHECK FOR EXCEPTIONS AND ERRORS !!!
            # ----------------------------------------------------------------------------------------------------------------------------
            print('')
            try:
                how_many_IVs = int(input('How many independent variables you want to use for linear regression? (max 3 allowed !!!): '))

            except ValueError:
                print('')
                print('TRY AGAIN !!!')
                linear_regression()

            if (how_many_IVs >= 1) and (how_many_IVs <= 3):
                pass

            else:
                print('')
                print('Minimum 1 IV and maximum 3 IVs allowed, try again !!!')
                linear_regression()

            try:

                IVs_values = []
                for i in range(1, how_many_IVs + 1):
                    IVs = input('Independent variable number ' + str(i) + ' (x' + str(i) + ' values): ').split(', ')
                    IVs_values.append(IVs)

                IVs_values.insert(0, [float(i) - (i - 1) for i in range(1, len(IVs_values[0]) + 1)])

                y_values = input('Dependent variable values (y values): ').split(', ')

                X_arr = np.array(IVs_values)
                X_arr = X_arr.astype(np.float)
                Y_arr = np.array([float(i) for i in y_values])

            except ValueError:
                print('')
                print('Oops something went wrong, try again !!!')
                linear_regression()

            if how_many_IVs == 1:
                if len(X_arr[1, :]) != len(Y_arr):
                    print('')
                    print('For each variable must be same amount of numbers, try again !!!')
                    linear_regression()
                else:
                    pass

                if len(Y_arr) <= 4:
                    print('')
                    print('You must pass more amount of values than 4, try again !!!')
                    linear_regression()


            elif how_many_IVs == 2:
                if len(X_arr[1, :]) != len(Y_arr) or len(X_arr[2, :]) != len(Y_arr) or len(X_arr[1, :]) != len(X_arr[2, :]):
                    print('')
                    print('For each variable must be same amount of numbers, try again !!!')
                    linear_regression()
                else:
                    pass

                if len(Y_arr) <= 4:
                    print('')
                    print('You must pass more amount of values than 4, try again !!!')
                    linear_regression()

            elif how_many_IVs == 3:
                if len(X_arr[1, :]) != len(Y_arr) or len(X_arr[2, :]) != len(Y_arr) or len(X_arr[3, :]) != len(
                        Y_arr) or len(X_arr[1, :]) != len(X_arr[2, :]) or len(X_arr[1, :]) != len(X_arr[3, :]) or len(
                        X_arr[2, :]) != len(X_arr[3, :]):
                    print('')
                    print('For each variable must be same amount of numbers, try again !!!')
                    linear_regression()
                else:
                    pass

            if len(Y_arr) <= 4:
                print('')
                print('You must pass more amount of values than 4, try again !!!')
                linear_regression()
            # -------------------------------------------------------------------------------------------------------------------------


            # MORE INFO CALCULATIONS !!!

            # ------------------------START---------------------------- #
            x = X_arr[1:]

            x_mean = x.mean(axis=1).reshape(how_many_IVs, 1)
            y_mean = Y_arr.mean()

            # (Xi - X bar)
            x_residual = x - x_mean
            # (Yi - Y bar)
            y_residual = Y_arr - y_mean

            # (Xi - X bar) ** 2
            x_residual_squared = np.power(x_residual, 2)
            # (Yi - Y bar) ** 2
            y_residual_squared = y_residual ** 2

            # E (summation symbol) --> E(Xi - X bar) ** 2
            x_residual_squared_SUM = np.sum(x_residual_squared, axis=1).reshape(how_many_IVs, 1)
            # E(Yi - Y bar) ** 2 --> SST --> Sum of squares Total; Total Variation
            SST = np.round(np.sum(y_residual_squared), 5)



            # !!! COEFFICIENTS CALCULATIONS !!!
            # ----------------------------------------------------------------------------------------------------------------
            X_arr_transposed = X_arr.transpose()

            X_arr_matrix_multiplication = np.matmul(X_arr, X_arr_transposed)
            X_arr_matrix_multiplication_inversion = np.linalg.pinv(X_arr_matrix_multiplication)

            X_arr_Y_arr_matrix_multiplication = np.matmul(X_arr, Y_arr).reshape(how_many_IVs + 1, 1)

            coefficients = np.round(np.matmul(X_arr_matrix_multiplication_inversion, X_arr_Y_arr_matrix_multiplication), 5)

            b0_intercept = coefficients[0, 0]
            b1_slopes = coefficients[1:]
            # ---------------------------------------------------------------------------------------------------------------

            # store ' + b1slope * X1..2..n ' into that list
            slope_star_X = []
            for i in range(len(b1_slopes)):
                part = ' + ' + str(b1_slopes[i][0]) + ' * X' + str(i + 1)
                slope_star_X.append(part)

            # then each that part add into regression equation
            regression_equation = 'ŷ = ' + str(b0_intercept)
            for part in slope_star_X:
                regression_equation += part


            print('')
            print('!!! REGRESSION LINE EQUATION !!!')
            print('---------------------------------')
            print(regression_equation)

            # store actual part for calculations
            slope_star_X_calcul = []
            for i in range(len(b1_slopes)):
                part = ' + ' + str(b1_slopes[i][0]) + ' * X_arr[' + str(i + 1) + ', :]'
                slope_star_X_calcul.append(part)

            # Predicted Values
            y_hat = eval(str(b0_intercept) + ''.join(slope_star_X_calcul))
            # (Yi - ŷ)
            y_hat_residuals = Y_arr - y_hat
            # (Yi - ŷ) ** 2
            y_hat_residuals_squared = y_hat_residuals ** 2
            # E(Yi - ŷ) ** 2 --> SSE --> Sum of squares due to Error
            SSE = np.round(np.sum(y_hat_residuals_squared), 5)
            # SST = SSR + SSE --> SSR = SST - SSE --> Sum of Squares regression
            SSR = SST - SSE

            # variance (s ** 2)/MSE (Mean Squared Error)
            MSE = np.round(SSE / (len(Y_arr) - 2), 5)
            # Standard Error (s)/ sqrt(MSE)
            standard_error = np.round(np.sqrt(MSE), 5)

            # R**2 CORRELATION --> SSR / SST
            RSquared = np.round(SSR / SST, 5)

            # RSquared Adjusted --> 1 - (1 - RSquared)*(n - 1) / (n - p - 1) --> n = how many observations; p = how many IVs
            RSquared_Adj = np.round(1 - ((1 - RSquared) * (len(Y_arr) - 1)) / (len(Y_arr) - 2), 5)


            # R-sq_pred = 1 - PRESS / SST
            # H = X (X^TX)^-1 X^T --> take only diagonal values
            # PRESS --> Σ ( ei ** 2 / (1-hi) **2 )
            H = np.matmul(X_arr_matrix_multiplication_inversion, X_arr)
            H2 = np.matmul(X_arr_transposed, H)
            Hii = H2.diagonal()
            PRESS = np.sum(y_hat_residuals_squared / (1 - Hii) ** 2)

            RSquared_Pred = np.round(1 - PRESS / SST, 5)


            # read a t-table, there are numbers to calculate a t distributions
            t_table = pd.read_csv('t-distribution', delimiter=',')

            # make a list of column names in that file
            column_names = list(t_table.columns)
            # convert all column names except the 1st name into float and make a list of it called convert
            convert = [float(i) for i in column_names[1:]]
            # append into 1st position of converted list a name 'DF
            convert.insert(0, 'DF')
            # change the name columns into those that are in convert list
            t_table.columns = convert

            # arange the index of that table to start from 1 end to 100
            t_table.index = np.arange(1, 102)

            # get a value from a specific cell from that file --> .at[index_name, column_name]
            # if there are more than 100 values to predict (which is bad) then set the last row and its cell in t_table dataset
            # otherwise count with number of observations
            if len(Y_arr) > 100:
                t_value = t_table.at[101, 0.025]
            else:
                t_value = t_table.at[len(Y_arr) - how_many_IVs - 1, 0.025]


            # Into those list there will be stored actual CI and PI Lower and Upper values
            CI_lower_values = []
            CI_upper_values = []

            PI_lower_values = []
            PI_upper_values = []

            for i in range(len(Y_arr)):

                # CONFIDENCE INTERVAL CALCULATIONS !!!
                # --------------------------------------------------------------------------------------------------------------------------------------------
                CI_BANDS_standard_deviation = np.sqrt(MSE * np.matmul(X_arr[:, i], np.matmul(X_arr_matrix_multiplication_inversion, X_arr_transposed[i, :])))

                CI_LOWER_95 = np.round(y_hat - (t_value * CI_BANDS_standard_deviation), 5)
                CI_UPPER_95 = np.round(y_hat + (t_value * CI_BANDS_standard_deviation), 5)

                CI_lower_values.append(CI_LOWER_95[i])
                CI_upper_values.append(CI_UPPER_95[i])
                # --------------------------------------------------------------------------------------------------------------------------------------------

                # PREDICTION INTERVAL CALCULATIONS !!!
                # --------------------------------------------------------------------------
                s_pred = np.sqrt(standard_error ** 2 + CI_BANDS_standard_deviation ** 2)
                PI_LOWER_95 = np.round(y_hat - (t_value * s_pred), 5)
                PI_UPPER_95 = np.round(y_hat + (t_value * s_pred), 5)

                PI_lower_values.append(PI_LOWER_95[i])
                PI_upper_values.append(PI_UPPER_95[i])
                # --------------------------------------------------------------------------


            # make a numpy array out of them !!!
            CI_LOWER_95 = np.array(CI_lower_values)
            CI_UPPER_95 = np.array(CI_upper_values)

            PI_LOWER_95 = np.array(PI_lower_values)
            PI_UPPER_95 = np.array(PI_upper_values)


            # # make a dictionary with keys as column names and values as the value in that column
            dictionary_pandas_multiple = {}

            dictionary = {'Y': Y_arr,
                          'Y_hat': y_hat,
                          '95% CI Lower': CI_LOWER_95,
                          '95% CI Upper': CI_UPPER_95,
                          '95% PI Lower': PI_LOWER_95,
                          '95% PI Upper': PI_UPPER_95}

            # store into empty dictionary IV values
            for i in range(1, how_many_IVs + 1):
                dictionary_pandas_multiple['X' + str(i)] = X_arr[i, :]

            # update dictionary with only IVs and add into that what's in dict dictionary
            for key, value in dictionary.items():
                dictionary_pandas_multiple[key] = value

            # make a DataFrame from that Dict
            df1 = pd.DataFrame(dictionary_pandas_multiple)


            # contains sum of squares and mean square error
            dictionary_pandas_multiple_2 = {'SST': SST,
                                            'SSR': SSR,
                                            'SSE': SSE,
                                            'MSE': MSE}
            # index[0] --> for scalar error purposes
            df2 = pd.DataFrame(dictionary_pandas_multiple_2, index=[0])


            # !!! CONTAINS THE MOST IMPORTANT STATS WHEN YOU WANT TO EVALUATE HOW GOOD OUR REGRESSION IS !!!
            dictionary_pandas_multiple = {'Standard Error': standard_error,
                                          'RSquared': RSquared,
                                          'RSquared adj.': RSquared_Adj,
                                          'RSquared pred.': RSquared_Pred}
            df3 = pd.DataFrame(dictionary_pandas_multiple, index=[0])


            # .to_string(index=False) --> this will print the dataframe without the index
            print('')
            print(df1.to_string(index=False))
            print('')
            print('')
            print(df2.to_string(index=False))
            print('')
            print(df3.to_string(index=False))
            print('')

            # in case just print out the info in lines
            print('')
            print('Root Mean Squared Error / Standard Error (S) = ' + str(standard_error))
            print('RSquared (r ** 2) = ' + str(RSquared))
            print('RSquared Adjusted = ' + str(RSquared_Adj))
            print('RSquared Predicted = ' + str(RSquared_Pred))
            print('')
            print('SST (Sum of Squares Total) = ' + str(SST))
            print('SSR (Sum od Squares Regression) = ' + str(SSR))
            print('SSE (Sum of Squares due to Error) = ' + str(SSE))
            print('Mean Squared Error (MSE) = ' + str(MSE))

            # make graphs
            def make_plots():

                print('')
                ask_plots = input('Do you want to see graphs too? Yes/No: ').upper()

                if ask_plots == 'YES':
                    # take only those values the person has input
                    x = X_arr[1:, :]
                    # make a transposition
                    x_transposed = x.transpose()

                    # if there is only 1 IV
                    if how_many_IVs == 1:
                        x = x.reshape(-1)

                        # we are going to have 3 figures, so this is what we do
                        fig1, ax1 = plt.subplots()
                        fig2, ax2 = plt.subplots()
                        fig3, (ax3, ax4) = plt.subplots(1, 2)

                        # Figure 1, Simple Linear Regression, and Observed Values
                        # Observed Values, Scatter PLot
                        ax1.scatter(x, Y_arr, c='#6f029e', marker='D', label='Observed Values')
                        # Linear regression with Predicted Values
                        ax1.scatter(x, y_hat, color='k', label='Predicted Values')
                        ax1.plot(x, y_hat, color='k', label='Regression Line')

                        # Figure 2, Regression Line and 95% CI and PI Band !!!
                        # we sort values 'because we want that plot, line to be on the same road, plynula ciara
                        # Observed values
                        ax2.scatter(x, Y_arr, c='#6f029e', marker='D', label='Observed Values')
                        # Regression line, sorted values
                        ax2.plot(np.sort(x), np.sort(y_hat), color='k', label='Regression Line')

                        # 95% CI Lower, sorted values
                        ax2.plot(np.sort(x), np.sort(CI_LOWER_95), color='r')
                        # 95% CI Upper, sorted values
                        ax2.plot(np.sort(x), np.sort(CI_UPPER_95), color='r')

                        # 95% PI Lower, sorted values
                        ax2.plot(np.sort(x), np.sort(PI_LOWER_95), color='g')
                        # 95% PI Upper, sorted values
                        ax2.plot(np.sort(x), np.sort(PI_UPPER_95), color='g')

                        # we make empty plots with labels in legend 'cause in fill_between there is no label parameter
                        ax2.plot([], [], color='r', label='95% CI Band')
                        ax2.plot([], [], color='g', label='95% PI Band')

                        # fill between CI band (red) and PI band (green)
                        ax2.fill_between(np.sort(x), np.sort(CI_UPPER_95), np.sort(CI_LOWER_95), facecolor='r',
                                         interpolate=True, alpha=0.2)
                        ax2.fill_between(np.sort(x), np.sort(PI_UPPER_95), np.sort(PI_LOWER_95), facecolor='g',
                                         interpolate=True, alpha=0.05)

                        # Figure 3, Residual Plot !!!
                        # For ax3 -> Residuals vs X
                        # make a scatter plot between X and residuals (y - y_predicted)
                        ax3.scatter(x, y_hat_residuals, c='#1E8449', marker='h', label='Residuals')
                        # make a horizontal dashed line where y = 0, and goes through all width of plot --> .hlines()
                        ax3.hlines(y=0, xmin=x.min(), xmax=x.max(), linestyles='--', color='#17202A')

                        # For ax4 -> Residuals vs Y_predicted
                        ax4.scatter(y_hat, y_hat_residuals, c='#1E8449', marker='h', label='Residuals')
                        # make a horizontal dashed line where y = 0, and goes through all width of plot --> .hlines()
                        ax4.hlines(y=0, xmin=y_hat.min(), xmax=y_hat.max(), linestyles='--', color='#17202A')


                        # customize the figures, names etc...
                        ax1.legend(loc='upper left')
                        ax1.set_xlabel('X-axis')
                        ax1.set_ylabel('Y-axis')
                        ax1.set_title('Simple Linear Regression')

                        ax2.legend(loc='upper left')
                        ax2.set_xlabel('X-axis')
                        ax2.set_ylabel('Y-axis')
                        ax2.set_title('Regression Line and 95% CI and PI Band')

                        ax3.legend(loc='lower left')
                        ax3.set_xlabel('X-axis')
                        ax3.set_ylabel('Residuals')
                        ax3.set_title('Residual Plot Against X (IV)')

                        ax4.legend(loc='lower left')
                        ax4.set_xlabel('Y PREDICTED-axis')
                        ax4.set_ylabel('Residuals')
                        ax4.set_title('Residual Plot Against Y_predicted (y_hat)')

                        fig3.suptitle('RESIDUAL SCATTER PLOTS')

                        plt.show()


                    elif how_many_IVs == 2:

                        # we are going to have 4 figures, so this is what we do

                        # This is Dependent Variable (Y) vs Independent Variable (Xn) Scatter plots
                        # they will be sharing y axis --> sharey = ax1
                        fig1 = plt.figure()
                        ax1 = plt.subplot2grid((8, 1), (0, 0), rowspan=3, colspan=1, fig=fig1)
                        ax2 = plt.subplot2grid((8, 1), (5, 0), rowspan=3, colspan=1, sharey=ax1, fig=fig1)

                        # DEPENDENT VARIABLE VALUES vs PREDICTED VALUES SCATTER PLOT
                        # CI Bands and PI Bands
                        fig2, ax3 = plt.subplots()

                        # 3D Linear regression !!!
                        fig3 = plt.figure()
                        ax4 = fig3.add_subplot(111, projection='3d')

                        # Residual Scatter Plot
                        fig4, ax5 = plt.subplots()



                        # Figure 1, Observed Values Scatter Plot
                        # Observed Values
                        ax1.scatter(x[0], Y_arr, c='#6f029e', marker='D', label='Observed Values X1')
                        ax2.scatter(x[1], Y_arr, c='#0020C8', marker='D', label='Observed Values X2')



                        # Figure 2, DEPENDENT VARIABLE VALUES vs PREDICTED VALUES SCATTER PLOT, and 95% CI and PI Band !!!
                        # we sort values 'because we want that plot, line to be on the same road, plynula ciara

                        # Y vs Y_PREDICTED scatter plot
                        ax3.scatter(np.sort(y_hat), np.sort(Y_arr), c='#6f029e', marker='D', label='Y vs Y_PRED')

                        # !!! CALCULATIONS FOR SMOOTHER LINE A LITTLE !!!
                        # -------------------------------------------------------------------------------
                        x_new = np.linspace(y_hat.min(), y_hat.max(), 200)

                        spl_CI_LOWER = make_interp_spline(np.sort(y_hat), np.sort(CI_LOWER_95), k=1)
                        y_smooth_CI_LOWER = spl_CI_LOWER(x_new)

                        spl_CI_UPPER = make_interp_spline(np.sort(y_hat), np.sort(CI_UPPER_95), k=1)
                        y_smooth_CI_UPPER = spl_CI_UPPER(x_new)

                        spl_PI_LOWER = make_interp_spline(np.sort(y_hat), np.sort(PI_LOWER_95), k=1)
                        y_smooth_PI_LOWER = spl_PI_LOWER(x_new)

                        spl_PI_UPPER = make_interp_spline(np.sort(y_hat), np.sort(PI_UPPER_95), k=1)
                        y_smooth_PI_UPPER = spl_PI_UPPER(x_new)
                        # ------------------------------------------------------------------------------

                        # CI 95%
                        ax3.plot(x_new, y_smooth_CI_LOWER, color='r')
                        ax3.plot(x_new, y_smooth_CI_UPPER, color='r')

                        # 95% PI
                        ax3.plot(x_new, y_smooth_PI_LOWER, color='g')
                        ax3.plot(x_new, y_smooth_PI_UPPER, color='g')

                        # we make empty plots with labels in legend 'cause in fill_between there is no label parameter
                        ax3.plot([], [], color='r', label='95% CI Band')
                        ax3.plot([], [], color='g', label='95% PI Band')

                        # fill between CI band (red) and PI band (green)
                        ax3.fill_between(x_new, y_smooth_CI_UPPER, y_smooth_CI_LOWER, facecolor='r',
                                         interpolate=True, alpha=0.2)
                        ax3.fill_between(x_new, y_smooth_PI_UPPER, y_smooth_PI_LOWER, facecolor='g',
                                         interpolate=True, alpha=0.05)


                        # Figure 3, 3D LINEAR REGRESSION !!!
                        # Scatter plot our values x1 x2 and y
                        ax4.scatter(x[0], x[1], Y_arr, c='#FF0000', marker='D')

                        # now we are going to create a regression line in 3D Plot !!!
                        # ----------------------------------------------------------------------
                        x_surf, y_surf = np.meshgrid(np.linspace(x[0].min(), x[0].max(), 100),
                                                     np.linspace(x[1].min(), x[1].max(), 100))
                        onlyX = pd.DataFrame({'X1': x_surf.ravel(), 'X2': y_surf.ravel()})

                        regress = LinearRegression()
                        regress.fit(x_transposed, Y_arr)
                        fittedY = regress.predict(onlyX)
                        fittedY = np.array(fittedY)

                        # to make a line we want, like a desk --> .plot_surface
                        ax4.plot_surface(x_surf, y_surf, fittedY.reshape(x_surf.shape),
                                         color='b', rstride=2, cstride=2, alpha=0.3)
                        # -----------------------------------------------------------------------



                        # Figure 4, Residual Plot !!!
                        # For ax5 -> Residuals vs Y_predicted
                        ax5.scatter(y_hat, y_hat_residuals, c='#1E8449', marker='h', label='Residuals', s=100)
                        # make a horizontal dashed line where y = 0, and goes through all width of plot --> .hlines()
                        ax5.hlines(y=0, xmin=y_hat.min(), xmax=y_hat.max(), linestyles='--', color='#17202A')
                        


                        # customize the figures, names etc...
                        ax1.legend(loc='upper left')
                        ax1.set_xlabel('X1-axis')
                        ax1.set_ylabel('Y-axis')
                        ax1.set_title('SCATTER PLOT Y vs X1')

                        ax2.legend(loc='upper left')
                        ax2.set_xlabel('X2-axis')
                        ax2.set_ylabel('Y-axis')
                        ax2.set_title('SCATTER PLOT Y vs X2')

                        ax3.legend(loc='upper left')
                        ax3.set_xlabel('Y PREDICTED-axis')
                        ax3.set_ylabel('Y-axis')
                        ax3.set_title('Y vs Y_PREDICTED SCATTER PLOT & CI 95%, PI 95% Intervals')

                        ax4.set_xlabel('X1-axis')
                        ax4.set_ylabel('X2-axis')
                        ax4.set_zlabel('Y-axis')
                        ax4.set_title('3D SCATTER PLOT Y vs X1 & X2; REGRESSION LINE')

                        ax5.legend(loc='lower left')
                        ax5.set_xlabel('Y PREDICTED-axis')
                        ax5.set_ylabel('Residuals')
                        ax5.set_title('Residual Plot Against Y_predicted (y_hat)')

                        fig1.suptitle('DEPENDENT VARIABLE (Y) vs INDEPENDENT VARIABLE(X) SCATTER PLOTS')
                        fig3.suptitle('3D PLOT Y vs X1 & X2; REGRESSION LINE')
                        fig4.suptitle('RESIDUAL SCATTER PLOTS')

                        plt.show()


                    elif how_many_IVs == 3:

                        # This is Dependent Variable (Y) vs Independent Variable (Xn) Scatter plots
                        # they will be sharing y axis --> sharey = ax1
                        fig1 = plt.figure()
                        ax1 = plt.subplot2grid((13, 1), (0, 0), rowspan=3, colspan=1, fig=fig1)
                        ax2 = plt.subplot2grid((13, 1), (5, 0), rowspan=3, colspan=1, sharey=ax1, fig=fig1)
                        ax3 = plt.subplot2grid((13, 1), (10, 0), rowspan=3, colspan=1, sharey=ax1, fig=fig1)

                        # DEPENDENT VARIABLE VALUES vs PREDICTED VALUES SCATTER PLOT
                        # CI Bands and PI Bands
                        fig2, ax4 = plt.subplots()

                        # Residual Scatter Plot
                        fig3, ax5 = plt.subplots()



                        # Figure 1, Observed Values Scatter Plot
                        # Observed Values Scatter Plot
                        ax1.scatter(x[0], Y_arr, c='#6f029e', marker='D', label='Observed Values X1')
                        ax2.scatter(x[1], Y_arr, c='#0020C8', marker='D', label='Observed Values X2')
                        ax3.scatter(x[2], Y_arr, c='#FFA200', marker='D', label='Observed Values X3')



                        # Figure 2, DEPENDENT VARIABLE VALUES vs PREDICTED VALUES, and 95% CI and PI Band !!!
                        # we sort values 'because we want that plot, line to be on the same road, plynula ciara

                        # Y vs Y_PREDICTED scatter plot
                        ax4.scatter(np.sort(y_hat), np.sort(Y_arr), c='#6f029e', marker='D', label='Y vs Y_PRED')

                        # !!! CALCULATIONS FOR SMOOTHER LINE IN PI AND CI BANDS !!!
                        # -----------------------------------------------------------------------------
                        x_new = np.linspace(y_hat.min(), y_hat.max(), 200)

                        spl_CI_LOWER = make_interp_spline(np.sort(y_hat), np.sort(CI_LOWER_95), k=1)
                        y_smooth_CI_LOWER = spl_CI_LOWER(x_new)

                        spl_CI_UPPER = make_interp_spline(np.sort(y_hat), np.sort(CI_UPPER_95), k=1)
                        y_smooth_CI_UPPER = spl_CI_UPPER(x_new)

                        spl_PI_LOWER = make_interp_spline(np.sort(y_hat), np.sort(PI_LOWER_95), k=1)
                        y_smooth_PI_LOWER = spl_PI_LOWER(x_new)

                        spl_PI_UPPER = make_interp_spline(np.sort(y_hat), np.sort(PI_UPPER_95), k=1)
                        y_smooth_PI_UPPER = spl_PI_UPPER(x_new)
                        # -----------------------------------------------------------------------------

                        # CI 95%
                        ax4.plot(x_new, y_smooth_CI_LOWER, color='r')
                        ax4.plot(x_new, y_smooth_CI_UPPER, color='r')

                        # 95% PI
                        ax4.plot(x_new, y_smooth_PI_LOWER, color='g')
                        ax4.plot(x_new, y_smooth_PI_UPPER, color='g')

                        # we make empty plots with labels in legend 'cause in fill_between there is no label parameter
                        ax4.plot([], [], color='r', label='95% CI Band')
                        ax4.plot([], [], color='g', label='95% PI Band')

                        # fill between CI band (red) and PI band (green)
                        ax4.fill_between(x_new, y_smooth_CI_UPPER, y_smooth_CI_LOWER, facecolor='r',
                                         interpolate=True, alpha=0.2)
                        ax4.fill_between(x_new, y_smooth_PI_UPPER, y_smooth_PI_LOWER, facecolor='g',
                                         interpolate=True, alpha=0.05)



                        # Figure 3, Residual Plot !!!
                        # For ax5 -> Residuals vs Y_predicted
                        ax5.scatter(y_hat, y_hat_residuals, c='#1E8449', marker='h', label='Residuals', s=100)
                        # make a horizontal dashed line where y = 0, and goes through all width of plot --> .hlines()
                        ax5.hlines(y=0, xmin=y_hat.min(), xmax=y_hat.max(), linestyles='--', color='#17202A')



                        # customize the figures, names etc...
                        ax1.legend(loc='upper left')
                        ax1.set_xlabel('X1-axis')
                        ax1.set_ylabel('Y-axis')
                        ax1.set_title('SCATTER PLOT Y vs X1')

                        ax2.legend(loc='upper left')
                        ax2.set_xlabel('X2-axis')
                        ax2.set_ylabel('Y-axis')
                        ax2.set_title('SCATTER PLOT Y vs X2')

                        ax3.legend(loc='upper left')
                        ax3.set_xlabel('X3-axis')
                        ax3.set_ylabel('Y-axis')
                        ax3.set_title('SCATTER PLOT Y vs X3')

                        ax4.legend(loc='upper left')
                        ax4.set_xlabel('Y PREDICTED-axis')
                        ax4.set_ylabel('Y-axis')
                        ax4.set_title('Y vs Y_PREDICTED SCATTER PLOT & CI 95%, PI 95% Intervals')

                        ax5.legend(loc='lower left')
                        ax5.set_xlabel('Y PREDICTED-axis')
                        ax5.set_ylabel('Residuals')
                        ax5.set_title('Residual Plot Against Y_predicted (y_hat)')

                        fig1.suptitle('DEPENDENT VARIABLE (Y) vs INDEPENDENT VARIABLE(X) SCATTER PLOTS')
                        fig3.suptitle('RESIDUAL SCATTER PLOTS')

                        plt.show()


                elif ask_plots == 'NO':
                    print('')
                    print('Ok, you have decided NOT to see the graphs.')

                else:
                    print('')
                    print('You have written Yes/No wrong, try again !!!')
                    make_plots()

            make_plots()

            exit()

        linear_regression()


multiple_regression()
