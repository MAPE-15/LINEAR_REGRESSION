# SIMPLE LINEAR REGRESSION CALCULATOR, GRAPH MAKER WITHOUT USING SCIKIT LEARN
# FINISHED !!!

# This is meant to create a linear regression based on one IV

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# we use specific art style how our plots gonna look like
plt.style.use('ggplot')

def regression():
    print('')
    ask = input('OK, you have decided to try this SIMPLE LINEAR REGRESSION CALCULATOR, want to proceed? Yes/No: ').upper()

    if ask == 'YES':
        print('')
        x_values = input('Independent variable values (x values): ').split(', ')
        y_values = input('Dependent variable values (y values): ').split(', ')

        # Here are some example values you can try it on
        # x1 = 1.7, 1.5, 2.8, 5, 1.3, 2.2, 1.3
        # y1 = 368, 340, 665, 954, 331, 556, 376

        # x2 = 34, 108, 64, 88, 99, 51
        # y2 = 5, 17, 11, 8, 14, 5

        # x3 = 4, 1, 3, 6, 1, 3, 3, 2, 5, 3
        # y3 = 7, 5.4, 6.6, 7.4, 4.8, 6.4, 7, 5.6, 7.3, 6.4

        # here some example with csv file
        # df = pd.read_csv('insurance2.csv', delimiter=',')
        # x_values = []
        # y_values = []
        # for age, bmi in zip(df['age'], df['bmi']):
        #     x_values.append(age)
        #     y_values.append(bmi)


        # check for any errors
        try:
            # convert those values input into floats if possible
            x_values = [float(i) for i in x_values]
            y_values = [float(i) for i in y_values]

        # when error occurs let him/her try again
        except ValueError:
            print('')
            print('You have passed something wrong in the input, try again the program !!!')
            regression()

        # if there are gonna be more x values than y values or vise versa raise an error
        if (len(x_values) > len(y_values)) or (len(x_values) < len(y_values)):
            print('')
            print('For each variable must be same amount of numbers, try again !!!')
            regression()

        # if he/she only input 2 or less values raise an error too
        if len(y_values) <= 2:
            print('')
            print('You must pass more amount of values than 2, try again !!!')
            regression()

        # !!! IF THERE ARE MORE THAN 30 VALUES AND YOU WANT ONLY FIRST 30 VALUES NO MORE
        # elif len(y_values) > 30:
        #     x_values = x_values[:31]
        #     y_values = y_values[:31]


        # turn those lists into numpy arrays
        # NUMPY ARRAYS
        x = np.array(x_values)
        y = np.array(y_values)


        # REGRESSION EQUATION CALCULATIONS !!!

        # ------------------------START---------------------------- #
        x_mean = x.mean()
        y_mean = y.mean()

        # (Xi - X bar)
        x_residual = x - x_mean
        # (Yi - Y bar)
        y_residual = y - y_mean

        # (Xi - X bar) ** 2
        x_residual_squared = x_residual ** 2
        # (Yi - Y bar) ** 2
        y_residual_squared = y_residual ** 2

        # E (summation symbol) --> E(Xi - X bar) ** 2
        x_residual_squared_SUM = np.sum(x_residual_squared)
        # E(Yi - Y bar) ** 2 --> SST --> Sum of squares Total
        SST = np.sum(y_residual_squared)
        SST = np.round(SST, 5)

        # (Xi - X bar) * (Yi - Y bar)
        XY_residuals_multiplied = x_residual * y_residual
        # E(Xi - X bar) * (Yi - Y bar)
        XY_residuals_multiplied_SUM = np.sum(XY_residuals_multiplied)


        # !!! B1 --> SLOPE
        b1_slope = XY_residuals_multiplied_SUM / x_residual_squared_SUM
        b1_slope = np.round(b1_slope, 5)

        # b1 * x bar (x_mean)
        slope_X_mean_multiplied = b1_slope * x_mean
        print(slope_X_mean_multiplied)
        # !!! B0 --> INTERCEPT
        b0_intercept = y_mean - slope_X_mean_multiplied
        b0_intercept = np.round(b0_intercept, 5)


        print('')
        if b1_slope > 0:
            regression_equation = 'ŷ = ' + str(b0_intercept) + ' + ' + str(b1_slope) + '* X'
            print('REGRESSION LINE EQUATION: ' + regression_equation)

        elif b1_slope < 0:
            regression_equation = 'ŷ = ' + str(b0_intercept) + ' ' + str(b1_slope) + ' * X'
            print('REGRESSION LINE EQUATION: ' + regression_equation)

        elif b1_slope == 0 and b0_intercept != 0:
            regression_equation = 'ŷ = ' + str(b0_intercept)
            print('REGRESSION LINE EQUATION: ' + regression_equation)

        elif b1_slope != 0 and b0_intercept == 0:
            regression_equation = 'ŷ = ' + str(b1_slope)
            print('REGRESSION LINE EQUATION: ' + regression_equation)

        else:
            regression_equation = 'ŷ = 0'
            print('REGRESSION LINE EQUATION: ' + regression_equation)

        # --------------------------------END-------------------------------#


        # MORE INFO ABOUT MODEL CALCULATIONS !!!
        # ---------------------------------------START----------------------------------- #


        # ŷ = b1 * x + b0
        y_hat = b1_slope * x + b0_intercept

        # (Yi - ŷ)
        y_hat_residuals = y - y_hat
        # (Yi - ŷ) ** 2
        y_hat_residuals_squared = y_hat_residuals ** 2
        # E(Yi - ŷ) ** 2 --> SSE --> Sum of squares due to Error
        SSE = np.sum(y_hat_residuals_squared)
        SSE = np.round(SSE, 5)
        # SST = SSR + SSE --> SSR = SST - SSE --> Sum of Squares regression
        SSR = SST - SSE
        SSR = np.round(SSR, 5)

        # R**2 CORRELATION --> SSR / SST
        RSquared = np.round(SSR / SST, 5)

        # RSquared Adjusted --> 1 - (1 - RSquared)*(n - 1) / (n - p - 1) --> n = how many observations; p = how many IVs
        RSquared_Adj = np.round(1 - ((1 - RSquared) * (len(y_values) - 1)) / (len(y_values) - 2), 5)

        # PRESS --> predicted residual sum of squares --> Σ(ei / (1-hi))**2
        # ei == y - y_hat --> y_hat_residuals
        # hi == 1/n + x_residual_squared / x_residual_squared_SUM
        h_leverage = 1 / len(y_values) + (x_residual_squared / x_residual_squared_SUM)
        PRESS = np.sum((y_hat_residuals / (1 - h_leverage)) ** 2)
        RSquared_Pred = np.round(1 - PRESS / SST, 5)


        # variance (s ** 2)/MSE (Mean Squared Error)
        MSE = SSE / (len(y_values) - 2)
        MSE = np.round(MSE, 5)

        # Standard Error (s)/ sqrt(MSE)
        standard_error = np.round(np.sqrt(MSE), 5)

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


        # 95% CI FOR SLOPE CALCULATIONS !!!
        # ----------------------START----------------------- #

        # get a value from a specific cell from that file --> .at[index_name, column_name]
        # if there are more than 100 values to predict (which is bad) then set the last row and its cell in t_table dataset
        # otherwise count with number of observations
        if len(y_values) > 100:
            t_value = t_table.at[101, 0.025]
        else:
            t_value = t_table.at[len(y_values) - 2, 0.025]

        # Standard deviation of the slope (Sb1)
        CI_standard_deviation = np.round(standard_error / np.sqrt(x_residual_squared_SUM), 7)
        # 95% LOWER SLOPE
        Lower_95_slope = np.round(b1_slope - (t_value * CI_standard_deviation), 7)
        # 95% UPPER SLOPE
        Upper_95_slope = np.round(b1_slope + (t_value * CI_standard_deviation), 7)

        print('')
        print('We are 95% confident that the interval == ' + '(' + str(Lower_95_slope) + '; ' + str(Upper_95_slope) + ')' + '\ncontains the true slope of the regression line.')

        # --------------------END---------------------------- #


        # 95% CI, Confidence Bands, based on mean; CALCULATIONS !!!
        # ---------------------START----------------------------- #

        # Standard deviation of estimator (Sy_hat_star)
        under_root_part = (1 / len(y_values)) + (x_residual_squared / x_residual_squared_SUM)
        CI_BANDS_standard_deviation = np.round(standard_error * np.sqrt(under_root_part), 7)
        # 95% LOWER CI
        CI_LOWER_95 = np.round(y_hat - (t_value * CI_BANDS_standard_deviation), 7)
        # 95% UPPER CI
        CI_UPPER_95 = np.round(y_hat + (t_value * CI_BANDS_standard_deviation), 7)

        # ---------------------------END-------------------------- #


        # 95% PI, Prediction Bands, based on individual value; CALCULATIONS !!!
        # ----------------------------START---------------------------- #

        # Standard deviation due to prediction (Spred)
        squared_S_pred = standard_error ** 2 + CI_BANDS_standard_deviation ** 2
        S_pred = np.round(np.sqrt(squared_S_pred), 7)
        # 95% LOWER PI
        PI_LOWER_95 = np.round(y_hat - (t_value * S_pred), 7)
        # 95% UPPER PI
        PI_UPPER_95 = np.round(y_hat + (t_value * S_pred), 7)

        # ----------------------------END---------------------------- #

        # ----------------------------------------------END--------------------------------------------- #

        # make a dictionary with keys as column names and values as the value in that column
        dictionary_pandas_multiple = {'x': x,
                                      'y': y,
                                      'y_hat': y_hat,
                                      '95% CI Lower': CI_LOWER_95,
                                      '95% CI Upper': CI_UPPER_95,
                                      '95% PI Lower': PI_LOWER_95,
                                      '95% PI Upper': PI_UPPER_95}
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
        dictionary_pandas_multiple_3 = {'Standard Error': standard_error,
                                        'RSquared': RSquared,
                                        'RSquared adj.': RSquared_Adj,
                                        'RSquared pred.': RSquared_Pred}
        df3 = pd.DataFrame(dictionary_pandas_multiple_3, index=[0])



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

                # we are going to have 2 figures, so this is what we do
                fig1, ax1 = plt.subplots()
                fig2, ax2 = plt.subplots()
                fig3, (ax3, ax4) = plt.subplots(1, 2)


                # Figure 1, Simple Linear Regression, and Observed Values
                # Observed Values
                ax1.scatter(x, y, c='#6f029e', marker='D', label='Observed Values')
                # Linear regression with Predicted Values
                ax1.scatter(x, y_hat, color='k', label='Predicted Values')
                ax1.plot(x, y_hat, color='k', label='Regression Line')


                # Figure 2, Regression Line and 95% CI and PI Band !!!
                # we sort values 'because we want that plot, line to be on the same road, plynula ciara
                # Observed values
                ax2.scatter(x, y, c='#6f029e', marker='D', label='Observed Values')
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
                ax2.fill_between(np.sort(x), np.sort(CI_UPPER_95), np.sort(CI_LOWER_95), facecolor='r', interpolate=True, alpha=0.2)
                ax2.fill_between(np.sort(x), np.sort(PI_UPPER_95), np.sort(PI_LOWER_95), facecolor='g', interpolate=True, alpha=0.05)


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
                ax1.set_xlabel('x-axis')
                ax1.set_ylabel('y-axis')
                ax1.set_title('Simple Linear Regression')

                ax2.legend(loc='upper left')
                ax2.set_xlabel('x-axis')
                ax2.set_ylabel('y-axis')
                ax2.set_title('Regression Line and 95% CI and PI Band')

                ax3.legend(loc='lower left')
                ax3.set_xlabel('x-axis')
                ax3.set_ylabel('residuals')
                ax3.set_title('Residual Plot Against X (IV)')

                ax4.legend(loc='lower left')
                ax4.set_xlabel('y_predicted - axis')
                ax4.set_ylabel('residuals')
                ax4.set_title('Residual Plot Against Y_predicted (y_hat)')

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

        # to see different values based on model predictions
        def individual():

            print('')
            ask_individual = input('Do you want to see the prediction of specific value(s) and see the information based on model? Yes/No: ').upper()

            if ask_individual == 'YES':
                print('')
                # pass the values you want to see predictions from
                individual_values = input('OK, so give me some value (IVs) you wanna see the prediction from: ').split(', ')

                # check for error
                try:
                    # convert those values into floats if possible
                    x_individual = [float(i) for i in individual_values]

                    # make a numpy array from it
                    x_indiv_arr = np.array(x_individual)
                    # calculate the predicted values based on our model equation
                    y_pred = b1_slope * x_indiv_arr + b0_intercept

                    # make a pandas DataFrame which will include those values he/she passed and the prediction of them
                    df3 = pd.DataFrame({'X_specified': x_indiv_arr,
                                        'Y_predicted': y_pred})

                    print('')
                    print('YOUR SPECIFIED X VALUES (IVs) AND THEIR PREDICTED VALUES BASED ON MODEL')
                    print('')
                    # .to_string(index=False) --> will print the DataFrame without an index
                    print(df3.to_string(index=False))

                except:
                    print('')
                    print('Oops something went wrong with calculations, try again your input numbers !!!')
                    individual()


            elif ask_individual == 'NO':
                print('')
                print('Ok, you have decided not to see the predicted values based on your input.')

            else:
                print('')
                print('You have written Yes/No wrong, try again !!!')
                individual()

        individual()

        print('')
        print('THE END')
        exit()


    elif ask == 'NO':
        print('')
        print('Ok, you have decided not to try this program shutting down...')
        exit()

    else:
        print('')
        print('You have written Yes/No wrong, try again!!!')
        regression()


regression()


