# LINEAR REGRESSION WITH USING SKLEARN, WILL GIVE ANOVA TABLE, R, R Pred,R Adj., EVERYTHING IMPORTANT !!!

# THIS IS THE BEST ONE OF THEM ALL !!!

# FINISHED !!!


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

# custom made, made it myself, just for plot analysis and customizing dataset, can be found in github plot analysis and dataset customizing
from DATASET_CUSTOMIZING.DATASET_READING_CUSTOMIZING_INPUT import make_dataset
from PLOT_ANALYSIS.PLOT_ANALYSIS_INPUT import make_analysis

# for making LinearRegression and train/test split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# for making some hard calculation such as p-value for each coefficient and for overall model
import scipy.stats as ss

# for saving and loading a model
import pickle


# this will make the plots look a little nicer
style.use('ggplot')

# show all columns in the dataframe while printing it
pd.set_option('display.max_columns', None)


def Confidence_Prediction_Intervals_of_new_data(X_dataset_with_new_data, Y_pred_with_new_data, t_value, MSE, how_many):

    '''
    # this function is for making calculation for CI and PI bands for NEW DATA

    :param X_dataset_with_new_data: a whole dataset with X but also with new data added to its end
    :param Y_pred_with_new_data: whole array with predicted values for X but also with predicted values for new data for X
    :param t_value: this is the t value extracted from t table --> t(α/2,n−(k+1)) --> a --> threshold chosen by analyst (0.005 most common) | n --> number of observations | k --> number of features (IVs, predictors)
    :param MSE: Mean Squared Error or S ** 2
    :param how_many: how many new data has been input
    :return: Confidence Intervals Upper and Lower 95% & Prediction Intervals Upper and Lower 95% but only for new data that why [-how_many:]
    '''


    '''
    X --> X dataset
    X_T --> X dataset but transposed, row values are column values
    inversion --> (X_T X) ** -1
    '''

    # ----------------------------- X / X_trans / X_trans X inversion ----------------------------------------------
    X_dataset_with_new_data_transposed = X_dataset_with_new_data.transpose()
    X_arr_matrix_multiplication = np.matmul(X_dataset_with_new_data_transposed, X_dataset_with_new_data)
    X_arr_matrix_multiplication_inversion = np.linalg.pinv(X_arr_matrix_multiplication)
    # ----------------------------- X / X_trans / X_trans X inversion ---------------------------------------------


    '''
    Confidence Interval --> y^h ± (t_value ×se(y^h))
    In words --> Sample estimate ± (t-multiplier × standard error)
    
    y^h --> is the "fitted value" or "predicted value" of the response when the predictor values are Xh. (y_pred)
    se(y^h) ---> calculated == sqrt(MSE * (X_Th (X_T X)−1 Xh) -----------> standard error of the fit (se(y_pred))
    Xh --> X dataset with new data added, with everything
    X_Th --> X dataset with new data added, with everything but transposed
    
    
    Prediction Interval --> y^h ± (t_value × s_pred)
    In words --> Sample estimate ± (t-multiplier × standard error)
    
    s_pred --> calculated == sqrt(MSE + se(y^h) ** 2) --> standard error of the prediction
    '''

    # ---------------------------------- CONFIDENCE AND PREDICTION INTERVALS OF NEW DATA ------------------------------

    # here will be stored CI and PI values
    CI_LOWER_values = []
    CI_UPPER_values = []

    PI_LOWER_values = []
    PI_UPPER_values = []

    # go in range of how many of the predicted values are
    for i in range(len(Y_pred_with_new_data)):

        # ----------------------------- STANDARD ERROR OF THE PREDICTED VALUE (OF THE ESTIMATE, se(y_pred)) -----------------------------------------
        standard_error_of_pred_value = np.sqrt(MSE * np.matmul(X_dataset_with_new_data_transposed[:, i], np.matmul(X_arr_matrix_multiplication_inversion, X_dataset_with_new_data[i, :])))
        # ----------------------------- STANDARD ERROR OF THE PREDICTED VALUE (OF THE ESTIMATE, se(y_pred)) -----------------------------------------

        # this will give the array of values which is calculated for each predicted value
        CI_LOWER_95 = np.round(Y_pred_with_new_data - (t_value * standard_error_of_pred_value), 2)
        CI_UPPER_95 = np.round(Y_pred_with_new_data + (t_value * standard_error_of_pred_value), 2)

        # but we are doing this calculation for each predicted value and we take the i'th CI Lower and Upper band and append it to the lists
        CI_LOWER_values.append(CI_LOWER_95[i])
        CI_UPPER_values.append(CI_UPPER_95[i])

        # ----------------------------- COMBINED VARIANCE FOR PREDICTION INTERVAL -----------------------------------------
        s_pred = np.sqrt(MSE + standard_error_of_pred_value ** 2)
        # ----------------------------- COMBINED VARIANCE FOR PREDICTION INTERVAL -----------------------------------------

        # this will give the array of values which is calculated for each predicted value
        PI_LOWER_95 = np.round(Y_pred_with_new_data - (t_value * s_pred), 2)
        PI_UPPER_95 = np.round(Y_pred_with_new_data + (t_value * s_pred), 2)

        # but we are doing this calculation for each predicted value and we take the i'th CI Lower and Upper band and append it to the lists
        PI_LOWER_values.append(PI_LOWER_95[i])
        PI_UPPER_values.append(PI_UPPER_95[i])

    # ---------------------------------- CONFIDENCE AND PREDICTION INTERVALS OF NEW DATA ------------------------------

    # return only CI and PI bands for new values which were added into the dataset's end but take all so [-how_many:]
    return CI_LOWER_values[-how_many:], CI_UPPER_values[-how_many:], PI_LOWER_values[-how_many:], PI_UPPER_values[-how_many:]


def SST_SSE_S_RPred_SEcoeff_CI_PI(X, Y, Y_pred, t_value):

    '''

    :param X: X dataset with all
    :param Y: Y column with all
    :param Y_pred: Predicted Values for all X
    :param t_value: this is the t value extracted from t table --> t(α/2,n−(k+1)) --> a --> threshold chosen by analyst (0.005 most common) | n --> number of observations | k --> number of features (IVs, predictors)
    :return: SST --> Sum of Squares Total | SSE --> Sum of Squares due to Error (Residual) | S --> standard error (the smaller the better) | R_pred --> How well is our model in predicting
            --> | std_of_coefficients --> standard error of each coefficient (intercept and slopes) | CI and PI lower values, only for making plot (when there is only 1 IV)
    '''


    # --------------------- SST/Y_res ---------------

    # to calculate SST --> ∑(y − y_mean)2
    Y_mean = np.mean(Y)
    Y_residual = Y - Y_mean
    Y_residual_squared = Y_residual ** 2
    SST = np.round(np.sum(Y_residual_squared), 5)
    # --------------------- SST/Y_res ---------------


    # -------------------- SSE / Y_Y_pred_res ----------------

    # to calculate SSE --> ∑(y − y_pred)2
    # must be significantly lower than SST !!!
    Y_hat_residuals = Y - Y_pred
    Y_hat_residuals_squared = Y_hat_residuals ** 2
    SSE = np.round(np.sum(Y_hat_residuals_squared), 5)
    # -------------------- SSE / Y_Y_pred_res ----------------


    # ------------------------ MSE / S --------------------------------

    # to calculate MSE (mean squared error)--> SSE / (n - (k + 1)) --> n = number of observations | k --> number of features (IVs, predictors)
    # S (standard error) --> sqrt(MSE)
    MSE = np.round(SSE / (len(Y) - (len(X[0, 1:]) + 1)), 5)
    S = np.round(np.sqrt(MSE), 5)
    # ------------------------ MSE / S --------------------------------


    '''
    X --> X dataset
    X_T --> X dataset but transposed, row values are column values
    inversion --> (X_T X) ** -1
    '''

    # ----------------------------- X / X_trans / X_trans X inversion ----------------------------------------------
    X_transposed = X.transpose()
    X_arr_matrix_multiplication = np.matmul(X_transposed, X)
    X_arr_matrix_multiplication_inversion = np.linalg.pinv(X_arr_matrix_multiplication)
    # ----------------------------- X / X_trans / X_trans X inversion ----------------------------------------------


    # ------------------------------- STANDARD ERROR OF COEFFICIENTS ----------------------------------------

    # X_arr_matrix_multiplication_inversion_DIAGONAL --> (X_T X) ** -1 !! all diagonal values
    # std_of_coefficients --> sqrt(MSE * X_arr_matrix_multiplication_inversion_DIAGONAL)

    X_arr_matrix_multiplication_inversion_DIAGONAL = X_arr_matrix_multiplication_inversion.diagonal()
    std_of_coefficients = np.round(np.sqrt(S ** 2 * X_arr_matrix_multiplication_inversion_DIAGONAL), 5)
    # ------------------------------- STANDARD ERROR OF COEFFICIENTS ----------------------------------------


    '''
    THIS CALCULATION IS EXPLAINED ABOVE ONLY DIFFERENCE IS THAT WE DON'T HAVE ANY NEW DATA ADDED AND WE TAKE ALL CI AND PI BANDS FOR EACH X
    '''
    # ------------------------------------- CONFIDENCE AND PREDICTION INTERVALS 95% ------------------------------------
    CI_LOWER_values = []
    CI_UPPER_values = []

    PI_LOWER_values = []
    PI_UPPER_values = []

    for i in range(len(Y)):

        # ----------------------------- STANDARD ERROR OF THE PREDICTED VALUE (OF THE ESTIMATE, se(y_pred)) -----------------------------------------
        standard_error_of_pred_value = np.sqrt(MSE * np.matmul(X_transposed[:, i], np.matmul(X_arr_matrix_multiplication_inversion, X[i, :])))
        # ----------------------------- STANDARD ERROR OF THE PREDICTED VALUE (OF THE ESTIMATE, se(y_pred)) -----------------------------------------

        CI_LOWER_95 = np.round(Y_pred - (t_value * standard_error_of_pred_value), 5)
        CI_UPPER_95 = np.round(Y_pred + (t_value * standard_error_of_pred_value), 5)

        CI_LOWER_values.append(CI_LOWER_95[i])
        CI_UPPER_values.append(CI_UPPER_95[i])

        # ----------------------------- COMBINED VARIANCE FOR PREDICTION INTERVAL -----------------------------------------
        s_pred = np.sqrt(S ** 2 + standard_error_of_pred_value ** 2)
        # ----------------------------- COMBINED VARIANCE FOR PREDICTION INTERVAL -----------------------------------------

        PI_LOWER_95 = np.round(Y_pred - (t_value * s_pred), 5)
        PI_UPPER_95 = np.round(Y_pred + (t_value * s_pred), 5)

        PI_LOWER_values.append(PI_LOWER_95[i])
        PI_UPPER_values.append(PI_UPPER_95[i])

    # ------------------------------------- CONFIDENCE AND PREDICTION INTERVALS 95% ------------------------------------



    # -------------------------------- PRESS / R PRED / H ---------------------------------

    # PRESS --> Prediction Sum of Squares --> E(Y_hat_residuals_squared / (1 - Hii) ** 2
    # Y_hat_residuals_squared --> (y_observed - y_predicted) ** 2

    # H --> hat matrix --> X (X_T X) ** -1 X_T
    # Hii --> hat matrix but only values in H that are diagonal, ONLY DIAGONAL VALUES !!!
    H = np.matmul(X_arr_matrix_multiplication_inversion, X_transposed)
    H2 = np.matmul(X, H)
    Hii = H2.diagonal()

    PRESS = np.sum(Y_hat_residuals_squared / (1 - Hii) ** 2)

    # R Squared Predicted --> how good is our model at predicting --> 1 - (PRESS / SST)
    R_pred = np.round(1 - (PRESS / SST), 5)
    # -------------------------------- PRESS / R PRED / H ---------------------------------


    return SST, SSE, S, R_pred, std_of_coefficients, CI_LOWER_values, CI_UPPER_values, PI_LOWER_values, PI_UPPER_values



def Linear_Regression(df):

    # make plot analysis if user wishes to do so
    make_analysis(df)


    while True:

        # check for any input error
        try:
            print(df)

            print('')
            print('THESE ARE YOUR COLUMNS NAMES:', list(df.columns))
            print('')
            # ask for column(s) which will be all X values, which will be our features
            ask_IVs = input('Type all column names in the dataset, you want for X (Independent variables / features) (split with comma + space): ').split(', ')

            # if that column that was input does not occur in the dataset, raise an error
            for col_X in ask_IVs:
                if col_X not in list(df.columns):
                    raise Exception

            print('')
            # ask for one single column which will be our label Y (DV)
            ask_DV = input('Type column name in the dataset you want for Y (Dependent variable / label): ').split(', ')

            # if there will be more than one column for label input, raise an error
            if len(ask_DV) != 1:
                raise Exception

            # and again if that label column does not occur in the dataset, raise an error
            for col_y in ask_DV:
                if col_y not in list(df.columns):
                    raise Exception


            print('')
            # ask for the number which will be how many times the scikit learn will make random train/test split and then from all those splits take only the best,
            #  --> the one which has the highest RSquared (.score), because each train/test split is random
            ask_how_many_times = int(input('''How many times do you want to make a random train/test split?
It will take the one best model with its best line and best RSquared, and best overall specs, the best train/test split possible out of the number you specify.
Type a number (min 10 times recommended) : '''))


        except ValueError:
            print('')
            print('Oops, something went wrong, an input must be a whole number, try again !!!')


        except Exception:
            print('')
            print('Oops, at least one the columns you have inout does not appear in the dataset, try again !!!')


        # if everything seems to be working and no error has been raised, break the loop
        else:
            break


    # X_sklearn, this is the 2D array which will include all values in X column(s) the user specified
    X_sklearn = np.array(df[ask_IVs])
    # y_sklearn, this is the 1D array which will include all values in Y column the user specified
    y_sklearn = np.array(df[''.join(ask_DV)])

    # We do this array for some calculations which are necessary to do with also vector full of ones 1
    # 2D array with of ones with shape the length of X, how many instances if has (rows) and only one column in dtype float
    full_ones = np.ones((len(X_sklearn), 1), dtype=float)
    # stack those full ones and X values horizontally, meaning each row will have its 1 as the 1st element
    X_arr_mat = np.hstack([full_ones, X_sklearn])


    while True:

        # those lists will include those coefficients which has the highest RSquared
        slopes_list = []
        intercept_list = []
        RSquared_final = -1

        print('')
        # ask the user if wants to save or load the model
        ask_save_load = input('Do you want to save or load your best model? Save/Load: ').upper()

        # if wants to save the best model
        if ask_save_load == 'SAVE':

            print('')
            # ask for the name you want your best model to be saved
            ask_save_name = input('OK, give me the name of your best model you want to save, it will be saved in this directory: ')


            # go as many times as player asked to do this random train/test split
            for _ in range(ask_how_many_times):

                # split the data, test_size is 20%
                X_sklearn_train, X_sklearn_test, y_sklearn_train, y_sklearn_test = train_test_split(X_sklearn, y_sklearn, test_size=0.2)

                # make an instance for Linear Regression
                model = LinearRegression()
                # and then fit (train) the model using the train split for X and Y
                model.fit(X_sklearn_train, y_sklearn_train)


                # this will include all slopes --> model.coef_
                slopes = model.coef_
                # this will include the intercept --> model.intercept_
                intercept = model.intercept_

                # to find the R squared --> .score() and in parentheses is the test split
                RSquared = np.round(model.score(X_sklearn_test, y_sklearn_test), 5)


                # take only the highest R squared,
                # and each time there is Higher R squared then before, remove the previous R Squared assign the new, clear the lists of coefficients and assign new coefficients with the best accuracy
                if RSquared > RSquared_final:
                    RSquared_final = RSquared

                    slopes_list.clear()
                    intercept_list.clear()

                    slopes_list.append(slopes)
                    intercept_list.append(intercept)

                    # also save that best model using open with() function and using pickle.dump(instance, f)
                    with open(ask_save_name + '.pickle', 'wb') as f:
                        pickle.dump(model, f)

            # if all is done, break the loop
            break


        # if user wants to load the model
        elif ask_save_load == 'LOAD':

            # ask for the model name
            ask_load_name = input('OK, give me the name of the saved model you want to load in (pickle file, f.e., name1.pickle): ')

            # check for any errors
            try:

                # to read that model, first open it in read binary mode and with open('name') function
                pickle_in = open(ask_load_name, 'rb')
                # and to make, to load that file and make an linear regression instance --> pickle.load()
                model = pickle.load(pickle_in)

                # append that model's slopes and intercept to the lists
                slopes_list.append(model.coef_)
                intercept_list.append(model.intercept_)

                # also run this as many times the user asked for to make random train/test split
                for _ in range(ask_how_many_times):
                    X_sklearn_train, X_sklearn_test, y_sklearn_train, y_sklearn_test = train_test_split(X_sklearn, y_sklearn, test_size=0.2)

                    RSquared = model.score(X_sklearn_test, y_sklearn_test)

                    # take the best r squared
                    if RSquared > RSquared_final:
                        RSquared_final = RSquared

                # break the loop if everything seems to be working
                break

            except Exception:
                print('')
                print('Oops, model name does not exist, try all over again !!!')

        else:
            print('')
            print('Oops, wrong input Save/Load, try again !!!')


    # ----------------------- SLOPES AND INTERCEPT ------------------
    slopes_final = np.round(np.array(slopes_list), 5)
    intercept_final = np.round(np.array(intercept_list), 5)
    # ----------------------- SLOPES AND INTERCEPT ------------------


    # -------------------------- Y predicted values -----------------------

    # to calculate the predicted values, the linear regression equation ---> y_predicted = b0 + b1 * x1 + b2 * x2 + ... + bn * xn
    y_pred = intercept_final + np.sum(X_sklearn * slopes_final, axis=1)
    # -------------------------- Y predicted values -----------------------


    # ------------------------------------------------ T VALUE --------------------------------------------------------
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
    if len(y_sklearn) > 100:
        t_value = t_table.at[101, 0.025]
    else:
        t_value = t_table.at[len(y_sklearn) - len(ask_IVs) - 1, 0.025]
    # ------------------------------------------------ T VALUE --------------------------------------------------------


    # -------------------------------------- R-sq (adjusted) ----------------------------------------------------------

    # to calculate the RSquared Adjusted, which is more accurate, because it is also based on the number of features and the number of observations, and also it will be always fewer than original R Squared
    # calculation --> 1 - ((1 - RSquared) * (n - 1)) / (n - 1 - k) --> | n --> number of observations | k --> number of features (IVs)
    RSquared_Adj = np.round(1 - ((1 - RSquared_final) * (len(y_sklearn) - 1)) / (len(y_sklearn) - 1 - len(ask_IVs)), 5)
    # -------------------------------------- R-sq (adjusted) ----------------------------------------------------------


    # ------------------------ SST, SSE, S, R_pred, STANDARD ERROR OF COEFFS, CONFIDENCE AND PREDICTION INTERVALS 95% --------------------------------------------------------

    # call that function which returns those values
    # but for X we don't include the X_sklearn because it lacks the 1st column which is full of ones
    # must contain the 1st column full of ones for making those calculation to get those returned values
    SST, SSE, S, R_pred, std_of_coeffs, CI_LOWER_95, CI_UPPER_95, PI_LOWER_95, PI_UPPER_95 = SST_SSE_S_RPred_SEcoeff_CI_PI(X=X_arr_mat, Y=y_sklearn, Y_pred=y_pred, t_value=t_value)
    # ------------------------ SST, SSE, S, R_pred, STANDARD ERROR OF COEFFS, CONFIDENCE AND PREDICTION INTERVALS 95% --------------------------------------------------------


    # --------------------------------- STANDARD ERROR OF COEFFICIENTS -----------------------------------

    # in std_of_coeffs array the first number is for intercept and the rest is for the slopes
    intercept_std = std_of_coeffs[0]
    slopes_std = std_of_coeffs[1:]
    # --------------------------------- STANDARD ERROR OF COEFFICIENTS -----------------------------------


    # ------------------------- LOWER 95% & UPPER 95% CONFIDENCE INTERVALS FOR SLOPES AND INTERCEPT----------------------------------

    # to calculated the lower and upper confidence intervals for each coefficient
    # coefficient +- (t_value * standard error of coefficient)

    lower_95_intercept = intercept_final - (t_value * intercept_std)
    upper_95_intercept = intercept_final + (t_value * intercept_std)

    lower_95_slopes = slopes_final - (t_value * slopes_std)
    upper_95_slopes = slopes_final + (t_value * slopes_std)
    # ------------------------- LOWER 95% & UPPER 95% CONFIDENCE INTERVALS FOR SLOPES AND INTERCEPT ----------------------------------


    # -------------------------------- F-value ------------------------------------

    # F value --> the bigger it is the better and the more significant our model is
    # calculation --> ((SST - SSE) / k) / S ** 2 --> | SST --> sum of squared total | SSE --> Sum of Squared due to error (residual) | k --> number of features (IVs) | S --> standard error (MSE --> s ** 2)
    F_value = np.round(((SST - SSE) / len(ask_IVs)) / S ** 2, 2)
    # -------------------------------- F-value ------------------------------------


    # --------------------------------------- T-Value/t Stat/t Ratio -------------------------------------------

    # t statistic foe each coefficient --> coefficient / standard error of coefficient
    T_value_intercept = np.round(intercept_final / intercept_std, 3)
    T_values_slopes = np.round(slopes_final / slopes_std, 3)
    # --------------------------------------- T-Value/t Stat/t Ratio -------------------------------------------


    # ------------------------------------- P-Value RIGHT TAIL OF COEFFICIENTS ---------------------------------------

    # p value for each coefficient, the smaller it is the better, must be lower than threshold chosen by analyst (0.005 is the most common threshold) if you want it to be significant
    # we are using scipy.stats module to make those calculation for each coefficient

    p_value_intercept = np.round((1 - ss.t.cdf(abs(T_value_intercept), df=len(y_sklearn) - 1 - len(ask_IVs))) * 2, 4)
    p_value_intercept = list(p_value_intercept)

    p_values_slopes = np.round((1 - ss.t.cdf(abs(T_values_slopes), df=len(y_sklearn) - 1 - len(ask_IVs))) * 2, 4)
    p_values_slopes = list(p_values_slopes.reshape(-1))

    # if p-value of intercept is less than 0.001, write it as '<0.001'
    if p_value_intercept[0] < 0.001:
        p_value_intercept[0] = '<0.001'

    # if p-value of the slope is less than 0.001, write it as '<0.001'
    for index, p_value_slope in enumerate(p_values_slopes):
        if p_value_slope < 0.001:
            p_values_slopes[index] = '<0.001'
    # ------------------------------------- P-Value RIGHT TAIL OF COEFFICIENTS ---------------------------------------


    # -------------------------------------- P-Value OF OVERALL REGRESSION MODEL ----------------------------------------

    # p value for the overall model, the smaller the better, must be lower than threshold chosen by analyst (0.005 is the most common threshold) if you want it to be significant
    # again we are using scipy.stat module to make that calculation
    p_value_model = np.round(ss.f.pdf(F_value, len(ask_IVs), len(y_sklearn) - 1 - len(ask_IVs)), 6)

    # if p-value of overall model is less than 0.001, write it as '<0.001'
    if p_value_model < 0.001:
        p_value_model = '<0.001'
    # -------------------------------------- P-Value OF OVERALL REGRESSION MODEL ----------------------------------------


    # ----------------------------------------- VARIANCE INFLATION FACTOR (VIF) ----------------------------------------------

    '''
    VIF indicates multicollinearity (the correlation between features (the worse the better) the smaller it is, the better (if more than 10 is unacceptable !!!)
    It makes the linear regression between each X in the dataset, and y is not included, so it is like each X that has been chosen will be the y for that calculation
    
    Calculation --> 1 / (1 - RSquared i)
    RSquared i --> is the R squared of the model when y is not included and for each X is included, the smaller it is the better
     --> because we want our model to be the worst, to have the worst accuracy between X, for multicollinearity check
    '''

    # make a copy of X_sklearn because we don't want to modify the orgiinal array
    X_sklearn_VIF = X_sklearn.copy()
    list_VIFs = []

    # if there is more than one feature make VIF calculation, because when there is only one the VIF is the smallest ast it can get --> 1.00
    if len(ask_IVs) > 1:

        # loop and through each features, but don't use it
        for _ in ask_IVs:

            # fit the model .fit(x, y)
            # X_sklearn_VIF[:, [x for x in range(1, len(ask_IVs))]] --> takes all 'columns', take each feature values except for the first feature, which is the 1st column --> x
            # X_sklearn_VIF[:, 0] --> take only the 1st column
            model.fit(X_sklearn_VIF[:, [x for x in range(1, len(ask_IVs))]], X_sklearn_VIF[:, 0])

            # calculate the Rsquared, and than the VIF and then add it to the list
            RSquared_VIF = model.score(X_sklearn_VIF[:, [x for x in range(1, len(ask_IVs))]], X_sklearn_VIF[:, 0])
            VIF = np.round(1 / (1 - RSquared_VIF), 2)
            list_VIFs.append(VIF)

            # but we want to do that for each X column to be the y for some time, for making RSquared
            # so this will takes the 1st column and add it right to the end, move the 1st column to be the last, this shifts till all column have been shifted
            X_sklearn_VIF = np.hstack([X_sklearn_VIF[:, [x for x in range(1, len(ask_IVs))]], X_sklearn_VIF[:, 0].reshape((len(X_sklearn), 1))])

    # if there is only one features, the VIF is 1.00
    elif len(ask_IVs) == 1:
        VIF = 1.00
        list_VIFs.append(VIF)
    # ----------------------------------------- VARIANCE INFLATION FACTOR (VIF) ----------------------------------------------


    print('')
    print(120 * '#')
    print(120 * '#')
    print('')
    print('REGRESSION ANALYSIS:', ''.join(ask_DV), 'VERSUS', ''.join([x + ' | ' for x in ask_IVs]))
    print('')
    print(120 * '#')
    print(120 * '#')
    print('')



    # ---------------------------------------------- ANALYSIS OF VARIANCE (ANOVA) --------------------------------------
    ANOVA_dict = {'Source': ['Regression', 'Residual', 'Total'],
                  'DF': [len(ask_IVs), len(y_sklearn) - 1 - len(ask_IVs), len(y_sklearn) - 1],
                  'SS': [SST - SSE, SSE, SST],
                  'Mean Square': [np.round((SST - SSE) / len(ask_IVs), 5), np.round(S ** 2, 5), ''],
                  'F-Value': [F_value, '', ''],
                  'P-Value': [p_value_model, '', '']}

    ANOVA_df = pd.DataFrame(ANOVA_dict)

    print('''
''')
    print('ANALYSIS OF VARIANCE (ANOVA)')
    print('')
    print(ANOVA_df.to_string(index=False))
    # ---------------------------------------------- ANALYSIS OF VARIANCE (ANOVA) --------------------------------------


    # ------------------------------------------- MODEL SUMMARY DF ----------------------------------------------------------
    model_summary_dict = {'S': S, 'R-sq': str(np.round(RSquared_final * 100, 2)) + '%',
                          'R-sq(adj)': str(np.round(RSquared_Adj * 100, 2)) + '%', 'R-sq(pred)': str(np.round(R_pred * 100, 2)) + '%'}

    model_summary_df = pd.DataFrame(model_summary_dict, index=[0])

    print('''
''')
    print('MODEL SUMMARY')
    print('')
    print(model_summary_df.to_string(index=False))
    # ------------------------------------------- MODEL SUMMARY ----------------------------------------------------------


    # ------------------------------------------------ COEFFICIENTS DF -----------------------------------------------------
    list_terms = ['Intercept'] + ask_IVs
    list_estimates = list(intercept_final) + list(slopes_final.reshape(-1))
    list_std_error_coeffs = [intercept_std] + list(slopes_std)
    list_t_stats = list(T_value_intercept) + list(T_values_slopes.reshape(-1))
    list_p_values_coeffs = p_value_intercept + p_values_slopes
    list_lower_95_coeffs = list(lower_95_intercept) + list(lower_95_slopes.reshape(-1))
    list_upper_95_coeffs = list(upper_95_intercept) + list(upper_95_slopes.reshape(-1))


    coefficients_dict = {'Term': list_terms,
                         'Estimate': list_estimates,
                         'Std. Error': list_std_error_coeffs,
                         'T-Value': list_t_stats,
                         'P-Value': list_p_values_coeffs,
                         'Lower 95%': list_lower_95_coeffs,
                         'Upper 95%': list_upper_95_coeffs,
                         'VIF': [''] + list_VIFs}

    Coefficients_df = pd.DataFrame(coefficients_dict)

    print('''
''')
    print('COEFFICIENTS')
    print('')
    print(Coefficients_df.to_string(index=False))
    # ------------------------------------------------ COEFFICIENTS DF -----------------------------------------------------


    # --------------------------------------------- REGRESSION EQUATION ------------------------------------------------
    equation_slopes_part = []

    for slope, IV in zip(list(slopes_final.reshape(-1)), ask_IVs):

        if slope > 0:
            equation_slopes_part.append(' + ' + str(slope) + ' ' + IV)

        else:
            equation_slopes_part.append(' ' + str(slope) + ' ' + IV)

    regression_equation = ''.join(ask_DV) + ' = ' + ''.join([str(intercept_final)[1:-1]] + equation_slopes_part)

    print('''
''')
    print('REGRESSION EQUATION')
    print('')
    print(regression_equation)
    # --------------------------------------------- REGRESSION EQUATION ------------------------------------------------



    # !!! ------------------------------------------------ PLOTS GRAPHS ---------------------------------------------------- !!!
    # !!! ------------------------------------------------ PLOTS GRAPHS ---------------------------------------------------- !!!
    # !!! ------------------------------------------------ PLOTS GRAPHS ---------------------------------------------------- !!!


    # ----------------------------------- Y PREDICTED VS X VALUE (ONLY WHEN 1 IV) -------------------------------------------

    # if there is only one feature (IV) make also plot between y predicted and x values
    if len(ask_IVs) == 1:

        figure = plt.figure(figsize=(6, 6))
        axes = plt.subplot2grid((6, 1), (2, 0), rowspan=5, colspan=1, fig=figure)

        # facecolor='none', edgecolors='#CB0202' the dots will be 'empty', only edgecolor is included
        axes.scatter(X_sklearn.reshape(-1), y_sklearn, marker='o', facecolor='none', edgecolors='#CB0202', s=80, label='Observed Values')
        axes.plot(X_sklearn.reshape(-1), y_pred, color='#1F1F1E', label='Regression Line')

        axes.legend(loc='best')
        axes.set_xlabel(''.join(ask_IVs))
        axes.set_ylabel(''.join(ask_DV))
        figure.suptitle('\n\n\nLINEAR REGRESSION LINE VS OBSERVED VALUES')
    # ----------------------------------- Y PREDICTED VS X VALUE (ONLY WHEN 1 IV) -------------------------------------------


    # -------------------------------------- Y OBSERVED VS Y PREDICTED (WHEN MORE THAN 1 IV) -------------------------------------------------------

    # plot between y observed (y-axis) and y predicted (x-axis)
    # this is a good plot, because the better the correlation coefficient is, the more significant our model is, we want the predicted values to be the same as observed values (which will never happen , but you know what i mean)

    fig1 = plt.figure(figsize=(6, 6))
    ax1 = plt.subplot2grid((6, 1), (2, 0), rowspan=5, colspan=1, fig=fig1)

    ax1.scatter(y_pred, y_sklearn, marker='o', facecolor='none', edgecolors='#CB0202', s=80, label=''.join(ask_DV) + '\nvs\nPredicted Value')

    df_corr = pd.DataFrame({'Y_pred': y_pred, 'Y': y_sklearn}).corr()
    # take the correlation value
    corr = np.round(df_corr.iloc[0, 1], 4)

    # make a text which will contain correlation value
    # bbox={'facecolor': 'blue', 'alpha': 0.2, 'pad': 10} --> will make in look like in square, in box
    ax1.text(0.15, 0.72, 'Correlation (r) = ' + str(corr),
             fontsize=12, transform=plt.gcf().transFigure,
             bbox={'facecolor': 'blue', 'alpha': 0.2, 'pad': 10})

    ax1.legend(loc='best')
    ax1.set_xlabel('Predicted Value')
    ax1.set_ylabel(''.join(ask_DV))
    fig1.suptitle('\n\n\n' + ''.join(ask_DV) + ' TO Y PREDICTED SCATTER PLOT')
    # -------------------------------------- Y OBSERVED VS Y PREDICTED -------------------------------------------------------


    # ------------------------------------------ RESIDUAL PLOT ---------------------------------------------

    # residual plot, the closer the dots are to the hline, which is 0 the better, we want the smallest distance possible, the absolute 0
    # the distance between observed points and the regression line must be as smallest as it can be, so 0

    fig2 = plt.figure(figsize=(6, 6))
    ax2 = plt.subplot2grid((6, 1), (2, 0), rowspan=5, colspan=1, fig=fig2)

    ax2.scatter(y_pred, y_sklearn - y_pred, marker='o', facecolor='none', edgecolors='#13005E', s=80, label='Residual\nvs\nPredicted Value')
    ax2.hlines(y=0, xmin=y_pred.min(), xmax=y_pred.max(), linestyles='--', color='#230D1B')

    ax2.legend(loc='best')
    ax2.set_xlabel('Predicted Value')
    ax2.set_ylabel('Residual')
    fig2.suptitle('\n\n\nRESIDUAL BY PREDICTED FOR ' + ''.join(ask_DV))
    # ------------------------------------------ RESIDUAL PLOT ---------------------------------------------


    # ------------------------------------- CI 95% & PI 95% PLOT (ONLY WHEN 1 IV) -----------------------------------------------

    # make CI and PI bands plot if there is only one feature in the model

    if len(ask_IVs) == 1:

        # make the copies of X_sklearn values (also must be 1D array so reshape(-1)), y_sklearn values, y_predicted values, and CI and PI lower and upper bands
        # then sort all values in ascending order, for making a plot, but only in ascending order when slope is positive

        X_sklearn_copy = np.sort(X_sklearn.reshape(-1))
        y_sklearn_copy = np.sort(y_sklearn)
        y_pred_copy = np.sort(y_pred)

        CI_UPPER_95_copy = np.sort(CI_UPPER_95)
        CI_LOWER_95_copy = np.sort(CI_LOWER_95)

        PI_UPPER_95_copy = np.sort(PI_UPPER_95)
        PI_LOWER_95_copy = np.sort(PI_LOWER_95)

        # if slope is negative sort all those values in descending order
        if slopes_final < 0:
            X_sklearn_copy[::-1].sort()
            y_sklearn_copy[::-1].sort()
            y_pred_copy[::-1].sort()

            CI_UPPER_95_copy[::-1].sort()
            CI_LOWER_95[::-1].sort()

            PI_UPPER_95[::-1].sort()
            PI_LOWER_95[::-1].sort()


        fig3 = plt.figure(figsize=(6, 6))
        ax3 = plt.subplot2grid((6, 1), (2, 0), rowspan=5, colspan=1, fig=fig3)

        ax3.scatter(X_sklearn_copy, y_sklearn_copy,  marker='o', facecolor='none', edgecolors='#CB0202', s=80, label='Observed Values')
        ax3.plot(X_sklearn_copy, y_pred_copy, color='#1F1F1E', label='Regression Line')

        ax3.plot(X_sklearn_copy, CI_UPPER_95_copy, color='#342296')
        ax3.plot(X_sklearn_copy, CI_LOWER_95_copy, color='#342296')

        ax3.plot(X_sklearn_copy, PI_UPPER_95_copy, color='#8F88B8', linestyle='--')
        ax3.plot(X_sklearn_copy, PI_LOWER_95_copy, color='#8F88B8', linestyle='--')


        # do empty plots because fill_between does not have label parameter, but we want that to be in our legend for the user to know what is what
        ax3.plot([], [], color='#342296', label='95% Confidence Limit')
        ax3.plot([], [], color='#8F88B8', linestyle='--', label='95% Prediction Limit')

        # fill between CI LOWER and CI UPPER in brighter color, and PI lower and PI upper in less brighter color
        ax3.fill_between(X_sklearn_copy, CI_UPPER_95_copy, CI_LOWER_95_copy, facecolor='#342296', interpolate=True, alpha=0.4)
        ax3.fill_between(X_sklearn_copy, PI_UPPER_95_copy, PI_LOWER_95_copy, facecolor='#8F88B8', interpolate=True, alpha=0.2)

        ax3.legend(loc='best')
        ax3.set_xlabel(''.join(ask_IVs))
        ax3.set_ylabel(''.join(ask_DV))
        fig3.suptitle('\n\n\nCONFIDENCE & PREDICTION INTERVALS 95%')
    # ------------------------------------- CI 95% & PI 95% PLOT -----------------------------------------------


    # ------------------------------------------ 3D SCATTER PLOT (ONLY WHEN 2 IVs) ---------------------------------------

    # if there are exactly 2 features in the model, make an 3D plot
    if len(ask_IVs) == 2:

        # fig_3d.add_subplot(111, projection='3d') --> this will tell matplotlib that we are making a 3D plot
        fig_3d = plt.figure(figsize=(6, 6))
        ax_3d = fig_3d.add_subplot(111, projection='3d')

        # make scatter plot --> x | y | z --> where z is our label (Y) column
        ax_3d.scatter(X_sklearn[:, 0], X_sklearn[:, 1], y_sklearn, marker='o', c='#FF0000')


        # ------------------------------------------- MAKE SURFACE PLOT (3D REGRESSION LINE) ---------------------------------
        x_surf, y_surf = np.meshgrid(np.linspace(X_sklearn[:, 0].min(), X_sklearn[:, 0].max(), 100),
                                     np.linspace(X_sklearn[:, 1].min(), X_sklearn[:, 1].max(), 100))
        only_x = pd.DataFrame({'X1': x_surf.ravel(), 'X2': y_surf.ravel()})

        model.fit(X_sklearn, y_sklearn)
        fittedY = model.predict(only_x)
        fittedY = np.array(fittedY)

        ax_3d.plot_surface(x_surf, y_surf, fittedY.reshape(x_surf.shape), color='b', rstride=2, cstride=2, alpha=0.4)
        # ------------------------------------------- MAKE SURFACE PLOT (3D REGRESSION LINE) ---------------------------------

        ax_3d.set_xlabel(ask_IVs[0])
        ax_3d.set_ylabel(ask_IVs[1])
        ax_3d.set_zlabel(''.join(ask_DV))
        ax_3d.set_title('3D SCATTER PLOT ' + ''.join(ask_DV) + ' vs ' + ask_IVs[0] + ', ' + ask_IVs[1] + '\n&\nREGRESSION LINE')
    # ------------------------------------------ 3D SCATTER PLOT (ONLY WHEN 2 IVs) ---------------------------------------

    plt.show()

    # !!! ------------------------------------------------ PLOTS GRAPHS ---------------------------------------------------- !!!
    # !!! ------------------------------------------------ PLOTS GRAPHS ---------------------------------------------------- !!!
    # !!! ------------------------------------------------ PLOTS GRAPHS ---------------------------------------------------- !!!


    # -------------------------------------------- PREDICT INPUT VALUE / NEW DATA -------------------------------------------------
    def predict_new_data():

        while True:

            print('')
            # ask the user if wants to make prediction for new data that will be input
            ask_new_data = input('Do you wanna make some predictions on new data, which you wanna type here? Yes/No: ').upper()

            if ask_new_data == 'YES':

                # make an empty array which will include all new data for X
                new_data_array = np.array([])

                # ask for each IV, feature that is in the model for the new data
                for independent_variable in ask_IVs:
                    ask_new_data_IV = input('New Data (split with comma + space) | ' + independent_variable + ' : ').split(', ')

                    # if the data can't be converted into floats, raise an error
                    try:
                        for new_data in ask_new_data_IV:
                            _ = float(new_data)

                    except ValueError:
                        print('')
                        print('Oops something went wrong with the new data you have input, they must be numeric and well separated, try again !!!')
                        predict_new_data()

                    # make a numpy array of it and in float dtype
                    ask_new_data_IV_array = np.array(ask_new_data_IV, dtype=float)

                    # append that array with new data into the empty numpy array
                    new_data_array = np.append(new_data_array, ask_new_data_IV_array)

                # check for any error
                try:
                    # but all data are all in one row, the array is not sorted and not in 2D
                    # so to reshape we have to first reshape that each row will be for each feature
                    # and we want it to be that each column will be for each feature soo make a transpose
                    new_data_array = new_data_array.reshape((len(ask_IVs), len(ask_new_data_IV_array))).transpose()

                    # and also make an array which will the exact same as the array with the new data, but also the 1st column will be full of one, for making some calculations with it
                    full_ones_new_data = np.ones((len(new_data_array), 1), dtype=float)
                    X_arr_mat_new_data = np.hstack([full_ones_new_data, new_data_array])

                except ValueError:
                    print('')
                    print('Oops something went wrong, each independent variable group must have the same number of number, try again !!!')
                    predict_new_data()


                # calculate the predicted values using slopes and intercept, using regression line equation
                predicted_values_new_data = intercept_final + np.sum(new_data_array * slopes_final, axis=1)

                # how many predictions we have
                how_many_new_data_predictions = len(predicted_values_new_data)

                # -------------------------------------- CONFIDENCE AND PREDICTION INTERVALS OF NEW DATA ------------------------------------------------

                # add that new data array (with ones) into the array with all X values observed (also with ones)
                X_data_with_new_data = np.vstack([X_arr_mat, X_arr_mat_new_data])
                # also add predicted values for new data into the predicted values for observed values
                Y_pred_with_new_data = np.hstack([y_pred, predicted_values_new_data])


                # calculate the CI and PI lower and upper for new data
                CI_LOWER_95_new_data, CI_UPPER_95_new_data, PI_LOWER_95_new_data, PI_UPPER_95_new_data = Confidence_Prediction_Intervals_of_new_data(
                    X_dataset_with_new_data=X_data_with_new_data,
                    Y_pred_with_new_data=Y_pred_with_new_data,
                    t_value=t_value, MSE=S ** 2, how_many=how_many_new_data_predictions)


                # --------------------------------------- NEW DATA SPECIFICS DATAFRAME ----------------------------------------------

                # makes a dataframe only with new data for X, with column names as the dataset has for X
                x_new_data_df = pd.DataFrame(new_data_array, columns=ask_IVs)

                # make a dataframe with pred values and with CI and PI upper and lower values
                new_data_pred_dict = {'Pred': predicted_values_new_data,
                                      '95% CI LOWER': CI_LOWER_95_new_data,
                                      '95% CI UPPER': CI_UPPER_95_new_data,
                                      '95% PI LOWER': PI_LOWER_95_new_data,
                                      '95% PI UPPER': PI_UPPER_95_new_data}
                new_data_pred_df = pd.DataFrame(new_data_pred_dict)


                # and to merge those dataframes use .join()
                new_data_x_with_pred = x_new_data_df.join(new_data_pred_df)

                print('')
                print('')
                print('NEW OBSERVATIONS WITH THEIR PREDICTIONS')
                print('')
                print(new_data_x_with_pred.to_string(index=False))
                print('')
                # --------------------------------------- NEW DATA SPECIFICS DATAFRAME ----------------------------------------------


                print('')
                ask_again = input('Do you wanna try predicting new values again? Yes/No: ').upper()

                if ask_again == 'YES':
                    print('OK, starting again.')

                elif ask_again == 'NO':
                    print('OK, no more predicting new values.')
                    break

                else:
                    print('Oops, something went wrong with your input (Yes/No), try again !!!')


            elif ask_new_data == 'NO':
                print('')
                print('OK, there will be no prediction making.')
                break

            else:
                print('')
                print('Oops, something went wrong with your input (Yes/No), try again !!!')


    # -------------------------------------------- PREDICT INPUT VALUE / NEW DATA -------------------------------------------------

    predict_new_data()


# read or make some customization for the dataset the user wants to use in this linear regression model
# df = make_dataset()

# Linear_Regression(df)
