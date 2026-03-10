'''linear_regression.py
Subclass of Analysis that performs linear regression on data
YOUR NAME HERE
CS 252: Mathematical Data Analysis Visualization
Spring 2026
'''
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

import analysis


class LinearRegression(analysis.Analysis):
    '''Perform and store linear regression and related analyses'''

    def __init__(self, data):
        '''LinearRegression class constructor

        Parameters:
        -----------
        data: Data object. Contains all data samples and variables in a dataset.
        '''
        super().__init__(data)

        # ind_vars: Python list of strings.
        #   1+ Independent variables (predictors) entered in the regression.
        self.ind_vars = None
        # dep_var: string. Dependent variable predicted by the regression.
        self.dep_var = None

        # A: ndarray. shape=(num_data_samps, num_ind_vars)
        #   Matrix for independent (predictor) variables in linear regression
        self.A = None

        # y: ndarray. shape=(num_data_samps, 1)
        #   Vector for dependent variable (true values) being predicted by linear regression
        self.y = None

        # R2: float. R^2 statistic
        self.R2 = None

        # Mean squared error (MSE). float. Measure of quality of fit
        self.mse = None

        # slope: ndarray. shape=(num_ind_vars, 1)
        #   Regression slope(s)
        self.slope = None
        # intercept: float. Regression intercept
        self.intercept = None
        # residuals: ndarray. shape=(num_data_samps, 1)
        #   Residuals from regression fit
        self.residuals = None

        # p: int. Polynomial degree of regression model (Week 2)
        self.p = 1

    def linear_regression(self, ind_vars, dep_var, method='scipy', p=1):
        '''Performs a linear regression on the independent (predictor) variable(s) `ind_vars`
        and dependent variable `dep_var` using the method `method`.

        Parameters:
        -----------
        ind_vars: Python list of strings. 1+ independent variables (predictors) entered in the regression.
            Variable names must match those used in the `self.data` object.
        dep_var: str. 1 dependent variable entered into the regression.
            Variable name must match one of those used in the `self.data` object.
        method: str. Method used to compute the linear regression. Here are the options:
            'scipy': Use scipy's lstsq function.
            'normal': Use normal equations.
            'qr': Use QR factorization

        TODO:
        - Use your data object to select the variable columns associated with the independent and
        dependent variable strings.
        - Perform linear regression using the appropriate method.
        - Compute R^2 on the fit and the residuals.
        - By the end of this method, all instance variables should be set (see constructor).

        NOTE: Use other methods in this class where ever possible (do not write the same code twice!)
        '''
        self.ind_vars = list(ind_vars)
        self.dep_var = dep_var
        self.p = p

        self.A = self.data.select_data(self.ind_vars)
        self.y = self.data.select_data([self.dep_var])

        if self.p > 1:
            if self.A.shape[1] != 1:
                raise ValueError('Polynomial regression only supports one independent variable.')
            A_use = self.make_polynomial_matrix(self.A, self.p)
        else:
            A_use = self.A

        if method == 'scipy':
            coeffs = self.linear_regression_scipy(A_use, self.y)
        elif method == 'normal':
            coeffs = self.linear_regression_normal(A_use, self.y)
        elif method == 'qr':
            coeffs = self.linear_regression_qr(A_use, self.y)
        else:
            raise ValueError(f"Unknown regression method '{method}'. Expected one of: 'scipy', 'normal', 'qr'.")

        coeffs = np.asarray(coeffs)
        if coeffs.ndim == 1:
            coeffs = coeffs.reshape(-1, 1)

        self.slope = coeffs[:-1, :]
        self.intercept = float(coeffs[-1, 0])

        y_pred = self.predict()
        self.residuals = self.compute_residuals(y_pred)
        self.R2 = self.r_squared(y_pred)
        self.mse = self.compute_mse()

    def linear_regression_scipy(self, A, y):
        '''Performs a linear regression using scipy's built-in least squares solver (scipy.linalg.lstsq).
        Solves the equation y = Ac for the coefficient vector c.

        Parameters:
        -----------
        A: ndarray. shape=(num_data_samps, num_ind_vars).
            Data matrix for independent variables.
        y: ndarray. shape=(num_data_samps, 1).
            Data column for dependent variable.

        Returns
        -----------
        c: ndarray. shape=(num_ind_vars+1, 1)
            Linear regression slope coefficients for each independent var PLUS the intercept term
        '''
        A = np.asarray(A)
        y = np.asarray(y)

        if y.ndim == 1:
            y = y.reshape(-1, 1)

        ones = np.ones((A.shape[0], 1))
        A_aug = np.hstack((A, ones))

        lstsq_result = scipy.linalg.lstsq(A_aug, y)
        if lstsq_result is None:
            raise ValueError('scipy.linalg.lstsq failed to return a solution.')
        c = lstsq_result[0]
        c = np.asarray(c)
        if c.ndim == 1:
            c = c.reshape(-1, 1)

        return c

    def linear_regression_normal(self, A, y):
        '''Performs a linear regression using the normal equations.
        Solves the equation y = Ac for the coefficient vector c.

        See notebook for a refresher on the equation

        Parameters:
        -----------
        A: ndarray. shape=(num_data_samps, num_ind_vars).
            Data matrix for independent variables.
        y: ndarray. shape=(num_data_samps, 1).
            Data column for dependent variable.

        Returns
        -----------
        c: ndarray. shape=(num_ind_vars+1, 1)
            Linear regression slope coefficients for each independent var AND the intercept term
        '''
        A = np.asarray(A)
        y = np.asarray(y)

        if y.ndim == 1:
            y = y.reshape(-1, 1)

        ones = np.ones((A.shape[0], 1))
        A_aug = np.hstack((A, ones))

        AtA = A_aug.T @ A_aug
        Aty = A_aug.T @ y
        c = np.linalg.solve(AtA, Aty)

        c = np.asarray(c)
        if c.ndim == 1:
            c = c.reshape(-1, 1)

        return c

    def linear_regression_qr(self, A, y):
        '''Performs a linear regression using the QR decomposition

        (Week 2)

        See notebook for a refresher on the equation

        Parameters:
        -----------
        A: ndarray. shape=(num_data_samps, num_ind_vars).
            Data matrix for independent variables.
        y: ndarray. shape=(num_data_samps, 1).
            Data column for dependent variable.

        Returns
        -----------
        c: ndarray. shape=(num_ind_vars+1, 1)
            Linear regression slope coefficients for each independent var AND the intercept term

        NOTE: You should not compute any matrix inverses! Check out scipy.linalg.solve_triangular
        to backsubsitute to solve for the regression coefficients `c`.
        '''
        A = np.asarray(A)
        y = np.asarray(y)

        if y.ndim == 1:
            y = y.reshape(-1, 1)

        ones = np.ones((A.shape[0], 1))
        A_aug = np.hstack((A, ones))

        Q, R = self.qr_decomposition(A_aug)
        c = scipy.linalg.solve_triangular(R, Q.T @ y)

        c = np.asarray(c)
        if c.ndim == 1:
            c = c.reshape(-1, 1)

        return c

    def qr_decomposition(self, A):
        '''Performs a QR decomposition on the matrix A. Make column vectors orthogonal relative
        to each other. Uses the Gram–Schmidt algorithm

        (Week 2)

        Parameters:
        -----------
        A: ndarray. shape=(num_data_samps, num_ind_vars+1).
            Data matrix for independent variables.

        Returns:
        -----------
        Q: ndarray. shape=(num_data_samps, num_ind_vars+1)
            Orthonormal matrix (columns are orthogonal unit vectors — i.e. length = 1)
        R: ndarray. shape=(num_ind_vars+1, num_ind_vars+1)
            Upper triangular matrix

        TODO:
        1. Q is found by the Gram–Schmidt orthogonalizing algorithm.
        Summary: Step thru columns of A left-to-right. You are making each newly visited column
        orthogonal to all the previous ones. You do this by projecting the current column onto each
        of the previous ones and subtracting each projection from the current column.
            - NOTE: Very important: Make sure that you make a COPY of your current column before
            subtracting (otherwise you might modify data in A!).
        2. Normalize each current column after orthogonalizing.
            - NOTE: When doing this, add a very small constant "fudge factor" of 1x10^-20 (i.e. 1e-20)
            to the denominator to prevent potential divisions by zero.
        3. R is found by equation summarized in notebook
        '''
        A = np.asarray(A, dtype=float)
        n_rows, n_cols = A.shape

        Q = np.zeros((n_rows, n_cols), dtype=float)

        for j in range(n_cols):
            v = A[:, j].copy()
            for i in range(j):
                proj = np.dot(Q[:, i], v) * Q[:, i]
                v = v - proj

            norm = np.linalg.norm(v)
            Q[:, j] = v / (norm + 1e-20)

        R = Q.T @ A
        return Q, R

    def predict(self, X=None):
        '''Use fitted linear regression model to predict the values of data matrix self.A.
        Generates the predictions y_pred = mA + b, where (m, b) are the model fit slope and intercept,
        A is the data matrix.

        Parameters:
        -----------
        X: ndarray. shape=(num_data_samps, num_ind_vars).
            If None, use self.A for the "x values" when making predictions.
            If not None, use X as independent var data as "x values" used in making predictions.

        Returns
        -----------
        y_pred: ndarray. shape=(num_data_samps, 1)
            Predicted y (dependent variable) values

        NOTE: You can write this method without any loops!
        '''
        if X is None:
            X_use = self.A
        else:
            X_use = X

        X_use = np.asarray(X_use)
        slope = np.asarray(self.slope)

        if X_use.ndim == 1:
            X_use = X_use.reshape(-1, 1)
        if slope.ndim == 1:
            slope = slope.reshape(-1, 1)

        if self.p > 1:
            X_use = self.make_polynomial_matrix(X_use, self.p)

        return X_use @ slope + self.intercept

    def r_squared(self, y_pred):
        '''Computes the R^2 quality of fit statistic

        Parameters:
        -----------
        y_pred: ndarray. shape=(num_data_samps,).
            Dependent variable values predicted by the linear regression model

        Returns:
        -----------
        R2: float.
            The R^2 statistic
        '''
        y_true = np.asarray(self.y)
        y_pred = np.asarray(y_pred)

        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)

        ss_res = np.sum((y_true - y_pred) ** 2)
        y_mean = np.mean(y_true)
        ss_tot = np.sum((y_true - y_mean) ** 2)

        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0

        return float(1 - (ss_res / ss_tot))

    def compute_residuals(self, y_pred):
        '''Determines the residual values from the linear regression model

        Parameters:
        -----------
        y_pred: ndarray. shape=(num_data_samps, 1).
            Data column for model predicted dependent variable values.

        Returns
        -----------
        residuals: ndarray. shape=(num_data_samps, 1)
            Difference between the y values and the ones predicted by the regression model at the
            data samples
        '''
        y_true = np.asarray(self.y)
        y_pred = np.asarray(y_pred)

        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)

        return y_true - y_pred

    def compute_mse(self):
        '''Computes the mean squared error in the predicted y compared the actual y values.
        See notebook for equation.

        Returns:
        -----------
        float. Mean squared error

        Hint: Make use of self.compute_residuals
        '''
        y_pred = self.predict()
        residuals = self.compute_residuals(y_pred)
        return float(np.mean(residuals ** 2))

    def scatter(self, ind_var, dep_var, title):
        '''Creates a scatter plot with a regression line to visualize the model fit.
        Assumes linear regression has been already run.

        Parameters:
        -----------
        ind_var: string. Independent variable name
        dep_var: string. Dependent variable name
        title: string. Title for the plot

        TODO:
        - Use your scatter() in Analysis to handle the plotting of points. Note that it returns
        the (x, y) coordinates of the points.
        - Sample evenly spaced x values for the regression line between the min and max x data values
        - Use your regression slope, intercept, and x sample points to solve for the y values on the
        regression line.
        - Plot the line on top of the scatterplot.
        - Make sure that your plot has a title (with R^2 value in it)
        '''
        x, y = analysis.Analysis.scatter(self, ind_var, dep_var, title)

        if self.ind_vars is None or self.dep_var is None or self.slope is None or self.intercept is None or self.R2 is None:
            raise ValueError('Run linear_regression before calling scatter.')

        if ind_var not in self.ind_vars:
            raise ValueError(f"{ind_var} is not one of the fitted independent variables: {self.ind_vars}")

        slope_idx = self.ind_vars.index(ind_var)
        b = float(self.intercept)

        x_flat = np.asarray(x).ravel()
        x_line = np.linspace(np.min(x_flat), np.max(x_flat), 200)

        if self.p > 1:
            A_line = self.make_polynomial_matrix(x_line.reshape(-1, 1), self.p)
            y_line = A_line @ np.asarray(self.slope).reshape(-1, 1) + b
            plt.plot(x_line, y_line.ravel(), color='red', linewidth=2)
        else:
            m = float(np.asarray(self.slope).reshape(-1, 1)[slope_idx, 0])
            y_line = m * x_line + b
            plt.plot(x_line, y_line, color='red', linewidth=2)
        plt.title(f'{title} (R^2 = {self.R2:.4f})')

        return x, y

    def pair_plot(self, data_vars, fig_sz=(12, 12), title='', hists_on_diag=True):
        '''Makes a pair plot with regression lines in each panel.
        There should be a len(data_vars) x len(data_vars) grid of plots, show all variable pairs
        on x and y axes.

        Parameters:
        -----------
        data_vars: Python list of strings. Variable names in self.data to include in the pair plot.
        fig_sz: tuple. len(fig_sz)=2. Width and height of the whole pair plot figure.
            This is useful to change if your pair plot looks enormous or tiny in your notebook!
        hists_on_diag: bool. If true, draw a histogram of the variable along main diagonal of
            pairplot.

        TODO:
        - Use your pair_plot() in Analysis to take care of making the grid of scatter plots.
        Note that this method returns the figure and axes array that you will need to superimpose
        the regression lines on each subplot panel.
        - In each subpanel, plot a regression line of the ind and dep variable. Follow the approach
        that you used for self.scatter. Note that here you will need to fit a new regression for
        every ind and dep variable pair.
        - Make sure that each plot has a title (with R^2 value in it)
        '''
        fig, axes = analysis.Analysis.pair_plot(self, data_vars, fig_sz=fig_sz, title=title)

        n_vars = len(data_vars)
        for row in range(n_vars):
            for col in range(n_vars):
                ax = axes[row, col] if n_vars > 1 else axes
                dep_var = data_vars[row]
                ind_var = data_vars[col]

                if hists_on_diag and row == col:
                    axes[row, col].remove()
                    axes[row, col] = fig.add_subplot(n_vars, n_vars, row * n_vars + col + 1)
                    if col < n_vars - 1:
                        axes[row, col].set_xticks([])
                    else:
                        axes[row, col].set_xlabel(data_vars[row])
                    if row > 0:
                        axes[row, col].set_yticks([])
                    else:
                        axes[row, col].set_ylabel(data_vars[row])

                    vals = self.data.select_data([ind_var]).ravel()
                    axes[row, col].hist(vals, bins=15)
                    axes[row, col].set_title(ind_var)
                    continue

                lr_panel = LinearRegression(self.data)
                lr_panel.linear_regression([ind_var], dep_var)

                if lr_panel.slope is None or lr_panel.intercept is None or lr_panel.R2 is None:
                    raise ValueError('Failed to fit regression in pair_plot panel.')

                x_vals = self.data.select_data([ind_var]).ravel()
                x_line = np.linspace(np.min(x_vals), np.max(x_vals), 200)
                y_line = float(lr_panel.slope[0, 0]) * x_line + float(lr_panel.intercept)
                ax.plot(x_line, y_line, color='red', linewidth=1.5)
                ax.set_title(f'R^2 = {lr_panel.R2:.3f}')

        return fig, axes

    def make_polynomial_matrix(self, A, p):
        '''Takes an independent variable data column vector `A and transforms it into a matrix appropriate
        for a polynomial regression model of degree `p`.

        (Week 2)

        Parameters:
        -----------
        A: ndarray. shape=(num_data_samps, 1)
            Independent variable data column vector x
        p: int. Degree of polynomial regression model.

        Returns:
        -----------
        ndarray. shape=(num_data_samps, p)
            Independent variable data transformed for polynomial model.
            Example: if p=10, then the model should have terms in your regression model for
            x^1, x^2, ..., x^9, x^10.

        NOTE: There should not be a intercept term ("x^0"), the linear regression solver method
        will take care of that.
        '''
        A = np.asarray(A)
        if A.ndim == 1:
            A = A.reshape(-1, 1)

        cols = [A ** power for power in range(1, p + 1)]
        return np.hstack(cols)

    def poly_regression(self, ind_var, dep_var, p, method='normal'):
        '''Perform polynomial regression — generalizes self.linear_regression to polynomial curves
        (Week 2)
        NOTE: For single linear regression only (one independent variable only)

        Parameters:
        -----------
        ind_var: str. Independent variable entered in the single regression.
            Variable names must match those used in the `self.data` object.
        dep_var: str. Dependent variable entered into the regression.
            Variable name must match one of those used in the `self.data` object.
        p: int. Degree of polynomial regression model.
             Example: if p=10, then the model should have terms in your regression model for
             x^1, x^2, ..., x^9, x^10, and a column of homogeneous coordinates (1st).
        method: str. Least squares solver method to use.
            Supported options: 'normal', 'scipy', 'qr' (to be added later)

        TODO:
        - This method should mirror the structure of self.linear_regression (compute all the same things)
        - Differences are:
            - You create a matrix based on the independent variable data matrix (self.A) with columns
            appropriate for polynomial regresssion. Do this with self.make_polynomial_matrix.
            - You set the instance variable for the polynomial regression degree (self.p)
        '''
        self.ind_vars = [ind_var]
        self.dep_var = dep_var
        self.p = p

        self.A = self.data.select_data(self.ind_vars)
        self.y = self.data.select_data([self.dep_var])

        A_poly = self.make_polynomial_matrix(self.A, p)

        if method == 'scipy':
            coeffs = self.linear_regression_scipy(A_poly, self.y)
        elif method == 'normal':
            coeffs = self.linear_regression_normal(A_poly, self.y)
        elif method == 'qr':
            coeffs = self.linear_regression_qr(A_poly, self.y)
        else:
            raise ValueError(f"Unknown regression method '{method}'. Expected one of: 'scipy', 'normal', 'qr'.")

        coeffs = np.asarray(coeffs)
        if coeffs.ndim == 1:
            coeffs = coeffs.reshape(-1, 1)

        self.slope = coeffs[:-1, :]
        self.intercept = float(coeffs[-1, 0])

        y_pred = self.predict()
        self.residuals = self.compute_residuals(y_pred)
        self.R2 = self.r_squared(y_pred)
        self.mse = self.compute_mse()

    def get_fitted_slope(self):
        '''Returns the fitted regression slope.
        (Week 2)

        Returns:
        -----------
        ndarray. shape=(num_ind_vars, 1). The fitted regression slope(s).
        '''
        return self.slope

    def get_fitted_intercept(self):
        '''Returns the fitted regression intercept.
        (Week 2)

        Returns:
        -----------
        float. The fitted regression intercept(s).
        '''
        return self.intercept

    def initialize(self, ind_vars, dep_var, slope, intercept, p):
        '''Sets fields based on parameter values.
        (Week 2)

        Parameters:
        -----------
        ind_vars: Python list of strings. 1+ independent variables (predictors) entered in the regression.
            Variable names must match those used in the `self.data` object.
        dep_var: str. Dependent variable entered into the regression.
            Variable name must match one of those used in the `self.data` object.
        slope: ndarray. shape=(num_ind_vars, 1)
            Slope coefficients for the linear regression fits for each independent var
        intercept: float.
            Intercept for the linear regression fit
        p: int. Degree of polynomial regression model.

        TODO:
        - Use parameters and call methods to set all instance variables defined in constructor.
        '''
        self.ind_vars = list(ind_vars)
        self.dep_var = dep_var
        self.slope = np.asarray(slope)
        self.intercept = float(intercept)
        self.p = p

        self.A = self.data.select_data(self.ind_vars)
        self.y = self.data.select_data([self.dep_var])

        y_pred = self.predict()
        self.residuals = self.compute_residuals(y_pred)
        self.R2 = self.r_squared(y_pred)
        self.mse = self.compute_mse()