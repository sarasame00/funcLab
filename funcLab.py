import numpy as np  # Import NumPy for numerical operations
import gspread  # Import gspread for Google Sheets API access
from google.auth import default  # Import default for Google authentication
import matplotlib.pyplot as plt  # Import matplotlib for plotting
import sympy as sym  # Import SymPy for symbolic mathematics
from scipy.optimize import curve_fit  # Import curve_fit for curve fitting

# Authenticate and authorize Google Sheets API client
from google.colab import auth
auth.authenticate_user()  # Authenticate the user for Google Colab environment
creds, _ = default()  # Get default authentication credentials
gc = gspread.authorize(creds)  # Authorize the gspread client with credentials

# Initialize pretty printing for symbolic outputs
sym.init_printing()

def importData(fileName, sheetName, numCol, firstRow=1, lastRow=None):
    """
    Import data from a specified Google Sheet.

    Parameters:
    -----------
    fileName : str
        The name of the Google Sheet file.
    sheetName : str
        The name of the worksheet within the Google Sheet.
    numCol : int or list of int
        The column number(s) to import (1-based indexing).
    firstRow : int, optional
        The number of the first row to import (default is 1).
    lastRow : int, optional
        The number of the last row to import (default is None, meaning all rows).

    Returns:
    --------
    list or list of lists
        A list of data from the specified columns. If multiple columns are specified, a list of lists is returned.
    """
    # Check if Google Sheets API client is initialized
    if 'gc' not in globals():
        print("Google Sheets API client is not initialized. Run authentication code.")
        return

    try:
        ss = gc.open(fileName)  # Open the Google Sheet by name
        ws = ss.worksheet(sheetName)  # Access the specified worksheet
    except Exception as e:
        print(f"Error opening file or sheet: {e}")
        return

    if isinstance(numCol, int):
        numCol = [numCol]  # Ensure numCol is a list for uniform processing

    dataColMod = []  # List to store cleaned data for all columns

    for nc in numCol:
        try:
            dataCol = ws.col_values(nc)  # Retrieve all values from the specified column
            dataCol = dataCol[firstRow-1:lastRow] if lastRow else dataCol[firstRow-1:]  # Slice data according to firstRow and lastRow
            dataColMod.append([safe_eval(d) for d in dataCol])  # Convert data to numerical format
        except Exception as e:
            print(f"Error processing column {nc}: {e}")
            return

    return dataColMod[0] if len(dataColMod) == 1 else dataColMod  # Return single list or list of lists

def safe_eval(value):
    """
    Safely convert a string to a float, handling non-numeric values.

    Parameters:
    -----------
    value : str
        The string to convert.

    Returns:
    --------
    float
        The converted float value, or NaN if conversion fails.
    """
    try:
        return float(value.replace(',', '.'))  # Replace commas with dots for decimal conversion
    except ValueError:
        return float('nan')  # Handle non-numeric data as NaN

def curveFit(func, x, y):
    """
    Fit a curve to the data using the specified function and calculate fit statistics.

    Parameters:
    -----------
    func : callable
        The function to fit the data to. It should accept x values and parameters, and return y values.
    x : array-like
        The independent variable data.
    y : array-like
        The dependent variable data.

    Returns:
    --------
    dict
        A dictionary containing:
        - 'parameters': Optimized parameters
        - 'parameter_uncertainties': Parameter uncertainties
        - 'r_squared': R-squared value
    """
    X = np.asarray(x)  # Convert input data to numpy arrays
    Y = np.asarray(y)

    try:
        popt, pcov = curve_fit(func, X, Y)  # Fit the curve
    except Exception as e:
        print(f"Error during curve fitting: {e}")
        return {}

    perr = np.sqrt(np.diag(pcov))  # Compute standard uncertainties of the parameters
    residuals = Y - func(X, *popt)  # Calculate residuals
    ss_res = np.sum(residuals**2)  # Sum of squared residuals
    ss_tot = np.sum((Y - np.mean(Y))**2)  # Total sum of squares
    r_squared = 1 - (ss_res / ss_tot)  # R-squared value

    results = {
        'parameters': popt,
        'parameter_uncertainties': perr,
        'r_squared': r_squared
    }

    # Print fitting results
    print("Fitting results:")
    for i, param in enumerate(popt):
        print(f'Parameter a_{i} = {param:.4f} ± {perr[i]:.4f}')
    print(f'R² = {r_squared:.4f}')

    return results

class Variable:
    def __init__(self, sym, val, inc):
        """
        Initialize a variable with a symbolic representation, value, and uncertainty.

        Parameters:
        -----------
        sym : str
            The name of the symbolic variable.
        val : float
            The value of the variable.
        inc : float
            The uncertainty of the variable.
        """
        self.sym = sym.Symbol(sym)  # Initialize symbolic variable
        self.val = val  # Value of the variable
        self.inc = inc  # Uncertainty of the variable

def regression(x, y, table=False):
    """
    Perform linear regression on the data and return or print the results.

    Parameters:
    -----------
    x : array-like
        The independent variable data.
    y : array-like
        The dependent variable data.
    table : bool, optional
        If True, print the regression results in LaTeX table format.
        If False, print the results in plain text.

    Returns:
    --------
    dict
        A dictionary containing:
        - 'slope': Slope of the regression line (as a Variable object)
        - 'intercept': Intercept of the regression line (as a Variable object)
        - 'r_squared': R-squared value
    """
    def func(x, A, B):
        return A * x + B  # Linear function for fitting

    curve = curveFit(func, x, y)  # Fit the curve using the linear function
    
    # Create Variable objects for the slope and intercept
    A = Variable('A', curve['parameters'][0], curve['parameter_uncertainties'][0])
    B = Variable('B', curve['parameters'][1], curve['parameter_uncertainties'][1])
    R2 = curve['r_squared']  # R-squared value

    results = {
        'slope': A,
        'intercept': B,
        'r_squared': R2
    }

    if table:
        # Print results in LaTeX table format
        print('\\begin{tabular}{ccc}')
        print('$A$ & $B$ & $r^2$ \\\\ \hline')
        print(f'${A.val:.4f} \pm {A.inc:.4f}$ & ${B.val:.4f} \pm {B.inc:.4f}$ & {R2:.4f} \\\\ \hline')
        print('\\end{tabular}')
    else:
        # Print regression coefficients and R-squared value
        print(f'A = {A.val:.4f} ± {A.inc:.4f}')
        print(f'B = {B.val:.4f} ± {B.inc:.4f}')
        print(f'R² = {R2:.4f}')

    return results

def propIncertesa(fun, variables):
    """
    Propagate uncertainties through a symbolic function using partial derivatives.

    Parameters:
    -----------
    fun : sympy expression
        The symbolic function through which uncertainties are propagated.
    variables : list of Variable or Variable
        A list of Variable objects (or a single Variable) representing variables with their uncertainties.

    Returns:
    --------
    tuple
        A tuple containing:
        - List of evaluated function values.
        - List of uncertainties associated with the function values.
    """
    # Ensure variables is a list
    if not isinstance(variables, list):
        variables = [variables]

    errfun = 0  # Initialize symbolic expression for propagated uncertainty
    for var in variables:
        sigma_s = sym.Symbol('sigma_' + var.sym.name)  # Symbol for the uncertainty of the variable
        derivative = sym.diff(fun, var.sym)  # Compute the partial derivative of the function with respect to the variable
        errfun += (derivative * sigma_s) ** 2  # Add the squared term to the total uncertainty expression

    errfun = sym.sqrt(errfun)  # Take the square root to get the combined uncertainty expression

    # Print the symbolic uncertainty expression in LaTeX format
    print("Symbolic uncertainty expression:")
    print(sym.latex(errfun))

    def substitute_values_and_uncertainties():
        """
        Substitute values and uncertainties into the symbolic expressions and evaluate them.

        Returns:
        --------
        tuple
            Evaluated function value and uncertainty.
        """
        evaluated_fun = fun
        evaluated_errfun = errfun

        for var in variables:
            if isinstance(var.val, (float, int, sym.core.numbers.Float)):
                # Substitute the variable value into the function and uncertainty expressions
                evaluated_fun = evaluated_fun.subs(var.sym, var.val)
                evaluated_errfun = evaluated_errfun.subs(var.sym, var.val)
            
            if isinstance(var.inc, (float, int, sym.core.numbers.Float)):
                # Substitute the uncertainty into the uncertainty expression
                evaluated_errfun = evaluated_errfun.subs(sym.Symbol('sigma_' + var.sym.name), var.inc)

        return evaluated_fun, evaluated_errfun

    valors = []
    incerteses = []

    for var in variables:
        if isinstance(var.val, (list, np.ndarray)):
            # If the variable value is a list or array, compute the function value and uncertainty for each entry
            for value in var.val:
                evaluated_fun, evaluated_errfun = substitute_values_and_uncertainties()
                valors.append(evaluated_fun.subs(var.sym, value).evalf())  # Evaluate function value
                incerteses.append(evaluated_errfun.subs(var.sym, value).evalf())  # Evaluate uncertainty
        else:
            if isinstance(var.inc, (list, np.ndarray)):
                # If the uncertainty is a list or array, compute the uncertainty for each entry
                for uncertainty in var.inc:
                    evaluated_errfun = errfun.subs(sym.Symbol('sigma_' + var.sym.name), uncertainty)
                    incerteses.append(evaluated_errfun.evalf())  # Evaluate uncertainty
            else:
                evaluated_fun, evaluated_errfun = substitute_values_and_uncertainties()
                valors.append(evaluated_fun.evalf())  # Evaluate function value
                incerteses.append(evaluated_errfun.evalf())  # Evaluate uncertainty

    return valors, incerteses

def mean_and_uncertainty(values, instrumental_error):
    """
    Calculate the mean and combined uncertainty of a set of values.

    Parameters:
    -----------
    values : list of float
        The list of measured values.
    instrumental_error : float
        The instrumental error, which is the uncertainty associated with the measurement process.

    Returns:
    --------
    tuple
        A tuple containing:
        - Mean of the values.
        - Combined uncertainty (statistical error combined with instrumental error).
    """
    if len(values) == 0:
        raise ValueError("The input list of values is empty. Cannot calculate mean and uncertainty.")
    
    if instrumental_error < 0:
        raise ValueError("Instrumental error must be non-negative.")
    
    values = np.array(values)  # Convert values to a NumPy array for numerical operations
    mean = np.mean(values)  # Calculate the mean of the values
    std_dev = np.std(values, ddof=1)  # Compute the standard deviation (sample standard deviation)
    statistical_error = std_dev / np.sqrt(len(values))  # Calculate the statistical error
    combined_uncertainty = np.sqrt(statistical_error**2 + instrumental_error**2)  # Combine statistical and instrumental uncertainties
    
    # Print the mean and combined uncertainty
    print(f"Mean: ${mean:.2f} \pm {combined_uncertainty:.2f}$")
    
    return mean, combined_uncertainty

def plotDades(ax, x, y, label=None, color='b', marker='o', markersize=3):
    """
    Plot data with error bars on the given axis.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The Matplotlib axis object to plot on.
    x : Variable
        The independent variable data, including uncertainties.
    y : Variable
        The dependent variable data, including uncertainties.
    label : str, optional
        Label for the data series (default is None).
    color : str, optional
        Color of the data points and error bars (default is 'b' for blue).
    marker : str, optional
        Marker style for the data points (default is 'o' for circles).
    markersize : int, optional
        Size of the markers (default is 3).

    Raises:
    -------
    TypeError
        If either x or y is not an instance of the Variable class.
    """
    if not isinstance(x, Variable) or not isinstance(y, Variable):
        raise TypeError("Both x and y must be instances of the Variable class.")
    
    ax.tick_params(direction='in', right=True, top=True)  # Customize tick parameters
    ax.grid(color='#eeeeee', linestyle='--')  # Add grid with specified style

    error_params = {
        'xerr': x.inc,  # Error bars for x data
        'yerr': y.inc,  # Error bars for y data
        'capsize': 0,  # No cap size for error bars
        'elinewidth': 0.5,  # Line width for error bars
        'linewidth': 0,  # No line width for data points
        'marker': marker,  # Marker style for data points
        'markersize': markersize,  # Size of markers
        'markerfacecolor': color,  # Fill color of markers
        'markeredgecolor': color,  # Edge color of markers
        'ecolor': color  # Color of error bars
    }
    
    if label:
        error_params['label'] = label  # Add label to legend if provided

    ax.errorbar(x.val, y.val, **error_params)  # Plot data with error bars
    
    if label:
        ax.legend()  # Show legend if a label was provided

