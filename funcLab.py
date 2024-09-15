import numpy as np  # Import NumPy for handling numerical operations
import gspread  # Import gspread to access Google Sheets through the API
from google.auth import default  # Import default for Google authentication handling
import matplotlib.pyplot as plt  # Import matplotlib for creating plots
import sympy as sym  # Import SymPy for symbolic mathematical calculations
from scipy.optimize import curve_fit  # Import curve_fit for fitting curves to data
import math

# Authenticate the user and set up access to Google Sheets API
from google.colab import auth
auth.authenticate_user()  # Authenticate the user in the Google Colab environment
creds, _ = default()  # Retrieve the default credentials for authentication
gc = gspread.authorize(creds)  # Authorize the gspread client with the obtained credentials

# Set up symbolic mathematics pretty printing
sym.init_printing()


def importData(fileName, sheetName, cellRange, orientation='columns', data_type='numeric'):
    """
    Import data from a specified range in a Google Sheet using Excel-style range notation.

    Parameters:
    -----------
    fileName : str
        The name of the Google Sheet file.
    sheetName : str
        The specific worksheet within the Google Sheet.
    cellRange : str
        The range of cells to import, denoted in Excel-style notation (e.g., 'A1:A10', 'B2:F7', 'A1').
    orientation : str, optional
        Use 'columns' to import data as columns (default) or 'rows' to import data as rows.
    data_type : str, optional
        Use 'numeric' (default) to convert values to floats, 'string' to retain values as strings, or 'raw' for no conversion.

    Returns:
    --------
    list, list of lists, or single value
        Returns a list of data for each row/column within the specified range.
        Returns a flat list if the range is a single row or column.
        Returns a single value if the range is a single cell.
    """
    # Verify if the Google Sheets API client is set up
    if 'gc' not in globals():
        print("Google Sheets API client is not set up. Please run the authentication code.")
        return

    try:
        # Access the Google Sheet and retrieve data
        ss = gc.open(fileName)
        ws = ss.worksheet(sheetName)
        data = ws.get(cellRange)
    except Exception as e:
        print(f"Error retrieving range {cellRange}: {e}")
        return None

    # Function to convert data based on the specified type
    def convert_data(cell):
        if data_type == 'numeric':
            return safe_eval(cell)
        elif data_type == 'string':
            return str(cell)
        else:
            return cell  # Return the raw value if 'raw' is specified

    # Handle the case for a single cell
    if len(data) == 1 and len(data[0]) == 1:
        data_value = data[0][0]
        return convert_data(data_value)

    # Handle the case for a single row
    if len(data) == 1:
        return [convert_data(cell) for cell in data[0]]  # Return the single row as a flat list with conversion

    # Handle the case for a single column
    if all(len(row) == 1 for row in data):
        return [convert_data(row[0]) for row in data]  # Return the single column as a flat list with conversion

    # Transpose data if importing columns and orientation is 'columns'
    if orientation == 'columns':
        data = np.transpose(data)

    # Convert data to numeric or keep as strings for multi-row/multi-column scenarios
    if data_type == 'numeric':
        data = [[safe_eval(cell) for cell in row] for row in data]
    elif data_type == 'string':
        data = [[str(cell) for cell in row] for row in data]

    return data


def safe_eval(value):
    """
    Convert a string to a float, handling cases where the string is not a numeric value.

    Parameters:
    -----------
    value : str
        The string to be converted.

    Returns:
    --------
    float
        The converted float value, or NaN if conversion is not possible.
    """
    try:
        return float(value.replace(',', '.'))  # Replace commas with dots to standardize decimal notation
    except ValueError:
        return float('nan')  # Return NaN for non-numeric values

def curveFit(func, x, y):
    """
    Fit a curve to the provided data using the given function and calculate the fitting statistics.

    Parameters:
    -----------
    func : callable
        The function to use for fitting the data. It should take x values and parameters, returning y values.
    x : array-like
        Data for the independent variable.
    y : array-like
        Data for the dependent variable.

    Returns:
    --------
    dict
        A dictionary with:
        - 'parameters': Optimized parameters of the fit
        - 'parameter_uncertainties': Uncertainties of the parameters
        - 'r_squared': R-squared statistic of the fit
    """
    X = np.asarray(x)  # Convert input data to numpy arrays
    Y = np.asarray(y)

    try:
        # Perform the curve fitting
        popt, pcov = curve_fit(func, X, Y)
    except Exception as e:
        print(f"Error during curve fitting: {e}")
        return {}

    # Compute the uncertainties of the parameters
    perr = np.sqrt(np.diag(pcov))

    # Calculate fitted values
    fit_y = func(X, *popt)
    
    # Determine the residuals
    residuals = Y - fit_y
    
    # Calculate the sum of squared residuals
    ss_res = np.sum(residuals**2)
    
    # Calculate the total sum of squares
    ss_tot = np.sum((Y - np.mean(Y))**2)
    
    # Compute the R-squared statistic
    r_squared = 1 - (ss_res / ss_tot)
    
    # Display the fitting results
    print("Fitting results:")
    for i in range(len(popt)):
        print(f'Parameter a_{i} = {popt[i]} ± {perr[i]}')
    print(f'R² = {r_squared:.4f}')

    # Return the fitting results
    return {
        'parameters': popt,
        'parameter_uncertainties': perr,
        'r_squared': r_squared
    }


class Variable:
    def __init__(self, sim, val, inc):
        """
        Create a variable with symbolic representation, value, and uncertainty.

        Parameters:
        -----------
        sym : str
            The symbolic name of the variable.
        val : float
            The value of the variable.
        inc : float
            The uncertainty associated with the variable.
        """
        self.sim = sym.Symbol(sim)  # Create a symbolic variable
        self.val = val  # The variable's value
        self.inc = inc  # The variable's uncertainty

def regression(x, y, table=False):
    """
    Conduct linear regression on the data and present the results.

    Parameters:
    -----------
    x : array-like
        Data for the independent variable.
    y : array-like
        Data for the dependent variable.
    table : bool, optional
        If True, display the regression results in LaTeX table format.
        If False, display the results in plain text.

    Returns:
    --------
    dict
        A dictionary with:
        - 'slope': Slope of the regression line (represented as a Variable object)
        - 'intercept': Intercept of the regression line (represented as a Variable object)
        - 'r_squared': R-squared statistic
    """
    def func(x, A, B):
        return A * x + B  # Linear function for fitting

    curve = curveFit(func, x, y)  # Fit the curve using a linear function
    
    # Create Variable objects for slope and intercept
    A = Variable('A', curve['parameters'][0], curve['parameter_uncertainties'][0])
    B = Variable('B', curve['parameters'][1], curve['parameter_uncertainties'][1])
    R2 = curve['r_squared']  # R-squared statistic

    results = {
        'slope': A,
        'intercept': B,
        'r_squared': R2
    }

    if table:
        # Display results in LaTeX table format
        print('\\begin{tabular}{ccc}')
        print('$A$ & $B$ & $r^2$ \\\\ \hline')
        print(f'${A.val:.4f} \pm {A.inc:.4f}$ & ${B.val:.4f} \pm {B.inc:.4f}$ & {R2:.4f} \\\\ \hline')
        print('\\end{tabular}')
    else:
        # Display regression coefficients and R-squared value
        print(f'A = {A.val:.4f} ± {A.inc:.4f}')
        print(f'B = {B.val:.4f} ± {B.inc:.4f}')
        print(f'R² = {R2:.4f}')

    return results

def propUncertainty(fun, variables):
    """
    Propagate uncertainties through a symbolic function using partial derivatives.

    Parameters:
    -----------
    fun : sympy expression
        The symbolic function for uncertainty propagation.
    variables : list of Variable or Variable
        A list (or single instance) of Variable objects representing the variables with their uncertainties.

    Returns:
    --------
    tuple
        A tuple containing:
        - List of evaluated function values.
        - List of uncertainties associated with the function values.
    """
    # Ensure that 'variables' is a list
    if not isinstance(variables, list):
        variables = [variables]

    # Initialize symbolic expression for uncertainty propagation
    errfun = 0
    for var in variables:
        sigma_s = sym.Symbol('sigma_' + var.sim.name)  # Symbol for the uncertainty of the variable
        derivative = sym.diff(fun, var.sim)  # Compute the partial derivative of the function with respect to the variable
        errfun += (derivative * sigma_s) ** 2  # Sum the squared term to the total uncertainty expression

    errfun = sym.sqrt(errfun)  # Take the square root to get the combined uncertainty expression

    # Display the symbolic uncertainty expression in LaTeX format
    print("Symbolic uncertainty expression:")
    print(sym.latex(errfun))

    def substitute_values_and_uncertainties(val, inc):
        """
        Substitute values and uncertainties into the symbolic expressions and evaluate them.

        Returns:
        --------
        tuple
            The evaluated function value and its associated uncertainty.
        """
        evaluated_fun = fun
        evaluated_errfun = errfun

        for var in variables:
            evaluated_fun = evaluated_fun.subs(var.sim, val)
            evaluated_errfun = evaluated_errfun.subs(var.sim, val)
            
            # Substitute the uncertainty into the uncertainty expression
            evaluated_errfun = evaluated_errfun.subs(sym.Symbol('sigma_' + var.sim.name), inc)

        return evaluated_fun, evaluated_errfun

    valors = []
    incerteses = []

    for var in variables:
        # Ensure the lengths of values and uncertainties match
        if isinstance(var.val, (list, np.ndarray)):
            if isinstance(var.inc, (list, np.ndarray)):
                if len(var.val) != len(var.inc):
                    raise ValueError("Values and uncertainties lists/arrays must have the same length.")
                
                for value, uncertainty in zip(var.val, var.inc):
                    evaluated_fun, evaluated_errfun = substitute_values_and_uncertainties(value, uncertainty)
                    valors.append(evaluated_fun.evalf())  # Evaluate the function value
                    incerteses.append(evaluated_errfun.evalf())  # Evaluate the uncertainty
            else:
                for value in var.val:
                    evaluated_fun, evaluated_errfun = substitute_values_and_uncertainties(value, var.inc)
                    valors.append(evaluated_fun.evalf())  # Evaluate the function value
                    incerteses.append(evaluated_errfun.evalf())  # Evaluate the uncertainty
        else:
            if isinstance(var.inc, (list, np.ndarray)):
                for uncertainty in var.inc:
                    evaluated_errfun = errfun.subs(sym.Symbol('sigma_' + var.sim.name), uncertainty)
                    incerteses.append(evaluated_errfun.evalf())  # Evaluate the uncertainty
            else:
                evaluated_fun, evaluated_errfun = substitute_values_and_uncertainties(var.val, var.inc)
                valors.append(evaluated_fun.evalf())  # Evaluate the function value
                incerteses.append(evaluated_errfun.evalf())  # Evaluate the uncertainty

    return valors, incerteses


def mean(values, instrumental_error):
    """
    Compute the mean and combined uncertainty of a set of measurements.

    Parameters:
    -----------
    values : list of float
        The set of measured values.
    instrumental_error : float
        The uncertainty associated with the measurement process.

    Returns:
    --------
    tuple
        A tuple with:
        - Mean of the values.
        - Combined uncertainty (statistical error combined with instrumental error).
    """
    if len(values) == 0:
        raise ValueError("The list of values is empty. Mean and uncertainty cannot be calculated.")
    
    if instrumental_error < 0:
        raise ValueError("Instrumental error must be non-negative.")
    
    values = np.array(values)  # Convert the list of values to a NumPy array
    mean = np.mean(values)  # Compute the mean of the values
    std_dev = np.std(values, ddof=1)  # Calculate the standard deviation (sample standard deviation)
    statistical_error = std_dev / np.sqrt(len(values))  # Compute the statistical error
    combined_uncertainty = np.sqrt(statistical_error**2 + instrumental_error**2)  # Combine statistical and instrumental uncertainties
    
    # Display the mean and combined uncertainty
    print(f"Mean: ${mean:.2f} \pm {combined_uncertainty:.2f}$")
    
    return mean, combined_uncertainty


def plotData(ax, x, y, label=None, color='k', marker='o', markersize=3):
    """
    Plot data with error bars on the specified axis.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axis object from Matplotlib where the data will be plotted.
    x : Variable
        Data for the independent variable, including uncertainties.
    y : Variable
        Data for the dependent variable, including uncertainties.
    label : str, optional
        Label for the data series (default is None).
    color : str, optional
        Color for the data points and error bars (default is 'k' for black).
    marker : str, optional
        Style of the markers (default is 'o' for circles).
    markersize : int, optional
        Size of the markers (default is 3).

    Raises:
    -------
    TypeError
        If x or y are not instances of the Variable class.
    """
    if not isinstance(x, Variable) or not isinstance(y, Variable):
        raise TypeError("x and y must be instances of the Variable class.")
    
    ax.tick_params(direction='in', right=True, top=True)  # Adjust tick parameters
    ax.grid(color='#eeeeee', linestyle='--')  # Add grid with specified color and style

    error_params = {
        'xerr': x.inc,  # Error bars for x data
        'yerr': y.inc,  # Error bars for y data
        'capsize': 0,  # No cap size for error bars
        'elinewidth': 0.5,  # Line width for error bars
        'linewidth': 0,  # No line width for data points
        'marker': marker,  # Marker style
        'markersize': markersize,  # Size of the markers
        'markerfacecolor': color,  # Fill color of markers
        'markeredgecolor': color,  # Edge color of markers
        'ecolor': color  # Color of the error bars
    }
    
    if label:
        error_params['label'] = label  # Add label to the legend if provided

    ax.errorbar(x.val, y.val, **error_params)  # Plot data with error bars
    
    if label:
        ax.legend()  # Display legend if a label was provided


def plotFit(ax, x, y, fit_func, label='Curve Fit', color='k', xrange=None):
    """
    Plot data with error bars on the given axis and optionally fit a curve to the data.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axis object from Matplotlib where the data and fit will be plotted.
    x : Variable
        Data for the independent variable, including uncertainties.
    y : Variable
        Data for the dependent variable, including uncertainties.
    fit_func : callable, optional
        The function to fit to the data (default is None). Should take x values and parameters and return y values.
    label : str, optional
        Label for the fit curve (default is 'Curve Fit').
    color : str, optional
        Color for the data points and fit curve (default is 'k' for black).
    xrange : tuple, optional
        The range for the x-axis over which to plot the fit (default is None).

    Raises:
    -------
    TypeError
        If x or y are not instances of the Variable class.
    """
    if not isinstance(x, Variable) or not isinstance(y, Variable):
        raise TypeError("x and y must be instances of the Variable class.")
    
    ax.tick_params(direction='in', right=True, top=True)  # Adjust tick parameters
    ax.grid(color='#eeeeee', linestyle='--')  # Add grid with specified color and style

    # Convert Variable instances to numpy arrays for curve fitting
    X = np.asarray(x.val)
    Y = np.asarray(y.val)
        
    # Perform curve fitting
    fit_results = curveFit(fit_func, X, Y)
        
    if fit_results:
        # Extract the parameters from the fit results
        popt = fit_results['parameters']
       
        if xrange is not None:
            X = np.arange(xrange[0], xrange[1], (xrange[1] - xrange[0]) / 1000)

        # Compute the fitted y values
        fit_y = fit_func(X, *popt)

        # Plot the fitted curve
        ax.plot(X, fit_y, color=color, linestyle='-', label=label)  # Plot the fit curve
        ax.legend()

        return fit_results

class Table:
    def __init__(self):
        self.data = []
        self.history = []
        self.redo_stack = []

    def save_state(self):
        """Save the current state of the table for undo."""
        self.history.append([row[:] for row in self.data])
        self.redo_stack = []  # Clear redo stack on new change

    def undo(self):
        """Undo the last action."""
        if not self.history:
            print("No actions to undo.")
            return
        self.redo_stack.append([row[:] for row in self.data])
        self.data = self.history.pop()

    def redo(self):
        """Redo the last undone action."""
        if not self.redo_stack:
            print("No actions to redo.")
            return
        self.history.append([row[:] for row in self.data])
        self.data = self.redo_stack.pop()

    def print_data(self):
        """Print the current table."""
        if not self.data:
            print("Table is empty.")
            return
        for i in self.data:
            print(i)

    def add_row(self, row_values, index=None):
        """Add a row at the specified index (default: at the end)."""
        self.save_state()  
        if index is None:
            index = len(self.data)  # Insert at the end if no index is given
        if index < 0 or index > len(self.data):
            print("Error: Index out of bounds.")
            return
        if self.data and len(row_values) != len(self.data[0]):
            print("Error: Row length does not match the number of columns.")
            return
        string_list = [str(element) for element in row_values]
        self.data.insert(index, string_list)

    def delete_row(self, index):
        """Delete a row by index."""
        self.save_state()  
        if index < 0 or index >= len(self.data):
            print("Error: Row index out of bounds.")
            return
        self.data.pop(index)

    def add_column(self, col_values, index=None):
        """Add a column at the specified index (default: at the end)."""
        self.save_state()  
        if not self.data:
            print("Error: Table is empty, add rows first.")
            return
        if len(col_values) != len(self.data):
            print("Error: Column length does not match the number of rows.")
            return
        if index is None:
            index = len(self.data[0])  # Insert at the end if no index is given
        for i in range(len(col_values)):
            self.data[i].insert(index, str(col_values[i]))

    def delete_column(self, index):
        """Delete a column by index."""
        self.save_state()  
        if not self.data:
            print("Error: Table is empty.")
            return
        if index < 0 or index >= len(self.data[0]):
            print("Error: Column index out of bounds.")
            return
        for i in range(len(self.data)):
            self.data[i].pop(index)

    def change_value(self, row, col, new_value):
        """Change a specific value in the table."""
        self.save_state()  
        if row < 0 or row >= len(self.data):
            print("Error: Row index out of bounds.")
            return
        if col < 0 or col >= len(self.data[0]):
            print("Error: Column index out of bounds.")
            return
        self.data[row][col] = str(new_value)

    def add_uncertainties(self, uncertainties, axis='column', index=0):
        """
        Add uncertainties (± values) to either a specific row or column.
        :param uncertainties: A list of uncertainties to add.
        :param axis: 'column' or 'row' to specify where uncertainties should be added.
        :param index: The index of the row/column where uncertainties will be added.
        """
        self.save_state()  
        if axis == 'column':
            if len(uncertainties) != len(self.data):
                print("Error: Uncertainty length must match the number of rows.")
                return
            for i in range(len(self.data)):
                self.data[i][index] += f' $\\pm$ {uncertainties[i]}'
        elif axis == 'row':
            if len(uncertainties) != len(self.data[0]):
                print("Error: Uncertainty length must match the number of columns.")
                return
            for i in range(len(self.data[index])):
                self.data[index][i] += f' $\\pm$ {uncertainties[i]}'
        else:
            print("Error: axis must be 'column' or 'row'.")

    def import_from_range(self, fileName, sheetName, cellRange):
        """Import data from a given range."""
        self.save_state()  
        try:
             # Verify if the Google Sheets API client is set up
            if 'gc' not in globals():
                print("Google Sheets API client is not set up. Please run the authentication code.")
                return
            try:
                # Access the Google Sheet and retrieve data
                ss = gc.open(fileName)
                ws = ss.worksheet(sheetName)
                data = ws.get(cellRange)
            except Exception as e:
                print(f"Error retrieving range {cellRange}: {e}")
                return None

            # Handle the case for a single cell
            if len(data) == 1 and len(data[0]) == 1:
                data_value = data[0][0]
                return data_value

            # Handle the case for a single row
            if len(data) == 1:
                return [cell for cell in data[0]]  # Return the single row as a flat list with conversion

            # Handle the case for a single column
            if all(len(row) == 1 for row in data):
                return [row[0] for row in data]  # Return the single column as a flat list with conversion

            data = [[str(cell) for cell in row] for row in data]

            self.data = data

        except Exception as e:
            print(f"Error importing data: {e}")
    
    def transpose(self):
        """Transpose the table, converting rows to columns and vice versa."""
        self.save_state()  
        if not self.data:
            print("Error: Table is empty.")
            return
        self.data = list(map(list, zip(*self.data)))


    def latex_table(self, caption='', label='', hlines="all", vlines="none"):
        """Generate LaTeX code for the table."""
        if not self.data:
            print("Error: Table is empty.")
            return
        
        #handle vertical lines
        if vlines == "all":
            cs = '|c' * (len(self.data[0])-1) + '|c|'  
        elif vlines == "none":
            cs = 'c' * len(self.data[0])  
        else:
            cs = ''
            for i in range(len(self.data[0])):
                if vlines[i] == '1':
                    cs += '|c'
                elif vlines[i] == '0':
                    cs += 'c'
            if vlines[len(self.data[0])] == '1':
                cs += '|'

        print('\\begin{table}[h!]')
        print('    \\centering')
        print(f'    \\caption{{{caption}}}')
        print(f'    \\label{{tab:{label}}}')
        print(f'    \\begin{{tabular}}{{{cs}}}')
        if hlines == "all": print('\\hline')
        elif hlines[0] == '1': print('\\hline')

        for row in self.data:
            if hlines == "all": print('        ' + ' & '.join(row) + ' \\\\ \\hline')
            elif hlines == "none": print('        ' + ' & '.join(row) + ' \\\\')
            elif hlines[self.data.index(row) + 1] == '1': print('        ' + ' & '.join(row) + ' \\\\ \\hline')
            else: print('        ' + ' & '.join(row) + ' \\\\')
        print('    \\end{tabular}')
        print('\\end{table}')
