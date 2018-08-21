import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np

from functools import partial

from .base_plot import BasePlot

class CurvePlot(BasePlot):
    """
    Compare different curves in one plot. In machine learning research, it is 
    extremely useful, e.g. compare training losses with different baselines. 
    
    Note that the uncertainty (error bands) is supported, a standard use case
    is that each baseline run several times by using different random seeds. 
    
    Either with or without uncertainty, it depends on what kind of data added
    to the plotter via `add(name, data)`. If the data is one-dimentional, it will
    be treated as a single curve. If the data is two-dimensional, it will be plotted
    with uncertainty. 
    
    To generate a modern high quality research plot, we use Seaborn.lineplot with 
    Pandas.DataFrame data structure. 
    
    For more advanced use cases, feel free to inherit this class and overide `__call__`. 
    """
    def add(self, name, data, xvalues=None):
        """
        Add a curve data (either one of multiple) with option to select range of values
        for horizontal axis. If not provided, then it will be automatically set to integers. 
        
        Note that only one list of xvalues needed, because all curve data should share identical
        horizontal axis. If a batch of xvalues are provided and they are not identical, 
        then each line will be interpolated and new shared xvalues and queried y values will be computed.
        
        Args:
            name (str): name of the curve
            data (list/ndarray): curve data. If multiple curves (list) provided, then it will plot uncertainty bands. 
            xvalues (list, optional): values for horizontal axis. If None, then it set to be integers. 
        """
        # Convert data to ndarray
        data = np.array(data)
        # Get number of data points
        N = data.shape[-1]  # handle both single/multiple curve data, since last dimension always be data size
        # Set xvalues
        if xvalues is None:
            xvalues = np.arange(1, N+1)
        else:
            xvalues = np.array(xvalues)
        
        # Sanity check if all xvalues are identical, otherwise the uncertainty bands are impossible to plot
        # We check if first element is not scallar to determine batched xvalues, because we should allow
        # x values with different length for each line, then it's impossible to have multidimensional array
        if not np.isscalar(xvalues[0]):  # independent x values for each line, so check it !
            check_pass = np.all([np.array_equal(xvalues[0], x) for x in xvalues[1:]])
            # Interpolate the lines to share same x values if check is failed to pass
            if not check_pass:
                # Get new shared xvalues and queried y values from interpolated lines
                xvalues, data = self._interp_data(all_x=xvalues, all_y=data, num_points=500)
            else:  # passed the check, batch with same xvalues, but we want one xvalues only
                xvalues = xvalues[0]  # take first one, because rest of them are identical
        
        # Make data
        D = {'data': data, 'xvalues': xvalues}
        
        # Call parent add to save data
        super().add(name=name, data=D)
    
    def __call__(self, 
                 colors=None, 
                 scales=None, 
                 alphas=None, 
                 **kwargs):
        """
        Args:
            colors (list): A list of colors, each for plotting one data item
            scales (list): A list of scales of standard deviation, each for plotting one uncertainty band
            alphas (list): A list of alphas (transparency), each for plotting one uncertainty band
            **kwargs: keyword aguments used to specify the plotting options. 
                 A list of possible options:
                     - ax (Axes): given Matplotlib Axes to plot the figure
                     - title (str): title of the plot
                     - xlabel (str): label of horizontal axis
                     - ylabel (str): label of vertical axis
                     - xlim (tuple): [min, max] as limit of horizontal axis
                     - ylim (tuple): [min, max] as limit of veritical axis
                     - logx (bool): log-scale of horizontal axis
                     - logy (bool): log-scale of vertical axis
                     - integer_x (bool): enforce integer coordinates of horizontal axis
                     - integer_y (bool): enforce integer coordinates of vertical axis
                     - legend_loc (str): location string of the legend.
                     - num_tick (int): Maximum number of major ticks in horizontal axis. 
                     - xscale_magnitude (str): Format the major ticks in horizontal axis based on 
                         the given magnitude. e.g. 'N': raw value, 'K': every one thousand or 'M': every one million.

        Returns:
            ax (Axes): A matplotlib Axes representing the generated plot.
        """
        # Use seaborn to make figure looks modern
        sns.set()
        
        # Seaborn auto-coloring
        if colors is None:
            colors = sns.color_palette(n_colors=len(self.data))
        
        # Create a figure
        if 'ax' in kwargs:  # use provided Axes
            ax = kwargs['ax']
        else:  # create an Axes
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[6, 4])
        
        # Iterate all saved data
        for i, (name, D) in enumerate(self.data.items()):
            # Unpack the data
            data = D['data']
            xvalues = D['xvalues']
            
            # Enforce ndarray data with two-dimensional
            data = self._make_data(data)
            # Retrieve the generated color
            color = colors[i]
            
            # Compute the mean of the data
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            
            # Plot mean curve
            ax.plot(xvalues, 
                    mean, 
                    color=color, 
                    label=name)
            # Plot all uncertainty bands
            if scales is None or alphas is None:  # if nothing provided, one std uncertainty band by default
                scales = [1.0]
                alphas = [0.5]
            for scale, alpha in zip(scales, alphas):
                ax.fill_between(xvalues, 
                                mean - scale*std, 
                                mean + scale*std, 
                                facecolor=color, 
                                alpha=alpha)
                
        # Make legend for each mean curve (data item)
        if 'legend_loc' in kwargs:
            ax.legend(loc=kwargs['legend_loc'])
        else:
            ax.legend()
        
        # Make title if provided
        if 'title' in kwargs:
            ax.set_title(kwargs['title'])
        
        # Make x-y label if provided
        if 'xlabel' in kwargs:
            ax.set_xlabel(kwargs['xlabel'])
        if 'ylabel' in kwargs:
            ax.set_ylabel(kwargs['ylabel'])
        
        # Enforce min/max of axes if provided
        if 'xlim' in kwargs:
            ax.set_xlim(kwargs['xlim'])
        if 'ylim' in kwargs:
            ax.set_ylim(kwargs['ylim'])
            
        # Enforce log-scale of axes if provided
        if 'logx' in kwargs and kwargs['logx']:
            ax.set_xscale('log')
        if 'logy' in kwargs and kwargs['logy']:
            ax.set_yscale('log')
        
        # Enforce axis having integer coordinates if provided
        if 'integer_x' in kwargs and kwargs['integer_x']:
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        if 'integer_y' in kwargs and kwargs['integer_y']:
            ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
            
        # Enforce the maximum number of major ticks in horizontal axis
        if 'num_tick' in kwargs:
            ax.xaxis.set_major_locator(plt.MaxNLocator(kwargs['num_tick']))
            
        # Format the major ticks in horizontal axis
        if 'xscale_magnitude' in kwargs:
            format_function = partial(self.tick_formatter, scale_magnitude=kwargs['xscale_magnitude'])
            ax.xaxis.set_major_formatter(plt.FuncFormatter(format_function))
        
        return ax
    
    def _make_data(self, x):
        # Enforce the data being ndarray
        x = np.array(x)
        # Enforce 2-dim data
        if x.ndim == 1:
            x = np.expand_dims(x, 0)
        elif x.ndim == 2:
            x = x
        else:
            raise TypeError(f'The input data must be either one- or two-dimensional. But got {x.ndim}.')
            
        return x
    
    def _interp_data(self, all_x, all_y, num_points=500):
        """
        Piecewise linear interpolation for each line and return the x-y values from interpolated line. 
        
        This is for the case that multiple lines have different x value points, which is impossible
        to plot uncertainty bands. 
        
        Args:
            all_x (list): x values for each line
            all_y (list): y values for each line
            num_points (int): number of points to generate from interpolated lines. 
            
        Returns:
            new_x (list): shared query x values between minimum and maximum possible x values.
            interp_all_y (list): y values from each interpolated line. 
        """
        # Obtain minimum and maximum x values from given data
        # We iterate over each line, because they might have different length, and impossible to broadcast
        min_x = min([np.min(x) for x in all_x])
        max_x = max([np.max(x) for x in all_x])
        # Generate an array of new query x values between min and max possible x values
        new_x = np.linspace(min_x, max_x, num=num_points)
        
        # Generate new y values from each interpolated line given the shared query x values
        interp_all_y = [np.interp(new_x, x, y) for x, y in zip(all_x, all_y)]
        
        return new_x, interp_all_y
        
    def tick_formatter(self, x, pos, scale_magnitude=None):
        """
        A function to set major functional formatter. 

        Args:
            x (object): data value, internal argument used by Matplotlib
            pos (object): position, internal argument used by Matplotlib
            scale_magnitude (str): string description of scaling magnitude, use functools.partial
                to make a function specified with this scaler but without put it as a required argument. 
                Possible values: 
                    - 'N': no scaling, raw value
                    - 'K': every one thousand
                    - 'M': every one million
                When None is given, then it automatically detect for N, K or M. 

        Returns:
            A formatted string of the tick given the data value. 
        """
        msg = f'expected K, M or None, got {scale_magnitude}'
        assert scale_magnitude in ['N', 'K', 'M', None], msg
        
        # Auto-assign scale magnitude to K or M depending on the x value
        if scale_magnitude is None:
            if x < 1000:  # less than a thousand, so show raw value
                scale_magnitude = 'N'
            elif x >= 1000 and x < 1000000:  # between a thousand and a million
                scale_magnitude = 'K'
            elif x >= 1000000:  # more than a million
                scale_magnitude = 'M'
        
        # Make scaler based on string format
        if scale_magnitude == 'N':
            scaled_x = x
            scale_str = ''
        elif scale_magnitude == 'K':
            scaled_x = x/1000
            scale_str = 'K'
        elif scale_magnitude == 'M':
            scaled_x = x/1000000
            scale_str = 'M'

        # Format the data value and return it
        if scaled_x == 0:  # no format for zero
            return '0'
        elif scaled_x > 0 and scaled_x < 1:  # less than 1, then keep one decimal
            return f'{round(scaled_x, 1)}{scale_str}'
        else:  # more than 1
            return f'{int(scaled_x)}{scale_str}'
