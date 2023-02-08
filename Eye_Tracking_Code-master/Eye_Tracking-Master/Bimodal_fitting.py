from plotting_helpers import *
import pandas as pd
from sklearn.neighbors import KernelDensity
import csv
#from unidip import UniDip
from datetime import datetime
import os
from os import path
import re
import seaborn as sns

import csv
import numpy as np
from datetime import datetime


#plt.style.use('dark_background')


class EyeConvergenceAnalyser(BasePlotting):
    """Class for analysing eye convergence.

    Parameters
    ----------
    data : array, shape (n_samples,)
        Array containing eye vergence angles (in degrees). Positive angles mean eyes are converged and negative angles
        mean eyes are diverged.

    bandwidth : float, default: 2.0
        The bandwidth of the kernel used for kernel density estimation (degrees).

    default_threshold : float, default: 50.
        The default value of the eye convergence threshold (degrees).

    threshold_limits : tuple, default: (35., 65.)
        The range of angles (degrees) in which to search for the convergence threshold.

    verbose : bool, default: True
        Whether to be verbose.

    Other Parameters
    ----------------
    kwargs : dict
        Passed to rcParams to change defaults to plotting.

    Attributes
    ----------
    min_angle : int
        The minimum eye vergence angle in the data (rounded down to the nearest integer).

    max_angle : int
        The maximum eye vergence angle in the data (rounded up to the nearest integer).

    bin_edges : 1D array
        Bin edges for plotting histograms and performing kernel density estimation. Values range from the minimum angle
        to the maximum angle in one degree steps.

    counts : 1D array
        Counts for plotting histograms. The number of frames in each bin.

    kde_counts : 1D array
        Computed normalised counts in each bin after performing kernel density estimation on the data.

    threshold : int
        The threshold used to define eye convergence. Calculated as the antimode (local minimum) of the distribution of
        eye vergence angles after kernel density estimation.

    convergence_score : float (between 0 and 1)
        The proportion of frames that are above the convergence threshold.

    Notes
    -----
    If a fish performs prey capture, then the distribution of eye vergence angles should be bimodal where the first and
    largest peak corresponds to the eyes being in a resting position and the second, smaller peak corresponds to a state
    where the eyes are converged (i.e. prey capture). A convergence score is defined as the proportion of time the fish
    spends with its eyes in the converged state and so serves as a direct readout of the amount of time a fish spends
    doing prey capture.

    To calculate an eye convergence score, kernel density estimation is first performed on the data. This smooths the
    distribution of eye vergence angles which is necessary for calculating the convergence threshold. Normalised values
    of the kernel density estimation in one degree bins are stored in the kde attribute.

    The convergence threshold (stored in the threshold attribute) is defined as the antimode in the distribution of eye
    vergence angles, i.e. the local minimum between the main peak (eyes at rest) and the second peak (eyes converged)
    that lies within the threshold limits.

    Methods should be called in the following order:
        - kernel_density_estimation
        - find_convergence_threshold
        - calculate_convergence_score

    The class has various plotting methods for:
        - checking the distribution of eye vergence angles in the data
        - checking the output of the kernel density estimation
        - checking the convergence threshold
    """

    def __init__(self, data, file_name, bandwidth=2.0, default_threshold=20.,
                 threshold_limits=(25, 100), verbose=True, **kwargs):

        # BasePlotting.__init__(self, **kwargs)
        self.data = data
        self.file_name = file_name
        self.bandwidth = bandwidth
        self.verbose = verbose
        self.threshold = default_threshold
        self.min_threshold = min(threshold_limits)
        self.max_threshold = max(threshold_limits)
        print(self.max_threshold)
        print(self.file_name)

    def kernel_density_estimation(self):
        """Performs kernel density estimation of the data using a gaussian with the given bandwidth.

        Returns
        -------
        self

        Notes
        -----
        After calling this method, the following attributes become accessible:
        min_angle, max_angle, bin_edges, counts, kde
        """
        print((self.data).ndim)

        print(self.bandwidth)
        self.min_angle = np.floor(self.data.min())

        self.max_angle = np.ceil(self.data.max())

        self.bin_edges = np.arange(self.min_angle, self.max_angle + 1)

        # self.bin_edges = np.arange(3.0, 63.0)

        self.counts, self.bin_edges = np.histogram(self.data, bins=self.bin_edges)
        if self.verbose:
            print('Performing kernel density estimation...')
        # perform kernel density estimation
        self.kde = KernelDensity(kernel='gaussian', bandwidth=self.bandwidth).fit(np.expand_dims(self.data, 1))
        # get the log counts
        log_counts = self.kde.score_samples(np.expand_dims(self.bin_edges, 1))
        # convert logarithmic values to absolute counts
        self.kde_counts = np.exp(log_counts)
        # find the value of the mode
        self.mode = self.bin_edges[np.argmax(self.kde_counts)]
        if self.verbose:
            print('done!')
        return self

    def find_convergence_threshold(self):
        """Finds the antimode of kernel density estimated distribution and sets it as the convergence threshold.

        Returns
        -------
        threshold : int
            The threshold used to define eye convergence (the antimode of the distribution of eye vergence angles).

        Notes
        -----
        This method finds the local minimum of the eye convergence distribution that lies within the threshold limits.
        If a local minimum does not exist within the limit, the convergence threshold defaults to the initial value.

        Smoothed distribution:
           _
          / \   _
         /   \_/ \
        /         \

        First derivative:

        + \     _
        0  \   / \
        -   \_/   \

        Local maxima and minima occur when the first derivative is zero. Specifically, local minima occur when there is
        an inversion of the sign of the first derivative (from negative to positive).

        Sign of first derivative:

        + .    .
        0  .  . ..
        -   ..    .

        Second derivative:

         (local min)
              |
              V
        +     _
        0    / \_
        -  _/    \

        """
        # take the derivative of the distribution
        diffed = np.diff(self.kde_counts)
        # smooth the differentiated data
        smoothed = pd.Series(diffed).rolling(7, min_periods=0, center=True).mean().values
        # take the sign of the smoothed data (is the function increasing or decreasing)
        signed = np.sign(smoothed)
        # take the derivative of the sign of the first derivative
        second_diff = np.diff(signed)
        # find the indices of local minima (i.e. where the sign of first derivative goes from negative to positive)
        local_minima = np.where(second_diff < 0)[0] + 1
        # find values of the antimodes
        antimodes = self.bin_edges[local_minima]

        try:
            # Try to find an antimode within the threshold range
            #self.threshold = antimodes[(antimodes > self.min_threshold) & (antimodes < self.max_threshold)][0]
            self.threshold=antimodes[len(antimodes)-1]
        except IndexError:  # local minimum does not exist within the threshold range
            if self.verbose:
                print('No local minimum within limits!')
        if self.verbose:
            print('Eye convergence threshold:', self.threshold)
        return self.threshold

    def calculate_convergence_score(self):
        """Calculates the convergence score, i.e. the proportion of frames above the convergence threshold.

        Returns
        -------
        convergence_score: float
            The proportion of frames that are above the convergence threshold.
        """
        # find frames that are above threshold
        above_threshold = self.data[self.data >= self.threshold]
        # get number of frames above threshold
        converged_counts = len(above_threshold)
        # get total number of frames in the data
        total_counts = len(self.data)
        # calculate proportion of frames that are above threshold
        score = float(converged_counts) / float(total_counts)
        self.convergence_score = score
        if self.verbose:
            print('Eye convergence score:', self.convergence_score)
        return self.convergence_score

    def plot_histogram(self, save=False, output_path=None):
        """Plots a histogram of eye vergence angles.

        Parameters
        ----------
        save : bool, optional (default = False)
            - True: the figure is saved to the specified or default output path
            - False: the figure is shown in a window

        output_path : str, optional (default = None)
            Where to save the figure if save == True. The default path is: 'histogram_of_eye_convergence_angles.png'
        """

        fig, ax = plt.subplots(figsize=(8, 6))

        fig.suptitle('Distribution of eye vergence angles (threshold = ' + str(self.threshold) + ')')

        # ax.hist(self.data, bins=self.bin_edges,color='white')
        sns.distplot(self.data,ax=ax,color="red",bins=50,hist_kws=dict(edgecolor='white', linewidth=2))

        ax.set_xlabel('Angle (degrees)')
        ax.set_ylabel('Number of frames')

        if save:
            if output_path is None:
                output_path = 'histogram_of_eye_convergence_angles' + self.file_name + '.png'
            self.save_figure(fig, output_path)
        else:

            outdir = os.getcwd()

            if not (path.isdir(os.path.join(outdir, str(self.file_name)))):
                os.mkdir(os.path.join(outdir, str(self.file_name)))

            out_path = os.path.join(outdir, str(self.file_name))
            fig.savefig(out_path + '/' + 'histogram_of_eye_convergence_angles_'  + '.png')

    def plot_kernel_density_estimation(self, save=False, output_path=None):
        """Plots a histogram of observed eye vergence angles and the estimated distribution.

        Parameters
        ----------
        save : bool, optional (default = False)
            - True: the figure is saved to the specified or default output path
            - False: the figure is shown in a window

        output_path : str, optional (default = None)
            Where to save the figure if save == True. The default path is:
            'kernel_density_estimation_of_eye_convergence_angles.png'
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.suptitle('Kernel density estimation of eye vergence angles')

        # upper = []
        # lower = []
        # if self.threshold > 30:
        #     print("a")
        #     for i in range(0, len(self.data)):
        #         if self.data[i] > self.threshold - 2:
        #             upper.append(self.data[i])
        #         else:
        #             lower.append(self.data[i])
        # else:
        #     print("b")
        #     upper = self.data
        #
        # sns.distplot(lower, color='r')
        # sns.distplot(upper, color='b')

        ax.hist(self.data, bins=self.bin_edges)
        ax.plot(self.bin_edges, self.kde_counts * len(self.data), linewidth=3)

        ax.set_xlabel('Angle (degrees)')
        ax.set_ylabel('Number of frames')

        if save:
            if output_path is None:
                output_path = 'kernel_density_estimation_of_eye_convergence_angles' + self.file_name + '.png'
            self.save_figure(fig, output_path)
        else:
            outdir = os.getcwd()

            if not (path.isdir(os.path.join(outdir, str(self.file_name)))):
                os.mkdir(os.path.join(outdir, str(self.file_name)))

            out_path = os.path.join(outdir, str(self.file_name))
            fig.savefig(out_path + '/' + 'kernel_density_estimation_of_eye_convergence_angles_'  + '.png')

    def plot_threshold(self, save=False, output_path=None):
        """Plots a histogram of observed eye vergence angles and the estimated distribution.

        Makes two subplots. The left subplot shows the estimated distribution of eye vergence angles with everything
        above threshold shaded. The right subplot shows the observed distribution of eye vergence angles and the
        threshold (which is used to calculate the convergence_score).

        Parameters
        ----------
        save : bool, optional (default = False)
            - True: the figure is saved to the specified or default output path
            - False: the figure is shown in a window

        output_path : str, optional (default = None)
            Where to save the figure if save == True. The default path is: 'eye_convergence_threshold.png'
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharex=True, sharey=True)
        fig.suptitle('Eye convergence threshold')

        converged = self.bin_edges >= self.threshold

        sns.kdeplot (data=self.data,ax=ax1,linewidth=3,color='black')
                     #self.kde_counts * len(self.data), linewidth=3)
        ax1.fill_between(self.bin_edges[converged], 0, self.kde_counts[converged] * len(self.data))

        ax1.set_title('Kernel density estimation')
        ax1.set_xlabel('Angle (degrees)')
        ax1.set_ylabel('Counts')

        ax2.hist(self.data, bins=self.bin_edges)
        ax2.plot([self.threshold, self.threshold], [0, self.counts.max()], c='k', linestyle='dashed')

        ax2.set_title('Raw counts')
        ax2.set_xlabel('Angle (degrees)')

        ax1.set_xlim(-25, 100)
        ax1.set_xticks(np.arange(-25, 125, 25))

        if save:
            if output_path is None:
                output_path = 'eye_convergence_threshold' + self.file_name + '.png'
            self.save_figure(fig, output_path)
            plt.close(fig)
        else:
            outdir = os.getcwd()

            if not (path.isdir(os.path.join(outdir, str(self.file_name)))):
                os.mkdir(os.path.join(outdir, str(self.file_name)))

            out_path = os.path.join(outdir, str(self.file_name))
            fig.savefig(self.file_name + '/' + 'eye_convergence_threshold_' + '.png')

    plot_kde = plot_kernel_density_estimation

def main():

    ob1 = EyeConvergenceAnalyser(df)
    ob1.kernel_density_estimation()
    ob1.find_convergence_threshold()
    ob1.calculate_convergence_score()
    ob1.plot_histogram()
    ob1.plot_kernel_density_estimation()
    ob1.plot_threshold()
    ob1.threshold

    xval = ob1.threshold
    if xval < 30:
        xval = 40
    print(xval)

if __name__ == "__main__":
        main()





