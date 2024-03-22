"""
Module Name: Albedo
Module Author: Giovannina Mansir (nina.mansir@gmail.com)
Module Version: 0.0.1
Last Modified: 2023-12-13

Description:
This class takes a compressed and reduced FITS file of a planetary spectrum and contains
methods to analyze it and highlight information

Usage:
import matplotlib.pyplot as plt
import Alebedo
analysis = Albedo.DataAnalysis(body='')
Albedo.plot_albedo()

Dependancies:
-numpy
-matplotlib
-astropy
-copy
-glob
"""

import pdb
import numpy as np
from astropy import units as un
from astropy.io import fits
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import StrMethodFormatter
from matplotlib.gridspec import GridSpec
from scipy.interpolate import interp1d
from scipy.ndimage import convolve1d
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.signal import argrelextrema, peak_prominences
from scipy.ndimage import gaussian_filter1d
from datetime import datetime
import procastro
from PyAstronomy import pyasl
import astropy.constants as consts
from astropy import units as u
from astropy.nddata import StdDevUncertainty, NDData
from astropy.visualization import quantity_support
from astropy.modeling import models
from synphot import SourceSpectrum, units
from synphot import SpectralElement
from synphot import Observation
from synphot.models import Empirical1D, BlackBodyNorm1D
from brokenaxes import brokenaxes

#matplotlib.use('TkAgg')
#matplotlib.use('Agg')

class AlbedoAnalysis:
    def __init__(self, body):
        self.body = body.lower()
        self.work_dir = f'/home/gmansir/Thesis/{self.body.title()}/Data/'
        self.albedo_file = self.work_dir + f'PostMolecfit/MOV_{self.body.title()}_SCI_IFU_ALBEDO.fits'
        self.load_data()
        self.find_albedo()
        self.find_filter_colors()

    def load_data(self):
        # Johnson Filter Colors
        filter_keys = ['U_PHOT', 'B_PHOT', 'V_PHOT','R_PHOT', 'I_PHOT',
                       'J_PHOT','K_PHOT']
        try:
            hdu = fits.open(self.albedo_file, mode='readonly')

            # Check if the celestial body is in the header
            self.wav = hdu[0].data
            self.flux = hdu['FLUX'].data
            self.errors = hdu['ERRS'].data
            self.resp_crv = hdu['RESP_CRV'].data
            self.mask_tel = np.invert([bool(x) for x in hdu['MASK_TEL'].data])
            self.mask_wrn = np.invert([bool(x) for x in hdu['MASK_WRN'].data])
            self.bb_temp = hdu[0].header.get('BB_TEMP')
            self.flux_std = hdu[0].header.get('FLUX_STD')
            self.norm_idx = hdu[0].header.get('NORM_IDX')
            self.filters = [{fkey :hdu[0].header.get(fkey, 'Not Available')}
                            for fkey in filter_keys]

            print(f"Data loaded for celestial body: {self.body}")

        except FileNotFoundError:
            print(f"File not found: {self.albedo_file}")
        finally:
            # Close the file
            if 'hdu' in locals() and hdu is not None:
                hdu.close()

    def closest_index(self, arr, val):
        """
        Finds the index of the value closest to that requested in a given array

        :param arr: array
        :param val: value you are searching for
        :return: index in array of number closest to val
        """
        if type(arr) != list:
            arr = list(arr)
        close_func = lambda x: abs(x - val)
        close_val = min(arr, key=close_func)

        return arr.index(close_val)

    def annotate_plot(self, **kwargs):

        color_dict = {'Ammonia':'magenta', 'Acetylene':'red', 'CarbonMonoxide':'maroon',
                      'CarbonDioxide':'orangered','Diacetylene':'saddlebrown', 'Ethane':'darkorange',
                      'Ethylene':'orange', 'HydrogenSulfide':'darkgoldenrod', 'Methane':'olive',
                      'NitricOxide':'darkolivegreen', 'NitrogenOxide':'green', 'NitrogenDioxide':'darkgreen',
                      'Oxygen':'seagreen', 'Ozone':'teal', 'Phosphine':'indigo',
                      'SulfurDioxide':'darkviolet', 'Water':'purple'}

        if 'molecules' in kwargs:
            molecules = kwargs['molecules']
        else:
            molecules = list(color_dict.keys())
        if 'HydroCarbons' in molecules:
            molecules.remove('HydroCarbons')
            [molecules.append(m) for m in ['Methane', 'Ethane', 'Acetylene', 'Ethylene', 'Diacetylene']]
        if 'dontinclude' in kwargs:
            for nope in kwargs['dontinclude']:
                molecules.remove(nope)

        ylims = plt.ylim()
        xlims = plt.xlim()

        leg =[]
        if 'threshold' in kwargs.keys():
            scale = kwargs['threshold']
        else:
            scale = 'none'
        for m in molecules:
            fname = '/home/gmansir/Thesis/Elements/'+m+'.par'
            with open(fname) as f:
                reader = f.readlines()
                mol_lines = [10000/eval(row.split()[0]) for row in reader]
                intensities = [eval(row.split()[1]) for row in reader]
            if mol_lines == []:
                pass
            else:
                right_idx = self.closest_index(mol_lines, xlims[0])
                left_idx = self.closest_index(mol_lines, xlims[1])
                mini_intense = intensities[left_idx:right_idx]
                if scale == 'none':
                    threshold = 0.
                else:
                    threshold = np.mean(mini_intense) + np.std(mini_intense)*scale
                if mini_intense != []:
                    most_intense = np.max(mini_intense)
                    least_intense = np.min([v for v in mini_intense if v >= threshold])
                    leg.append(mpatches.Patch(color=color_dict[m], label=m))
                    for idx,l in enumerate(mol_lines[left_idx:right_idx]):
                        if mini_intense[idx] >= threshold:
                            alpha = (mini_intense[idx]-least_intense)/(most_intense-least_intense)
                            plt.axvline(l, ymin=ylims[0], ymax=ylims[1], color=color_dict[m], linestyle='solid',
                                        alpha=alpha)
        plt.legend(handles=leg, loc='upper left')

        plt.xlim(xlims)

    def annotate_plot_clean(self, ax, **kwargs):

        formula_dict = {'Ammonia': 'NH3', 'Acetylene': 'C2H2', 'CarbonMonoxide': 'CO',
                        'CarbonDioxide': 'CO2', 'Diacetylene': 'C4H2', 'Ethane': 'C2H6',
                        'Ethylene': 'C2H4', 'HydrogenSulfide': 'H2S', 'Methane': 'CH4',
                        'NitricOxide': 'NO', 'NitrogenOxide': 'N2O', 'NitrogenDioxide': 'NO2',
                        'Oxygen': 'O2', 'Ozone': 'O3', 'Phosphine': 'PH3', 'SulfurDioxide': 'SO2',
                        'Water': 'H2O'}

        hydrocarbons = ['Methane', 'Ethane', 'Acetylene', 'Ethylene', 'Diacetylene']
        hydrogenated = ['Ammonia', 'Acetylene', 'Diacetylene', 'Ethane', 'Ethylene',
                        'HydrogenSulfide', 'Methane', 'Phosphine', 'Water']
        if 'molecules' in kwargs:
            molecules = kwargs['molecules']
        else:
            molecules = list(formula_dict.keys())
        if 'HydroCarbons' in molecules:
            molecules.remove('HydroCarbons')
            [molecules.append(m) for m in hydrocarbons]
        if 'Hydrogenated' in molecules:
            molecules.remove('Hydrogenated')
            [molecules.append(m) for m in hydrogenated]
        if 'dontinclude' in kwargs:
            if 'telluric' in kwargs['dontinclude']:
                kwargs['dontinclude'].remove('telluric')
                [kwargs['dontinclude'].append(m) for m in ['Water','CarbonDioxide','CarbonMonoxide','Methane','Oxygen']]
            for nope in kwargs['dontinclude']:
                molecules.remove(nope)

        if 'titan' in molecules:
            molecules.remove('titan')
            [molecules.append(m) for m in ['Methane','Water','Ethane', 'Acetylene', 'Ethylene']]

        ylims = ax.get_ylim()
        xlims = ax.get_xlim()

        if 'tolerance' in kwargs.keys():
            tolerance = kwargs['tolerance']
        else:
            tolerance = 0.05
        # base off of percent of x-axis - adapt for zoom
        if 'threshold' in kwargs.keys():
            scale = kwargs['threshold']
        else:
            scale = 'none'
        # Make this number scientifically based for image caption
        # Colors? 20-th percentile vs 50 vs 80 for example
        mol_num = 0.
        for m in molecules:
            if m in hydrogenated:
                color = 'indianred'
                yloc = ylims[1]*(1.2+mol_num)
            else:
                color = 'steelblue'
                yloc = ylims[1] * (0.3 + mol_num)
            fname = '/home/gmansir/Thesis/Elements/' + m + '.par'
            with open(fname) as f:
                reader = f.readlines()
                mol_lines = [10000 / eval(row.split()[0]) for row in reader]
                intensities = [eval(row.split()[1]) for row in reader]
            if mol_lines == []:
                pass
            else:
                right_idx = self.closest_index(mol_lines, xlims[0])
                left_idx = self.closest_index(mol_lines, xlims[1])
                mol_lines = mol_lines[left_idx:right_idx]
                intensities = intensities[left_idx:right_idx]
                if scale == 'none':
                    threshold = 0.
                else:
                    threshold = np.mean(intensities) + np.std(intensities) * scale
                mini_mol_lines = []
                if intensities != []:
                    for idx, l in enumerate(mol_lines[left_idx:right_idx]):
                        if intensities[idx] >= threshold:
                            mini_mol_lines.append(l)
                telluric_ranges = []
                mini_mol_lines.sort()
                for line in mini_mol_lines:
                    # Check if the current line is within the tolerance of the last region
                    if len(telluric_ranges) > 0 and (line - telluric_ranges[-1][-1]) <= tolerance:
                        # If so, add the line to the current region
                        telluric_ranges[-1].append(line)
                    else:
                        # If not, create a new region with the current line
                        telluric_ranges.append([line])

                # Add horizontal bars for telluric absorption ranges
                for i in range(len(telluric_ranges)):
                    telluric_range = telluric_ranges[i]
                    length = telluric_range[-1] - telluric_range[0]
                    middle = np.mean(telluric_range)
                    #ax.barh(yloc, length, height=0.015,
                    ax.barh(0.75+mol_num, length, height=0.01,
                             left=telluric_range[0], color=color, alpha=0.6,
                            transform=plt.gca().get_xaxis_transform())
                    #ax.text(middle, yloc+0.03,
                    ax.text(middle,0.75+mol_num,
                            formula_dict[m], ha='center', va='bottom', color=color,
                            transform=plt.gca().get_xaxis_transform())
            mol_num -= 0.06
        ax.set_xlim(xlims)
        # pass ax limits as key words so at least it is easier to update
        # write keyword args explicitly when calling the method for ease of use

    def plot_data_and_regions(self, ax, wave, data):
        ax.plot(wave, data, color='black', linewidth=0.3)
        wave_wrn = np.ma.masked_array(wave, mask=self.mask_wrn)
        data_wrn = np.ma.masked_array(data, mask=self.mask_wrn)
        plt.plot(wave_wrn, data_wrn, linewidth=0.3, color='#121212')
        wave_tel = np.ma.masked_array(wave, mask=self.mask_tel)
        data_tel = np.ma.masked_array(data, mask=self.mask_tel)
        plt.plot(wave_tel, data_tel, linewidth=0.3, color='lightgray')
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(StrMethodFormatter('{x:.2f}'))
        ax.xaxis.set_minor_formatter(StrMethodFormatter('{x:.2f}'))

    def find_albedo(self):

        sp = SourceSpectrum(BlackBodyNorm1D, temperature=self.bb_temp)
        bb_flux = sp(self.wav * un.um)
        self.bb_flux = bb_flux / bb_flux[self.norm_idx]

        self.albedo = self.flux / self.bb_flux

    def find_filter_colors(self):

        # Add in Filters
        filters = {'johnson_u': 'U_PHOT', 'johnson_b': 'B_PHOT', 'johnson_v': 'V_PHOT',
                   'johnson_r': 'R_PHOT', 'johnson_i': 'I_PHOT', 'johnson_j': 'J_PHOT',
                   'johnson_k': 'K_PHOT'}
        # Create a spectrum object with unit info
        wave_ang = self.wav * 10000 * u.angstrom
        spectrum = self.flux * units.FLAM
        spectrum_obj = SourceSpectrum(Empirical1D, points=wave_ang, lookup_table=spectrum)

        self.filt_colors = dict.fromkeys(filters.keys())
        self.filt_locs = dict.fromkeys(filters.keys())
        for f in filters.keys():
            # Load a filter
            band = SpectralElement.from_filter(f)
            # Determine the color
            observation = Observation(spectrum_obj, band, force='taper')
            self.filt_colors[f] = observation.effstim(flux_unit='flam')
            self.filt_locs[f] = observation.effective_wavelength() / 10000

    def rv_correction(self):

        # File path
        sun_spec_vis = '/home/gmansir/Thesis/Sun/NARVAL.Sun.370_1048nm/NARVAL_Sun.txt'

        # Load in the data from the file
        vis_data = np.loadtxt(sun_spec_vis)

        # Assuming the first column is wavelength, second is flux, and third is errors
        sun_wave_vis = vis_data[:, 0] / 1000
        sun_flux_vis = vis_data[:, 1]

        wav_lim_low = self.closest_index(self.wav, sun_wave_vis[0]) + 1
        wav_lim_high = self.closest_index(self.wav, sun_wave_vis[-1])

        # File path
        sun_spec_ir = '/home/gmansir/Thesis/Sun/SOL_merged.fits'

        #Load in the data from the file
        with fits.open(sun_spec_ir) as hdul:
            sun_wave_ir = hdul[0].data / 1000
            sun_flux_ir = hdul[1].data

        solar_absorptions = [1.203485, 1.227412, 1.140696, 1.09180, 1.08727, 1.07899,
                             0.98920, 1.01487, 1.502919, 1.589281, 1.67553, 1.67236,
                             1.711338, 1.945829, 1.951114, 1.972798, 2.116965, 1.97823, 1.98675, 1.05880]
        solar_absorptions.sort()
        data_absorptions = [1.203104, 1.226985, 1.140339, 1.09144, 1.08689, 1.07864,
                            0.98903, 1.01461, 1.502442, 1.588782, 1.67501, 1.67189,
                            1.710817, 1.945225, 1.950529, 1.9721615, 2.116318, 1.9775, 1.98615, 1.05845]
        data_absorptions.sort()

        best_lines = [1,2,4,7,8,10,11,12,14,15,16,17,18]
        best_solars = [solar_absorptions[i] for i in best_lines]
        best_datas = [data_absorptions[i] for i in best_lines]


        # Plot the results
        fig1 = plt.figure(1)
        ax1 = fig1.add_subplot(111)
        ax1.plot(sun_wave_ir, sun_flux_ir-0.53, linewidth=0.3, color='black', label='NSO')
        ax1.plot(sun_wave_vis, sun_flux_vis-0.53, linewidth=0.3, color='grey', label='NARVAL')
        ax1.plot(self.wav, self.flux, linewidth='0.3', label=self.body)
        #for s, d, in zip(best_solars, best_datas):
        #    ax1.axvline(s, linewidth=0.3, color='orange')
        #    ax1.axvline(d, linewidth=0.3, color='rebeccapurple')
        plt.title('Sun Spectrum - NSO and NARVAL')
        plt.legend()
        plt.ylim(bottom=0.0)
        plt.show()

        def find_minimum(wavelengths, flux_data, target_wavelength, search_range):
            differences = np.abs(wavelengths - target_wavelength)
            indices_within_range = np.where(differences <= search_range)
            min_index = np.argmin(flux_data[indices_within_range])
            min_wavelength = wavelengths[indices_within_range][min_index]
            return min_wavelength

        def gaussian(x, amp, mean, stddev, baseline):
            return -amp * np.exp(-((x-mean)/(2*stddev))**2) + baseline

        shifts = []
        new_solar_mins = []
        new_data_mins = []
        fig5 = plt.figure(5)
        subplot=1
        index=1
        for solar_target, data_target in zip(best_solars, best_datas):
            # Find the central wavelength of the solar absorption
            solar_min = find_minimum(sun_wave_ir, sun_flux_ir, solar_target, 0.00001)
            # Find the central wavelength of the data absorption by finding the minimum (back-up technique)
            data_min = find_minimum(self.wav, self.flux, data_target, 0.001)
            # Clip out a range of wavelength values to focus on
            wavelength_range = np.abs(self.wav - data_target) <= 0.01
            x_data = self.wav[wavelength_range]
            y_data = self.flux[wavelength_range]
            # Search for large absorptions within the clipped region
            peaks, _ = find_peaks((-y_data)) #, prominence=0.0001)
            # Find the peak closest to the target wavelength (assumes I was fairly accurate)
            closest_peak_index = peaks[np.argmin(np.abs(x_data[peaks] - data_target))]
            # Clip a smaller region to really focus on this one peak
            peak_range = np.arange(max(0, closest_peak_index - 8), min(len(x_data), closest_peak_index + 8))
            # Make educated guesses about the amplitude and sigma for initialization
            amp_guess = max(y_data[peak_range]) - min(y_data[peak_range])
            sig_guess = np.abs(x_data[peak_range[0]] - x_data[peak_range[-1]]) / 4.
            initial_guess = [amp_guess, x_data[closest_peak_index], sig_guess, max(y_data[peak_range])]
            try:
                # Fit gaussians to hone in on the true amp, mean, and sigma, return the best mean
                params = curve_fit(gaussian, x_data[peak_range], y_data[peak_range], p0=initial_guess)
                gauss_min = params[0][1]
                # Plot a few to check result
                if subplot in [1,3,5]:
                    ax5 = fig5.add_subplot(3, 1, index)
                    low = self.closest_index(sun_wave_ir, x_data[0])
                    high = self.closest_index(sun_wave_ir, x_data[-1])
                    ax5.plot(sun_wave_ir[low:high],sun_flux_ir[low:high]-1, color='lightgray')
                    ax5.plot(x_data, y_data, color='black')
                    ax5.plot(x_data[peak_range], y_data[peak_range], color='blue')
                    ax5.plot(x_data[peak_range], gaussian(x_data[peak_range], params[0][0], params[0][1], params[0][2],
                                                          params[0][3]), color='deepskyblue')
                    ax5.axvline(solar_min, linestyle='--', color='orange', label='Solar Feature')
                    ax5.axvline(gauss_min, linestyle='--', color='deepskyblue', label='Gaussian Fit')
                    ax5.axvline(data_min, linestyle='--', color='violet', label='By Eye')
                    ax5.axvline((gauss_min-3.939984916106145e-05)/0.9995970818148777, linestyle='--',
                                color='limegreen', label='Shifted')
                    plt.legend()
                    index += 1
                subplot += 1
                # Save values
                new_solar_mins.append(solar_min)
                new_data_mins.append(gauss_min)
                shifts.append(solar_min-gauss_min)
            except RuntimeError:
                continue

        linregress = np.polyfit(new_solar_mins, new_data_mins, 1)
        print(f'Slope: {linregress[0]}, Y-Intercept: {linregress[1]}')
        regress_line = np.polyval(linregress, np.array(new_solar_mins))
        new_data_mins = np.array(new_data_mins)
        shifted_data_mins = (new_data_mins-linregress[1])/linregress[0]
        rvs = 2.99792458e8 * (new_data_mins-shifted_data_mins)/shifted_data_mins
        fig2 = plt.figure(2)
        gs = fig2.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0)
        ax2_main = fig2.add_subplot(gs[0])
        ax2_main.plot(new_solar_mins, new_data_mins, 'o', label='Data')
        ax2_main.plot(new_solar_mins, regress_line, label='Linear Regression')
        ax2_main.set_title(f'Linear Regression: {(1 - linregress[0]):.6f}, Mean: {np.mean(shifts):.6f},\nRV: {np.mean(rvs)/1000} km/s')
        ax2_main.set_xlabel('Solar Features (microns)')
        ax2_main.set_ylabel('Planet Features (microns)')
        ax2_residual = fig2.add_subplot(gs[1], sharex=ax2_main)
        ax2_residual.plot(new_solar_mins, regress_line - new_data_mins, 'o')
        ax2_residual.axhline(0, color='black', linestyle='--', linewidth=2)
        ax2_residual.set_xlabel('Solar Features (microns)')
        ax2_residual.set_ylabel('Residuals')
        plt.tight_layout()
        plt.show()

        self.rv_corr = (self.wav-linregress[1])/linregress[0]

        sun_lim = self.closest_index(sun_wave_ir, self.rv_corr[-1]) + 2
        wav_lim = self.closest_index(self.rv_corr, sun_wave_ir[0]) + 1

        # Compute Gaussian Convolution and fit for best kernel and sigma
        # Use just the features I picked out and mask with a 4* feature width
        # Form a few sub-groups and compare
        # Wavelength dependance on width? (XSHOOTER Manual?)
        def gaussian_kernel(size, sigma):
            x = np.linspace(int(-size / 2), int(size / 2), int(size))
            kernel = np.exp(-x ** 2 / (2 * sigma ** 2))
            return kernel / np.sum(kernel)

        def fitness_function(guesses, data_wav, data_flux, solar_wav, solar_flux):
            # Unpack parameters
            kernel_size, sigma = guesses

            kernel = gaussian_kernel(kernel_size, sigma)

            # Convolve the solar spectrum with the Gaussian kernel
            smoothed_solar_spec = convolve1d(solar_flux, kernel, mode='constant')

            interp_func = interp1d(solar_wav, smoothed_solar_spec, kind='slinear')
            interp_smooth = interp_func(data_wav)

            # Calculate mean squared error
            mse = np.mean((interp_smooth - data_flux) ** 2)

            return mse

        # Initial guess for kernel size and sigma
        # Optimize the fitness function
        #data_arr = self.rv_corr[wav_lim:]
        #solar_wav = sun_wave_ir[:sun_lim]
        #solar_wav= np.array(solar_wav.tolist())
        #solar_flux = sun_flux_ir[:sun_lim]
        #solar_flux = np.array(solar_flux.tolist())
        initial_guess=[50, 10]
        results = []
        for b in best_solars:
            data_idx = np.abs(self.rv_corr - b) <= 0.04
            data_wav = self.rv_corr[data_idx]
            data_flux = self.flux[data_idx]
            solar_idx = np.abs(sun_wave_ir - b) <= 0.04
            solar_wav = sun_wave_ir[solar_idx]
            solar_flux = sun_flux_ir[solar_idx]
            min_solar_wav = np.min(solar_wav)
            max_solar_wav = np.max(solar_wav)
            data_wav = data_wav[(data_wav >= min_solar_wav) & (data_wav <= max_solar_wav)]
            result = minimize(fitness_function, initial_guess, args=(data_wav,data_flux,solar_wav, solar_flux), method='Nelder-Mead')
            results.append(result.x)
            print(f"Wavelength: {b}, Size: {result.x[0]}, Sigma: {result.x[1]}")
        # Use these optimal values to smooth the solar spectrum
        optimal_gaussian = gaussian_kernel(size, sigma)
        smoothed_solar_spectrum = convolve1d(sun_flux_ir, optimal_gaussian, mode='constant')

        fig3 = plt.figure(3)
        ax3 = fig3.add_subplot(211)
        ax3.plot(sun_wave_ir[:sun_lim], sun_flux_ir[:sun_lim], color='black', linewidth=0.3, label='NSO Original')
        ax3.plot(sun_wave_ir[:sun_lim], smoothed_solar_spectrum[:sun_lim], linewidth=0.3, color='orange', label='NSO Conlvolved')
        ax3.plot(self.rv_corr[wav_lim:], self.flux[wav_lim:]+1.0, linewidth=0.3, label=self.body)
        ax3.set_ylim((0.5, 1.2))
        ax4 = fig3.add_subplot(212)
        ax4.plot(sun_wave_ir[:sun_lim], sun_flux_ir[:sun_lim], color='black', linewidth=0.3, label='NSO Original')
        ax4.plot(sun_wave_ir[:sun_lim], smoothed_solar_spectrum[:sun_lim], linewidth=0.3, color='orange', label='NSO Conlvolved')
        ax4.plot(self.rv_corr[wav_lim:], self.flux[wav_lim:]+1.0, linewidth=0.3, label=self.body)
        ax4.set_ylim((0.5, 1.2))
        ax4.set_xlim((1.48, 1.52))
        plt.xlabel('Wavelength (microns)')
        plt.ylabel('Relative Flux')
        plt.title('Convolution Results')
        plt.legend()
        plt.show()

        interp_function = interp1d(sun_wave_ir[:sun_lim], smoothed_solar_spectrum[:sun_lim], kind='slinear')
        interp_smoothed = interp_function(self.rv_corr[wav_lim:])

        w, h = plt.figaspect(1)
        fig1 = plt.figure(4, figsize=(w, h))
        plt.subplots_adjust(hspace=0.4)
        ax1 = fig1.add_subplot(211)
        self.plot_data_and_regions(ax1, self.rv_corr, self.flux)
        #ax1.set_xlim(left=0.98)
        ax1.set_xlim(1.138, 1.142)
        ax1.set_ylim(top=0.01, bottom=0.0)

        ax2 = fig1.add_subplot(212)
        ax2.plot(self.rv_corr[wav_lim:], self.flux[wav_lim:]/interp_smoothed, linewidth=0.3, color='black')
        ax2.set_xscale('log')
        ax2.set_xlim(1.138, 1.142)
        ax2.set_ylim(top=0.01, bottom=0.0)
        ax2.xaxis.set_major_formatter(StrMethodFormatter('{x:.2f}'))
        ax2.xaxis.set_minor_formatter(StrMethodFormatter('{x:.2f}'))
        plt.xlabel('Wavelength (microns)')
        plt.ylabel('Relative Flux')
        plt.suptitle('Albedo?')
        plt.show()

        self.sun_wave_ir = sun_wave_ir
        self.sun_flux_ir = sun_flux_ir

    def yiyo_rv(self, skipedge=20, num_pix=1500):

        # File path
        sun_spec_ir = '/home/gmansir/Thesis/Sun/SOL_merged.fits'

        #Load in the data from the file
        with fits.open(sun_spec_ir) as hdul:
            sun_wave_ir = hdul[0].data / 1000
            sun_flux_ir = hdul[1].data
        sun_lim = self.closest_index(sun_wave_ir, self.wav[-1]) + 2
        wav_lim = self.closest_index(self.wav, sun_wave_ir[0]) + 1
        sun_wave = sun_wave_ir[:sun_lim]
        sun_flux = sun_flux_ir[:sun_lim]
        wave = self.wav[wav_lim:]
        flux = self.flux[wav_lim:] + 1

        solar_absorptions = [1.203485, 1.227412, 1.140696, 1.09180, 1.08727, 1.07899,
                             0.98920, 1.01487, 1.502919, 1.589281, 1.67553, 1.67236,
                             1.711338, 1.945829, 1.951114, 1.972798, 2.116965]
        solar_absorptions.sort()
        data_absorptions = [1.203104, 1.226985, 1.140339, 1.09144, 1.08689, 1.07864,
                            0.98903, 1.01461, 1.502442, 1.588782, 1.67501, 1.67189,
                            1.710817, 1.945225, 1.950529, 1.9721615, 2.116318]
        data_absorptions.sort()

        best_lines = [1,3,6,7,9,10,11,13,14,15]
        best_solars = [solar_absorptions[i] for i in best_lines]
        best_datas = [data_absorptions[i] for i in best_lines]

        def compute_dRV(w, f, tw, tf, rvmin, rvmax, drv, skipedge=20, plot=False):
            # Function parameters:
            # w: observed wavelengths
            # f: observed flux
            # tw: template wavelengths
            # tf: template flux
            # rvmin, rvmax: minimum and maximum radial velocity values
            # drv: step size for radial velocity computation
            # skipedge: number of points to skip at each edge for cross-correlation
            # plot: flag for plotting intermediate results

            # Plotting the initial spectra and template (optional)
            if plot:
                plt.title('Template (blue) and spectra shifted (red), both normalized, before RV correction')
                plt.plot(tw, tf, 'b.-')
                plt.plot(w, f, 'r.-')
                plt.grid()
                plt.show()

            # Cross-correlation to compute radial velocity
            rv, cc = pyasl.crosscorrRV(w, f, tw, tf, rvmin, rvmax, drv, skipedge=skipedge)
            maxind = np.argmax(cc)

            # Display the result of cross-correlation
            print("Cross-correlation function is maximized at dRV = ", rv[maxind], " km/s")
            if rv[maxind] > 0:
                print(" A red-shift with respect to the template")
            else:
                print(" A blue-shift with respect to the template")

            # Plotting the cross-correlation function (optional)
            if plot:
                plt.plot(rv, cc, 'bp-')
                plt.plot(rv[maxind], cc[maxind], 'ro')
                plt.show()

                # Plotting the template and shifted spectra after RV correction (optional)
                plt.title('Template (blue) and spectra shifted (red), both normalized, after RV correction')
                plt.plot(tw, tf, 'b.-')
                plt.plot(w / (1 + rv[maxind] / consts.c.to(u.km / u.s).value), f, 'r.-')
                plt.grid()
                plt.show()

            # Return the determined radial velocity
            return rv[maxind]


        # Choose the radial velocity range and step size
        rvmin = -150
        rvmax = -111
        drv = 0.1

        # Optional: Choose parameters for cross-correlation (skipedge, plot)
        plot = False

        # Compute radial velocity shift
        sun_idx = [self.closest_index(sun_wave, s) for s in best_datas]
        sun2_idx = [self.closest_index(sun_wave, s) for s in best_solars]
        select = []
        for l,h in zip(sun_idx, sun2_idx):
            low = l-num_pix
            high = h+num_pix
            select.append((sun_wave[low], sun_wave[high]))
        sun_idx_masked = [np.where((sun_wave.data > s[0]) & (sun_wave.data < s[-1])) for s in select]
        data_idx_masked = [np.where((wave.data > s[0]) & (wave.data < s[-1])) for s in select]
        data_idx_masked = [d[0][3:-3] for d in data_idx_masked]
        rvs =[]
        for s, d in zip(sun_idx_masked, data_idx_masked):
            #print("Template Wavelength Range:", sun_wave[s].min(), sun_wave[s].max())
            #print("Data Wavelength Range:", wave[d].min(), wave[d].max())
            rv = compute_dRV(np.array(wave[d].tolist()), np.array(flux[d].tolist()), np.array(sun_wave[s].tolist()), np.array(sun_flux[s].tolist()), rvmin, rvmax,drv, skipedge=skipedge, plot=plot)
            rvs.append(rv)
        rv_shift = np.mean(rvs)

        # Apply RV correction to observed spectra
        #self.w_corr = wave / (1 - rv_shift / consts.c.to(u.km / u.s).value)
        self.w_corr = self.wav/((rv_shift*1000/2.99792458e8)+1)

        # Plot the template and shifted observed spectra after RV correction
        fig1 = plt.figure(6)
        ax1 = fig1.add_subplot(111)
        ax1.plot(sun_wave, sun_flux, color='deepskyblue', linewidth=0.3, label='Solar')
        ax1.plot(wave, flux, color='lightgray', linewidth=0.3, label='Before RV')
        ax1.plot(self.w_corr, self.flux, color='orangered', linewidth=0.3, label='After RV')
        plt.grid()
        plt.legend()
        plt.title(f'After RV correction: {rv_shift}')
        plt.show()

        # Now, w_corr contains the observed wavelengths corrected for the radial velocity shift

    def plot_albedo(self):
        """
        Plots the original cleaned spectrum and blackbody, as well as the
        division of the spectrum by the blackbody in Figure 1
        """
        w, h = plt.figaspect(1)
        fig1 = plt.figure(1, figsize=(w, h))
        plt.subplots_adjust(hspace=0.4)
        ax1 = fig1.add_subplot(211)
        self.plot_data_and_regions(ax1, self.wav, self.flux)
        ax1.plot(self.wav, self.bb_flux, label='Synphot Blackbody')

        ax2 = fig1.add_subplot(212)
        self.plot_data_and_regions(ax2, self.wav, self.albedo)

        for f in self.filt_colors.keys():
            ax2.plot(self.filt_locs[f], self.filt_colors[f], 'o', color='orangered')

        ax2.set_yscale('log')
        ax2.set_ylim(10e-4, 10e0)
        current_date = datetime.now().strftime('%Y-%m-%d')
        fig1.suptitle(f'{self.body.title()}\nGenerated on: {current_date}, '
                       f'\nCreated new file for analysis')

    def plot_analysis(self, **kwargs):

        zoom_regions = {'titan': [(0.98, 1.14), (1.5, 1.7)]
                        }

        w, h = plt.figaspect(1)
        fig1 = plt.figure(2, figsize=(w, h))
        # Plotting with broken axes using GridSpec
        spec = plt.GridSpec(2, 1, hspace=0.4)

        # Plotting the full spectrum in the top subplot
        ax1 = fig1.add_subplot(spec[0])
        self.plot_data_and_regions(ax1, self.wav, self.albedo)
        self.annotate_plot_clean(ax1, **kwargs)

        ax1.set_yscale('log')
        ax1.set_ylim(10e-3, 10e0)
        current_date = datetime.now().strftime('%Y-%m-%d')
        fig1.suptitle(f'{self.body.title()}\nGenerated on: {current_date}, '
                      f'\nAdded Plot Annotation and Zoomed Regions')

        # Creating a broken axis for the important regions in the bottom subplot
        #ax2 = fig1.add_subplot(spec[1])
        bax = brokenaxes(xlims=(zoom_regions[self.body]), subplot_spec=spec[1],
                         xscale='log', yscale='log')
        bax.loglog(self.wav, self.albedo, linewidth=0.3, color='black')
        wave_wrn = np.ma.masked_array(self.wav, mask=self.mask_wrn)
        data_wrn = np.ma.masked_array(self.albedo, mask=self.mask_wrn)
        bax.loglog(wave_wrn, data_wrn, linewidth=0.3, color='#121212')
        wave_tel = np.ma.masked_array(self.wav, mask=self.mask_tel)
        data_tel = np.ma.masked_array(self.albedo, mask=self.mask_tel)
        bax.loglog(wave_tel, data_tel, linewidth=0.3, color='lightgray')
        for ax in bax.axs:
            ax.xaxis.set_major_formatter(StrMethodFormatter('{x:.2f}'))
            ax.xaxis.set_minor_formatter(StrMethodFormatter('{x:.2f}'))
        #self.annotate_plot_clean(bax, **kwargs)

    def rv_compare(self):

        self.yiyo_rv(skipedge=20, num_pix=1500)
        self.rv_correction()

        solar_absorptions = [1.203485, 1.227412, 1.140696, 1.09180, 1.08727, 1.07899,
                             0.98920, 1.01487, 1.502919, 1.589281, 1.67553, 1.67236,
                             1.711338, 1.945829, 1.951114, 1.972798, 2.116965]
        solar_absorptions.sort()
        data_absorptions = [1.203104, 1.226985, 1.140339, 1.09144, 1.08689, 1.07864,
                            0.98903, 1.01461, 1.502442, 1.588782, 1.67501, 1.67189,
                            1.710817, 1.945225, 1.950529, 1.9721615, 2.116318]
        data_absorptions.sort()

        best_lines = [1,3,6,7,9,10,11,13,14,15]
        best_solars = [solar_absorptions[i] for i in best_lines]
        best_datas = [data_absorptions[i] for i in best_lines]

        fig = plt.figure(8)
        for idx, wv in enumerate([1,2,3]):
            i = idx + 1
            range = 0.001
            ax = fig.add_subplot(3, 1, i)
            slow = self.closest_index(self.sun_wave_ir, best_solars[wv] - range)
            shigh = self.closest_index(self.sun_wave_ir, best_solars[wv] + range)
            ax.plot(self.sun_wave_ir[slow:shigh], self.sun_flux_ir[slow:shigh] / np.mean(self.sun_flux_ir[slow:shigh]),
                    color='lightgray', linewidth=0.3, label='Sun')
            wlow = self.closest_index(self.wav, best_solars[wv] - range)
            whigh = self.closest_index(self.wav, best_solars[wv] + range)
            ax.plot(self.wav[wlow:whigh], self.flux[wlow:whigh] / np.mean(self.flux[wlow:whigh]), color='black',
                    linewidth=0.9, label='Original')
            ylow = self.closest_index(self.w_corr, best_solars[wv] - range)
            yhigh = self.closest_index(self.w_corr, best_solars[wv] + range)
            ax.plot(self.w_corr[ylow:yhigh], self.flux[ylow:yhigh] / np.mean(self.flux[ylow:yhigh]), color='forestgreen',
                    linewidth=0.9, label='CrossCorr')
            nlow = self.closest_index(self.rv_corr, best_solars[wv] - range)
            nhigh = self.closest_index(self.rv_corr, best_solars[wv] + range)
            ax.plot(self.rv_corr[nlow:nhigh], self.flux[nlow:nhigh] / np.mean(self.flux[nlow:nhigh]),
                    color='rebeccapurple', linewidth=0.9, label='Gaussfit')
            ax.axvline(best_solars[wv], color='orange', linestyle='--')
            ax.axvline(best_datas[wv], color='black', linestyle='--')
        plt.title('RV Correction Comparison')
        plt.legend()
        plt.show()


