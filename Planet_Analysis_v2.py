"""
Module Name: Planet_Analysis_v2
Module Author: Giovannina Mansir (nina.mansir@gmail.com)
Module Version: 2.0.1
Last Modified: 2023-04-06

Description:
This class is for manipulating IFU images from XSHOOTER and compliling it into a library for ease of understanding and
modeling.

Usage:
import matplotlib.pyplot as plt
import Planet_Analysis_v2 as PA
analysis = PA.DataAnalysis(body='')
work_dir = '/home/gmansir/Thesis/Titan_tests/reflex_end_products/2023-03-20T15:27:11/XSHOO.2019-09-26T02:17:55.664_tpl/'
analysis.spectral_cleanup(work_dir)
analysis.edge_matching()

Dependancies:
-numpy
-matplotlib
-astropy
-copy
-glob
"""

import os
import pdb
import warnings
import copy
import glob
import pandas as pd
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clip
from astropy.modeling import models
from astropy import units as u
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from matplotlib.ticker import StrMethodFormatter
from scipy.interpolate import UnivariateSpline as uspline
from scipy.interpolate import interp1d
from astropy import units as un
from specutils import Spectrum1D
from specutils.fitting import fit_generic_continuum
import re
from datetime import datetime

matplotlib.use('TkAgg')

class DataAnalysis():

    '''
    A collection of methods for flexible analysis of our planetary library data
    '''

    def __init__(self, **kwargs):

        """
        Initializes a new instance of the DataAnalysis class.

        Args:
            body (str): The name of the celestial body of focus.
        """

        self.body = None
        celestial_bodies = ['titan','enceladus', 'neptune1', 'neptune2', 'neptune3', 'neptune4',
                            'uranus1', 'uranus2', 'uranus3', 'uranus4', 'saturn1', 'saturn2',
                            'saturn3', 'saturn4', 'saturn5', 'saturn6', 'saturn7', 'saturn8',
                            'saturn9', 'saturn10', 'saturn11', 'saturn12', 'telluric', 'sunlike',
                            ]
        celestial_bodies = ', '.join(celestial_bodies)

        if 'body' in kwargs:
            body = kwargs['body'].lower()
            while body not in celestial_bodies:
                print("Invalid object entered.")
                body = input(f"Please enter a valid celestial body ({celestial_bodies}): ").lower()
            self.body = body
        else:
            self.body = input(f"Please enter a celestial body ({celestial_bodies}): ").lower()
            while self.body not in celestial_bodies:
                print("Invalid object entered.")
                self.body = input(f"Please enter a valid celestial body ({celestial_bodies}): ").lower()

        self.work_dir = f'/home/gmansir/Thesis/{self.body.title()}/Data/'

        print(f"Instance initiated for {self.body.title()}.")

    def make_molefit_ready(self, dumfile, specfile):

        self.dumfile = dumfile
        self.specfile = specfile
        self.expected_EXTNAMEs = {'FLUX': -1, 'ERRS': -1, 'QUAL': -1}
        HDUlist_spec = fits.open(self.specfile)
        HDUlist_dum = fits.open(self.dumfile)
        for (i_hdu, HDU) in enumerate(HDUlist_dum):
            if HDU.header.get('EXTNAME') in self.expected_EXTNAMEs.keys():
                self.expected_EXTNAMEs[HDU.header['EXTNAME']] = i_hdu

        nHDUlist=copy.deepcopy(HDUlist_dum)
        _HDUlist=copy.deepcopy(HDUlist_dum)
        # Append the HDUs so that we can re-pack...
        # It's really just so that we save the full header info and image format
        for ext in ['FLUX', 'ERRS', 'QUAL'] :
          nHDUlist.append(_HDUlist[self.expected_EXTNAMEs[ext]])
          nHDUlist[-1].header['EXTNAME']=_HDUlist[self.expected_EXTNAMEs[ext]].header['EXTNAME']+"_MOLECFIT_READY"
          nHDUlist[-1].data=np.array(np.shape(_HDUlist[self.expected_EXTNAMEs[ext]].data),dtype='int16')

        # Fix the WAVE scale keys...
        for (i_hdu,HDU) in enumerate(HDUlist_dum):
            for k in ['CTYPE', 'CRPIX', 'CRVAL', 'CDELT', 'CUNIT']:
                if k+'3' in HDUlist_dum[i_hdu].header: nHDUlist[i_hdu].header[k+'1'] = HDUlist_dum[i_hdu].header[k+'3']
                if k+'2' in nHDUlist[i_hdu].header : del nHDUlist[i_hdu].header[k+'2']
                if k+'3' in nHDUlist[i_hdu].header : del nHDUlist[i_hdu].header[k+'3']
        # Remove the CD matrix...
            for j in range(3):
                for k in range(3):
                    if 'CD%d_%d' %(j+1,k+1) in nHDUlist[i_hdu].header : del nHDUlist[i_hdu].header['CD%d_%d' %(j+1,k+1)]
        # Remove the SPECSYS key...
            if 'SPECSYS' in nHDUlist[i_hdu].header : del nHDUlist[i_hdu].header['SPECSYS']

        # Get data from specfile
        # FLUX:
        newFLUX = HDUlist_spec[1].data
        # ERRS:
        newERRS = HDUlist_spec[2].data
        # QUAL:
        newQUAL = HDUlist_spec[3].data
        # SPIKES: moving median filled data to avoid using for molecfit
        newSPIKE = HDUlist_spec[4].data
        # CLOSE:
        HDUlist_spec.close()

        # Sum each column of slit....
        nHDUlist[self.expected_EXTNAMEs['FLUX']].data = newFLUX
        #try:
        nHDUlist[self.expected_EXTNAMEs['ERRS']].data = newERRS
        nHDUlist[self.expected_EXTNAMEs['QUAL']].data = newQUAL
        spike_HDU = fits.ImageHDU(data=newSPIKE, name='SPIKES')
        nHDUlist.append(spike_HDU)
        fname = self.dumfile.replace('.fits', '_MOLECFIT_READY.fits').replace("MERGE3D", "MERGE1D")
        for (i_hdu, HDU) in enumerate(HDUlist_dum):
            nHDUlist[i_hdu].header['HIERARCH ESO PRO CATG'] = HDUlist_dum[0].header[
                                                                      'HIERARCH ESO PRO CATG'].replace("MERGE3D",
                                                                                                       "MERGE1D") + '_SUM_SUM'
        nHDUlist.writeto(fname, output_verify="fix+warn", overwrite=True, checksum=True)
        HDUlist_dum.close()
        print('File Written')
        #except KeyError:
        #    print('Failed to Write Molecfit Ready File')

    def spectral_cleanup(self, directory=None, molecfit_ready=True):

        """
        Collects files from a post XSHOOTER pipeline reduction, separated out into individual spaxel spectra. Then
        calculates their moving maximums, sigma clips the spikes, and median combines them. It saves the normalized
        median combined and sigma clipped spectrum into a file to be processed with make_molecfit_ready()

        :param work_dir: The working directory containing the separated spaxel spectra for processing

        :return: A normalized and sigma clipped fits file
        """

        if directory == None:
            pre_dir = self.work_dir + 'PreMolecfit/reflex_end_products/'
        else:
            pre_dir = directory
        flist = glob.glob(pre_dir+'/*')
        target = f'IFU_MERGE3D_DATA_OBJ_'
        base_file = [f for f in flist if target in f]
        base_file.sort()
        if len(base_file) == 0:
            raise ValueError(f'Please check directory, no viable images found in {pre_dir}.')
        band = base_file[0][-8:-5].upper()
        spaxels = [f for f in flist if 'pixel' in f]

        w, h = plt.figaspect(0.25)
        fig1 = plt.figure(1, figsize=(w, h))
        ax = fig1.add_subplot(211)
        all_data = []
        norm_data = []
        orig_data = []
        max_norm = []
        poly_specs = []
        for sp in spaxels:
            with fits.open(sp) as hdul:
                data = copy.deepcopy(hdul[0].data)
                odata = data / data[8888]
                orig_data.append(odata)
                ndata = data / np.median(data[1843:1860])
                norm_data.append(ndata)
                window_size = 24
                moving_medians = list(np.zeros(window_size - 1) + 2)
                moving_maximum = list(np.zeros(window_size - 1) + 2)
                mm = 0
                while mm < len(ndata) - window_size + 1:
                    window = ndata[mm:mm + window_size]
                    win_median = np.median(window)
                    moving_medians.append(win_median)
                    sorted_idx = np.argsort(window)
                    forth_highest_idx = sorted_idx[-8]
                    forth_highest_val = window[forth_highest_idx]
                    moving_maximum.append(forth_highest_val)
                    mm += 1
                moving_medians = np.array(moving_medians, dtype='f')
                residuals = ndata - moving_medians
                with warnings.catch_warnings():  # Ignore warnings due to NaNs or Infs
                    warnings.simplefilter("ignore")
                    sigclip = sigma_clip(residuals, sigma=4, cenfunc='median')
                clipped_data = np.ma.masked_array(ndata, mask=sigclip.mask)
                # filled_data = clipped_data.filled(moving_medians)
                filled_data = clipped_data.filled(np.nan)
                fdata = filled_data.tolist()
                all_data.append(fdata)
                CRVAL1 = hdul[0].header['CRVAL1']
                CDELT1 = hdul[0].header['CDELT1']
                NAXIS1 = hdul[0].header['NAXIS1']
                wave = np.array([CRVAL1 + CDELT1 * i for i in range(NAXIS1)]) / 1000.
                header = hdul[0].header
                moving_maximum = np.array(moving_maximum, dtype='f')
                poly_coeffs = np.polyfit(wave, moving_maximum, deg=6)
                poly_spec = np.polyval(poly_coeffs, wave)
                poly_specs.append(poly_spec)
                poly_norm = data / poly_spec
                max_norm.append(poly_norm)
                ax.plot(wave, filled_data, linewidth=0.5, label=(sp[-9:-5]))
        med_combined = np.nanmedian(all_data, axis=0)
        norm_combined = np.nanmedian(norm_data, axis=0)
        orig_combined = np.nanmedian(orig_data, axis=0)
        max_combined = np.nanmedian(max_norm, axis=0)
        max_combined = max_combined / np.median(max_combined[18433:18600])
        polys_combined = np.nanmedian(poly_specs, axis=0)
        spikes = np.nonzero(np.isnan(med_combined))
        med_errs = np.std(all_data, axis=0)
        window_size = 24
        moving_medians = list(np.zeros(window_size - 1) + 2)
        mm = 0
        while mm < len(med_combined) - window_size + 1:
            window = med_combined[mm:mm + window_size]
            win_median = np.nanmedian(window)
            moving_medians.append(win_median)
            mm += 1
        moving_medians = np.array(moving_medians, dtype='f')
        if isinstance(spikes[0], np.ndarray):
            spikes = np.concatenate(spikes)
        if len(spikes) == 0:
            pass
        else:
            for i in spikes:
                if i <= 2 or i >= len(med_combined) - 3:
                    continue
                med_combined[i - 2] = moving_medians[i - 2]
                med_combined[i - 1] = moving_medians[i - 1]
                med_combined[i] = moving_medians[i]
                med_combined[i + 1] = moving_medians[i + 1]
                med_combined[i + 2] = moving_medians[i + 2]

        print('Data Calculated')
        ax2 = fig1.add_subplot(212)
        ax2.plot(wave, orig_combined, linewidth=0.5, label='Original/float')
        ax2.plot(wave, norm_combined, linewidth=0.5, label='Original/median(range)')
        ax2.plot(wave, med_combined, linewidth=0.5, label='Range Norm, Sig Clip, Med Filled')
        ax2.plot(wave, max_combined, linewidth=0.5, label='Original/moving maximum')
        ax2.plot(wave, polys_combined, linewidth=0.5, label='Polyfit Spectra median combined')
        ax2.legend()
        ax.set_ylim([-0.2, 5])

        flux = np.array(med_combined)
        errs = np.array(med_errs)

        norm_sig_file = base_file[0][:-5] +'_NORM_SIG.fits'
        if os.path.exists(norm_sig_file):
            os.remove(norm_sig_file)
        HDU = fits.ImageHDU(data=flux, name='FLUX')
        hdulist = fits.HDUList([fits.PrimaryHDU(), HDU])
        HDU = fits.ImageHDU(data=errs, name='ERRS')
        hdulist.append(HDU)
        HDU = fits.ImageHDU(data=flux, name='QUAL')
        hdulist.append(HDU)
        HDU = fits.ImageHDU(data=spikes, name='SPIKES')
        hdulist.append(HDU)
        HDU = fits.ImageHDU(data=np.array(orig_combined), name='NO MODS')
        hdulist.append(HDU)
        HDU = fits.ImageHDU(data=np.array(norm_combined), name='NORMALIZED ONLY')
        hdulist.append(HDU)

        hdulist.writeto(norm_sig_file)
        hdulist.close()
        print('Wrote normalized and sigma clipped file')
        plt.show()

        if molecfit_ready == True:
            self.make_molefit_ready(base_file[0],norm_sig_file)

    def closest_index(self, arr, val):
        """
        Finds the index of the value closest to that requested in a given array

        :param arr: array
        :param val: value you are searching for
        :return: index in array of number closest to val
        """

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

    def fits_get_wave_data(self, file_path):

        with fits.open(file_path) as hdul:
            data = hdul[0].data
            data = data.tolist()
            CRVAL1 = hdul[0].header['CRVAL1']
            CDELT1 = hdul[0].header['CDELT1']
            NAXIS1 = hdul[0].header['NAXIS1']
            wave = np.array([CRVAL1 + CDELT1 * i for i in range(NAXIS1)]) / 1000.
        return wave, data

    def clip_data(self, wave, data, low_wavelength, high_wavelength):
        low_idx = (np.abs(wave - low_wavelength)).argmin()
        high_idx = (np.abs(wave - high_wavelength)).argmin()
        wave_clip = wave[low_idx:high_idx]
        data_clip = data[low_idx:high_idx]
        return wave_clip, data_clip

    def bin_data(self, wave_clip, data_clip, num_bins):
        bin_width = len(wave_clip) // num_bins
        binned_wave = []
        binned_data = []
        for bin_start in np.arange(0, len(wave_clip), bin_width):
            bin_end = bin_start + bin_width
            bin_wave = wave_clip[bin_start:bin_end]
            bin_data = data_clip[bin_start:bin_end]
            bin_mean = np.median(bin_data)
            binned_wave.append(bin_wave)
            binned_data.append(bin_mean)
        return binned_wave, binned_data

    def mask_and_compress(self, wave, data, mask_ranges):
        masked_wave = wave.copy()
        for m in mask_ranges:
            masked_wave = np.ma.masked_inside(masked_wave, m[0], m[1])
        mask = masked_wave.mask
        masked_data = np.ma.masked_array(data, mask, fill_value=np.nan)
        compressed_wave = masked_wave.compressed()
        compressed_data = masked_data.compressed()
        return compressed_wave, compressed_data

    def edge_matching(self, bonus_plots=False, annotate=True, wave_include=False, dichroic=True,
                      normalize_only=True, **kwargs):

        # UV-VIS Overlap: 0.545284 - 0.555926
        # VIS-NIR Overlap: 0.994165 - 1.01988

        # Searches the planet's directory for the most recent post-molecfit run of each wavelength band
        post_dir = self.work_dir + '/reflex_end_products/'
        post_files = glob.glob(post_dir + '*' + '.fits')

        # Define a dictionary that maps body names to file paths
        file_paths = {
            'titan': {
                #'UVB': '/home/gmansir/Thesis/Titan_tests/reflex_end_products/2023-03-20T15:27:11/MOV_TitanDiskIntegrated_SCI_IFU_MERGE1D_DATA_OBJ_UVB_sum_sum.fits',
                #'VIS': '/home/gmansir/Thesis/Titan_tests/reflex_tmp_products/molecfit/XSHOOTER/molecfit_correct_1/2023-08-10T16:26:35.921/SCIENCE_TELLURIC_CORR_MOV_Uranus_2_SCI_IFU_MERGE1D_DATA_OBJ_VIS_MOLECFIT_READY.fits',
                #'NIR': '/home/gmansir/Thesis/Titan_tests/reflex_tmp_products/molecfit/XSHOOTER/molecfit_correct_1/2023-08-10T16:03:39.798/SCIENCE_TELLURIC_CORR_MOV_Uranus_2_SCI_IFU_MERGE1D_DATA_OBJ_NIR_MOLECFIT_READY.fits',
                #'PUVB': '/home/gmansir/Thesis/Titan_tests/reflex_end_products/2023-03-20T15:27:11/XSHOO.2019-09-26T02:17:50.023_tpl/MOV_Titan_SCI_IFU_MERGE1D_DATA_OBJ_UVB_sum_sum.fits',
                #'PVIS': '/home/gmansir/Thesis/Titan_tests/reflex_end_products/2023-03-20T15:27:11/MOV_Uranus_2_SCI_IFU_MERGE1D_DATA_OBJ_VIS_MOLECFIT_READY.fits',
                #'PNIR': '/home/gmansir/Thesis/Titan_tests/reflex_end_products/2023-03-20T15:27:11/MOV_Uranus_2_SCI_IFU_MERGE1D_DATA_OBJ_NIR_MOLECFIT_READY.fits',
                #'DIUVB': '/home/gmansir/Thesis/Titan_tests/reflex_end_products/2023-03-20T15:27:11/MOV_TitanDiskIntegrated_SCI_IFU_MERGE1D_DATA_OBJ_UVB_sum_sum.fits',
                #'DIVIS': '/home/gmansir/Thesis/Titan_tests/reflex_end_products/2023-03-20T15:27:11/MOV_TitanDiskIntegrated_SCI_IFU_MERGE1D_DATA_OBJ_VIS_MOLECFIT_READY.fits',
                #'DINIR': '/home/gmansir/Thesis/Titan_tests/reflex_end_products/2023-03-20T15:27:11/MOV_TitanDiskIntegrated_SCI_IFU_MERGE1D_DATA_OBJ_NIR_MOLECFIT_READY.fits',
                'UVB': '/home/gmansir/Thesis/Titan/Data/MOV_DiskIntegrated_Titan_SCI_IFU_MERGE1D_DATA_OBJ_UVB_MOLECFIT_READY.fits',
                'VIS': '/home/gmansir/Thesis/Titan/reflex_tmp_products/molecfit/XSHOOTER/molecfit_correct_1/2023-10-05T20:11:31.557/SCIENCE_TELLURIC_CORR_MOV_DiskIntegrated_Titan_SCI_IFU_MERGE1D_DATA_OBJ_VIS_MOLECFIT_READY.fits',
                'NIR': '/home/gmansir/Thesis/Titan/reflex_tmp_products/molecfit/XSHOOTER/molecfit_correct_1/2023-10-06T13:07:19.862/SCIENCE_TELLURIC_CORR_MOV_DiskIntegrated_Titan_SCI_IFU_MERGE1D_DATA_OBJ_NIR_MOLECFIT_READY.fits',
                'PUVB': '/home/gmansir/Thesis/Titan/Data/MOV_Titan_SCI_IFU_MERGE1D_DATA_OBJ_UVB_MOLECFIT_READY.fits',
                'PVIS': '/home/gmansir/Thesis/Titan/Data/MOV_Titan_SCI_IFU_MERGE1D_DATA_OBJ_VIS_MOLECFIT_READY.fits',
                'PNIR': '/home/gmansir/Thesis/Titan/Data/MOV_Titan_SCI_IFU_MERGE1D_DATA_OBJ_NIR_MOLECFIT_READY.fits',
                'DIUVB': '/home/gmansir/Thesis/Titan/Data/MOV_DiskIntegrated_Titan_SCI_IFU_MERGE1D_DATA_OBJ_UVB_MOLECFIT_READY.fits',
                'DIVIS': '/home/gmansir/Thesis/Titan/Data/MOV_DiskIntegrated_Titan_SCI_IFU_MERGE1D_DATA_OBJ_VIS_MOLECFIT_READY.fits',
                'DINIR': '/home/gmansir/Thesis/Titan/Data/MOV_DiskIntegrated_Titan_SCI_IFU_MERGE1D_DATA_OBJ_NIR_MOLECFIT_READY.fits'
            },
            'enceladus': {
                #'UVB': '/home/gmansir/Thesis/Titan_tests/reflex_end_products/2023-03-20T15:27:11/XSHOO.2019-09-26T01:56:56.495_tpl/MOV_Enceladus_SCI_IFU_MERGE1D_DATA_OBJ_UVB_sum_sum.fits',
                #'VIS': '/home/gmansir/Thesis/Enceladus/reflex_tmp_products/molecfit/XSHOOTER_IFU/molecfit_correct_1/2023-06-07T21:18:07.376/SCIENCE_TELLURIC_CORR_MOV_Enceladus_SCI_IFU_MERGE1D_DATA_OBJ_VIS_MOLECFIT_READY.fits',
                #'NIR': '/home/gmansir/Thesis/Enceladus/reflex_tmp_products/molecfit/XSHOOTER_IFU/molecfit_correct_1/2023-06-07T16:12:16.881/SCIENCE_TELLURIC_CORR_MOV_Enceladus_SCI_IFU_MERGE1D_DATA_OBJ_NIR_MOLECFIT_READY.fits',
                #'PUVB': '/home/gmansir/Thesis/Titan_tests/reflex_end_products/2023-03-20T15:27:11/XSHOO.2019-09-26T01:56:56.495_tpl/MOV_Enceladus_SCI_IFU_MERGE1D_DATA_OBJ_UVB_MOLECFIT_READY.fits',
                #'PVIS': '/home/gmansir/Thesis/Titan_tests/reflex_end_products/2023-03-20T15:27:11/XSHOO.2019-09-26T01:57:01.735_tpl/MOV_Enceladus_SCI_IFU_MERGE1D_DATA_OBJ_VIS_MOLECFIT_READY.fits',
                #'PNIR': '/home/gmansir/Thesis/Titan_tests/reflex_end_products/2023-03-20T15:27:11/XSHOO.2019-09-26T01:57:05.003_tpl/MOV_Enceladus_SCI_IFU_MERGE1D_DATA_OBJ_NIR_MOLECFIT_READY.fits',
                #'DIUVB': '/home/gmansir/Thesis/Disk_Integrated_Specs/Enceladus_DiskIntegrated_UVB_O1.fits',
                #'DIVIS': '/home/gmansir/Thesis/Disk_Integrated_Specs/Enceladus_DiskIntegrated_VIS_O1.fits',
                #'DINIR': '/home/gmansir/Thesis/Disk_Integrated_Specs/Enceladus_DiskIntegrated_NIR_O1.fits'
                'UVB': '/home/gmansir/Thesis/Enceladus/Data/PreMolecfit/MOV_DiskIntegrated_Enceladus_SCI_IFU_MERGE1D_DATA_OBJ_UVB_MOLECFIT_READY.fits',
                'VIS': '/home/gmansir/Thesis/Enceladus/reflex_tmp_products/molecfit/XSHOOTER/molecfit_correct_1/2023-09-21T18:57:39.342/SCIENCE_TELLURIC_CORR_MOV_DiskIntegrated_Enceladus_SCI_IFU_MERGE1D_DATA_OBJ_VIS_MOLECFIT_READY.fits',
                'NIR': '/home/gmansir/Thesis/Enceladus/reflex_tmp_products/molecfit/XSHOOTER/molecfit_correct_1/2023-10-06T15:18:40.108/SCIENCE_TELLURIC_CORR_MOV_DiskIntegrated_Enceladus_SCI_IFU_MERGE1D_DATA_OBJ_NIR_MOLECFIT_READY.fits',
                'PUVB': '/home/gmansir/Thesis/Enceladus/Data/PreMolecfit/MOV_Enceladus_SCI_IFU_MERGE1D_DATA_OBJ_UVB_MOLECFIT_READY.fits',
                'PVIS': '/home/gmansir/Thesis/Enceladus/Data/PreMolecfit/MOV_Enceladus_SCI_IFU_MERGE1D_DATA_OBJ_VIS_MOLECFIT_READY.fits',
                'PNIR': '/home/gmansir/Thesis/Enceladus/Data/PreMolecfit/MOV_Enceladus_SCI_IFU_MERGE1D_DATA_OBJ_NIR_MOLECFIT_READY.fits',
                'DIUVB': '/home/gmansir/Thesis/Enceladus/Data/PreMolecfit/MOV_DiskIntegrated_Enceladus_SCI_IFU_MERGE1D_DATA_OBJ_UVB_MOLECFIT_READY.fits',
                'DIVIS': '/home/gmansir/Thesis/Enceladus/Data/PreMolecfit/MOV_DiskIntegrated_Enceladus_SCI_IFU_MERGE1D_DATA_OBJ_VIS_MOLECFIT_READY.fits',
                'DINIR': '/home/gmansir/Thesis/Enceladus/Data/PreMolecfit/MOV_DiskIntegrated_Enceladus_SCI_IFU_MERGE1D_DATA_OBJ_NIR_MOLECFIT_READY.fits'

        },
            'neptune1': {
                #'UVB': '/home/gmansir/Thesis/Neptune_old/Data/PreMolecfit/uvb/MOV_Neptune_SCI_IFU_MERGE1D_DATA_OBJ_UVB_sum_sum_3.fits',
                #'VIS': '/home/gmansir/Thesis/Neptune/reflex_tmp_products/molecfit/XSHOOTER_IFU/molecfit_correct_1/2023-06-08T18:25:51.420/SCIENCE_TELLURIC_CORR_MOV_Neptune_SCI_IFU_MERGE1D_DATA_OBJ_VIS_MOLECFIT_READY.fits',
                #'NIR': '/home/gmansir/Thesis/Neptune/reflex_tmp_products/molecfit/XSHOOTER_IFU/molecfit_correct_1/2023-06-08T17:19:03.838/SCIENCE_TELLURIC_CORR_MOV_Neptune_SCI_IFU_MERGE1D_DATA_OBJ_NIR_MOLECFIT_READY.fits',
                #'PUVB': '/home/gmansir/Thesis/Neptune_old/Data/PreMolecfit/uvb/MOV_Neptune_SCI_IFU_MERGE1D_DATA_OBJ_UVB_sum_sum_3.fits',
                #'PVIS': '/home/gmansir/Thesis/Neptune/reflex_end_products/2023-05-23T15:11:05/XSHOO.2019-08-30T04:48:43.725_tpl/MOV_Neptune_SCI_IFU_MERGE1D_DATA_OBJ_VIS_MOLECFIT_READY.fits',
                #'PNIR': '/home/gmansir/Thesis/Neptune/reflex_end_products/2023-05-23T15:11:05/XSHOO.2019-08-30T04:48:46.519_tpl/MOV_Neptune_SCI_IFU_MERGE1D_DATA_OBJ_NIR_MOLECFIT_READY.fits',
                #'DIUVB': '/home/gmansir/Thesis/Disk_Integrated_Specs/_Neptune_DiskIntegrated_UVB_O1.fits',
                #'DIVIS': '/home/gmansir/Thesis/Disk_Integrated_Specs/_Neptune_DiskIntegrated_VIS_O1.fits',
                #'DINIR': '/home/gmansir/Thesis/Disk_Integrated_Specs/_Neptune_DiskIntegrated_NIR_O1.fits'
                'UVB': '/home/gmansir/Thesis/Neptune/Data/PreMolecfit/Group1/MOV_DiskIntegrated_Neptune_1_SCI_IFU_MERGE1D_DATA_OBJ_UVB_MOLECFIT_READY.fits',
                'VIS': '/home/gmansir/Thesis/Neptune/reflex_tmp_products/molecfit/XSHOOTER/molecfit_correct_1/2023-09-22T11:17:36.769/SCIENCE_TELLURIC_CORR_MOV_DiskIntegrated_Neptune_1_SCI_IFU_MERGE1D_DATA_OBJ_VIS_MOLECFIT_READY.fits',
                'NIR': '/home/gmansir/Thesis/Neptune1/reflex_tmp_products/molecfit/XSHOOTER/molecfit_correct_1/2023-10-04T12:14:12.141/SCIENCE_TELLURIC_CORR_MOV_DiskIntegrated_Neptune_1_SCI_IFU_MERGE1D_DATA_OBJ_NIR_MOLECFIT_READY.fits',
                'PUVB': '/home/gmansir/Thesis/Neptune/Data/PreMolecfit/Group1/MOV_Neptune_1_SCI_IFU_MERGE1D_DATA_OBJ_UVB_MOLECFIT_READY.fits',
                'PVIS': '/home/gmansir/Thesis/Neptune/Data/PreMolecfit/Group1/MOV_Neptune_1_SCI_IFU_MERGE1D_DATA_OBJ_VIS_MOLECFIT_READY.fits',
                'PNIR': '/home/gmansir/Thesis/Neptune/Data/PreMolecfit/Group1/MOV_Neptune_1_SCI_IFU_MERGE1D_DATA_OBJ_NIR_MOLECFIT_READY.fits',
                'DIUVB': '/home/gmansir/Thesis/Neptune/Data/PreMolecfit/Group1/MOV_DiskIntegrated_Neptune_1_SCI_IFU_MERGE1D_DATA_OBJ_UVB_MOLECFIT_READY.fits',
                'DIVIS': '/home/gmansir/Thesis/Neptune/Data/PreMolecfit/Group1/MOV_DiskIntegrated_Neptune_1_SCI_IFU_MERGE1D_DATA_OBJ_VIS_MOLECFIT_READY.fits',
                'DINIR': '/home/gmansir/Thesis/Neptune/Data/PreMolecfit/Group1/MOV_DiskIntegrated_Neptune_1_SCI_IFU_MERGE1D_DATA_OBJ_NIR_MOLECFIT_READY.fits'
        },
            'neptune2': {
                'UVB': '/home/gmansir/Thesis/Neptune/Data/PreMolecfit/Group2/MOV_DiskIntegrated_Neptune_2_SCI_IFU_MERGE1D_DATA_OBJ_UVB_MOLECFIT_READY.fits',
                'VIS': '/home/gmansir/Thesis/Neptune1/reflex_tmp_products/molecfit/XSHOOTER/molecfit_correct_1/2023-10-05T18:53:22.183/SCIENCE_TELLURIC_CORR_MOV_DiskIntegrated_Neptune_2_SCI_IFU_MERGE1D_DATA_OBJ_VIS_MOLECFIT_READY.fits',
                'NIR': '/home/gmansir/Thesis/Neptune1/reflex_tmp_products/molecfit/XSHOOTER/molecfit_correct_1/2023-10-05T19:26:57.814/SCIENCE_TELLURIC_CORR_MOV_DiskIntegrated_Neptune_2_SCI_IFU_MERGE1D_DATA_OBJ_NIR_MOLECFIT_READY.fits',
                'PUVB': '/home/gmansir/Thesis/Neptune/Data/PreMolecfit/Group2/MOV_Neptune_2_SCI_IFU_MERGE1D_DATA_OBJ_UVB_MOLECFIT_READY.fits',
                'PVIS': '/home/gmansir/Thesis/Neptune/Data/PreMolecfit/Group2/MOV_Neptune_2_SCI_IFU_MERGE1D_DATA_OBJ_VIS_MOLECFIT_READY.fits',
                'PNIR': '/home/gmansir/Thesis/Neptune/Data/PreMolecfit/Group2/MOV_Neptune_2_SCI_IFU_MERGE1D_DATA_OBJ_NIR_MOLECFIT_READY.fits',
                'DIUVB': '/home/gmansir/Thesis/Neptune/Data/PreMolecfit/Group2/MOV_DiskIntegrated_Neptune_2_SCI_IFU_MERGE1D_DATA_OBJ_UVB_MOLECFIT_READY.fits',
                'DIVIS': '/home/gmansir/Thesis/Neptune/Data/PreMolecfit/Group2/MOV_DiskIntegrated_Neptune_2_SCI_IFU_MERGE1D_DATA_OBJ_VIS_MOLECFIT_READY.fits',
                'DINIR': '/home/gmansir/Thesis/Neptune/Data/PreMolecfit/Group2/MOV_DiskIntegrated_Neptune_2_SCI_IFU_MERGE1D_DATA_OBJ_NIR_MOLECFIT_READY.fits'
        },
            'uranus1': {
                #'UVB': '/home/gmansir/Thesis/Uranus/reflex_end_products/2023-05-29T17:27:12/XSHOO.2021-09-28T06:25:55.623_tpl/MOV_Uranus_2_SCI_IFU_MERGE1D_DATA_OBJ_UVB_sum_sum.fits',
                #'VIS': '/home/gmansir/Thesis/Uranus/reflex_tmp_products/molecfit/XSHOOTER/molecfit_correct_1/2023-08-18T14:21:12.421/SCIENCE_TELLURIC_CORR_MOV_Uranus_2_SCI_IFU_MERGE1D_DATA_OBJ_VIS_MOLECFIT_READY.fits',
                #'NIR': '/home/gmansir/Thesis/Telluric/reflex_tmp_products/molecfit/XSHOOTER/molecfit_correct_1/2023-08-08T17:24:19.851/SCIENCE_TELLURIC_CORR_MOV_Uranus_2_SCI_IFU_MERGE1D_DATA_OBJ_NIR_MOLECFIT_READY.fits',
                'UVB': '/home/gmansir/Thesis/Uranus1/reflex_end_products/2023-09-11T15:33:27/Group1/MOV_Disk_Integrated_Uranus_1_SCI_IFU_MERGE1D_DATA_OBJ_UVB_MOLECFIT_READY.fits',
                'VIS': '/home/gmansir/Thesis/Uranus1/reflex_tmp_products/molecfit/XSHOOTER/molecfit_correct_1/2023-09-15T16:33:56.474/SCIENCE_TELLURIC_CORR_MOV_Disk_Integrated_Uranus_1_SCI_IFU_MERGE1D_DATA_OBJ_VIS_MOLECFIT_READY.fits',
                'NIR': '/home/gmansir/Thesis/Uranus1/reflex_tmp_products/molecfit/XSHOOTER/molecfit_correct_1/2023-09-15T17:34:58.594/SCIENCE_TELLURIC_CORR_MOV_Disk_Integrated_Uranus_1_SCI_IFU_MERGE1D_DATA_OBJ_NIR_MOLECFIT_READY.fits',
                'PUVB': '/home/gmansir/Thesis/Uranus1/reflex_end_products/2023-09-11T15:33:27/Group1/XSHOO.2021-09-28T06:25:55.623/MOV_Uranus_2_SCI_IFU_MERGE1D_DATA_OBJ_UVB_MOLECFIT_READY.fits',
                'PVIS': '/home/gmansir/Thesis/Uranus1/reflex_end_products/2023-09-11T15:33:27/Group1/XSHOO.2021-09-28T06:26:00.854/MOV_Uranus_2_SCI_IFU_MERGE1D_DATA_OBJ_VIS_MOLECFIT_READY.fits',
                'PNIR': '/home/gmansir/Thesis/Uranus1/reflex_end_products/2023-09-11T15:33:27/Group1/XSHOO.2021-09-28T06:26:03.970/MOV_Uranus_2_SCI_IFU_MERGE1D_DATA_OBJ_NIR_MOLECFIT_READY.fits',
                'DIUVB': '/home/gmansir/Thesis/Uranus1/reflex_end_products/2023-09-11T15:33:27/Group1/MOV_Disk_Integrated_Uranus_1_SCI_IFU_MERGE1D_DATA_OBJ_UVB_MOLECFIT_READY.fits',
                'DIVIS': '/home/gmansir/Thesis/Uranus1/reflex_end_products/2023-09-11T15:33:27/Group1/MOV_Disk_Integrated_Uranus_1_SCI_IFU_MERGE1D_DATA_OBJ_VIS_MOLECFIT_READY.fits',
                'DINIR': '/home/gmansir/Thesis/Uranus1/reflex_end_products/2023-09-11T15:33:27/Group1/MOV_Disk_Integrated_Uranus_1_SCI_IFU_MERGE1D_DATA_OBJ_NIR_MOLECFIT_READY.fits'
            },
            'uranus2': {
                'UVB': '/home/gmansir/Thesis/Uranus1/reflex_end_products/2023-09-11T15:33:27/Group2/MOV_Disk_Integrated_Uranus_2_SCI_IFU_MERGE1D_DATA_OBJ_UVB_MOLECFIT_READY.fits',
                'VIS': '/home/gmansir/Thesis/Uranus1/reflex_end_products/molecfit/XSHOOTER/2023-09-16T20:21:03/MOV_Disk_Integrated_Uranus_2_SCI_IFU_MERGE1D_DATA_OBJ_VIS_MOLECFIT_READY/MOV_Uranus_2_SCIENCE_TELLURIC_CORR.fits',
                'NIR': '/home/gmansir/Thesis/Uranus1/reflex_end_products/molecfit/XSHOOTER/2023-09-16T20:21:03/MOV_Disk_Integrated_Uranus_2_SCI_IFU_MERGE1D_DATA_OBJ_NIR_MOLECFIT_READY/MOV_Uranus_2_SCIENCE_TELLURIC_CORR.fits',
                'PUVB': '/home/gmansir/Thesis/Uranus1/reflex_end_products/2023-09-11T15:33:27/Group2/XSHOO.2021-09-28T06:28:43.599/MOV_Uranus_2_SCI_IFU_MERGE1D_DATA_OBJ_UVB_MOLECFIT_READY.fits',
                'PVIS': '/home/gmansir/Thesis/Uranus1/reflex_end_products/2023-09-11T15:33:27/Group2/XSHOO.2021-09-28T06:28:00.595/MOV_Uranus_2_SCI_IFU_MERGE1D_DATA_OBJ_VIS_MOLECFIT_READY.fits',
                'PNIR': '/home/gmansir/Thesis/Uranus1/reflex_end_products/2023-09-11T15:33:27/Group2/XSHOO.2021-09-28T06:28:47.609/MOV_Uranus_2_SCI_IFU_MERGE1D_DATA_OBJ_NIR_MOLECFIT_READY.fits',
                'DIUVB': '/home/gmansir/Thesis/Uranus1/reflex_end_products/2023-09-11T15:33:27/Group2/MOV_Disk_Integrated_Uranus_2_SCI_IFU_MERGE1D_DATA_OBJ_UVB_MOLECFIT_READY.fits',
                'DIVIS': '/home/gmansir/Thesis/Uranus1/reflex_end_products/2023-09-11T15:33:27/Group2/MOV_Disk_Integrated_Uranus_2_SCI_IFU_MERGE1D_DATA_OBJ_VIS_MOLECFIT_READY.fits',
                'DINIR': '/home/gmansir/Thesis/Uranus1/reflex_end_products/2023-09-11T15:33:27/Group2/MOV_Disk_Integrated_Uranus_2_SCI_IFU_MERGE1D_DATA_OBJ_NIR_MOLECFIT_READY.fits'
            },
            'uranus3': {
                'UVB': '/home/gmansir/Thesis/Uranus1/reflex_end_products/2023-09-11T15:33:27/Group3/MOV_Disk_Integrated_Uranus_3_SCI_IFU_MERGE1D_DATA_OBJ_UVB_MOLECFIT_READY.fits',
                'VIS': '/home/gmansir/Thesis/Uranus1/reflex_tmp_products/molecfit/XSHOOTER/molecfit_correct_1/2023-09-20T14:36:06.294/SCIENCE_TELLURIC_CORR_MOV_Disk_Integrated_Uranus_3_SCI_IFU_MERGE1D_DATA_OBJ_VIS_MOLECFIT_READY.fits',
                'NIR': '/home/gmansir/Thesis/Uranus1/reflex_tmp_products/molecfit/XSHOOTER/molecfit_correct_1/2023-09-20T14:21:03.831/SCIENCE_TELLURIC_CORR_MOV_Disk_Integrated_Uranus_3_SCI_IFU_MERGE1D_DATA_OBJ_NIR_MOLECFIT_READY.fits',
                'PUVB': '/home/gmansir/Thesis/Uranus1/reflex_end_products/2023-09-11T15:33:27/Group3/XSHOO.2021-09-28T06:31:41.244/MOV_Uranus_2_SCI_IFU_MERGE1D_DATA_OBJ_UVB_MOLECFIT_READY.fits',
                'PVIS': '/home/gmansir/Thesis/Uranus1/reflex_end_products/2023-09-11T15:33:27/Group3/XSHOO.2021-09-28T06:31:46.475/MOV_Uranus_2_SCI_IFU_MERGE1D_DATA_OBJ_VIS_MOLECFIT_READY.fits',
                'PNIR': '/home/gmansir/Thesis/Uranus1/reflex_end_products/2023-09-11T15:33:27/Group3/XSHOO.2021-09-28T06:31:49.249/MOV_Uranus_2_SCI_IFU_MERGE1D_DATA_OBJ_NIR_MOLECFIT_READY.fits',
                'DIUVB': '/home/gmansir/Thesis/Uranus1/reflex_end_products/2023-09-11T15:33:27/Group3/MOV_Disk_Integrated_Uranus_3_SCI_IFU_MERGE1D_DATA_OBJ_UVB_MOLECFIT_READY.fits',
                'DIVIS': '/home/gmansir/Thesis/Uranus1/reflex_end_products/2023-09-11T15:33:27/Group3/MOV_Disk_Integrated_Uranus_3_SCI_IFU_MERGE1D_DATA_OBJ_VIS_MOLECFIT_READY.fits',
                'DINIR': '/home/gmansir/Thesis/Uranus1/reflex_end_products/2023-09-11T15:33:27/Group3/MOV_Disk_Integrated_Uranus_3_SCI_IFU_MERGE1D_DATA_OBJ_NIR_MOLECFIT_READY.fits'
            },
            'uranus4': {
                'UVB': '/home/gmansir/Thesis/Uranus1/reflex_end_products/2023-09-11T15:33:27/Group4/MOV_Disk_Integrated_Uranus_4_SCI_IFU_MERGE1D_DATA_OBJ_UVB_MOLECFIT_READY.fits',
                'VIS': '/home/gmansir/Thesis/Uranus1/reflex_tmp_products/molecfit/XSHOOTER/molecfit_correct_1/2023-09-20T13:46:04.232/SCIENCE_TELLURIC_CORR_MOV_Disk_Integrated_Uranus_4_SCI_IFU_MERGE1D_DATA_OBJ_VIS_MOLECFIT_READY.fits',
                'NIR': '/home/gmansir/Thesis/Uranus1/reflex_tmp_products/molecfit/XSHOOTER/molecfit_correct_1/2023-09-20T13:24:32.809/SCIENCE_TELLURIC_CORR_MOV_Disk_Integrated_Uranus_4_SCI_IFU_MERGE1D_DATA_OBJ_NIR_MOLECFIT_READY.fits',
                'PUVB': '/home/gmansir/Thesis/Uranus1/reflex_end_products/2023-09-11T15:33:27/Group4/XSHOO.2021-09-28T06:34:28.569/MOV_Uranus_2_SCI_IFU_MERGE1D_DATA_OBJ_UVB_MOLECFIT_READY.fits',
                'PVIS': '/home/gmansir/Thesis/Uranus1/reflex_end_products/2023-09-11T15:33:27/Group4/XSHOO.2021-09-28T06:33:45.495/MOV_Uranus_2_SCI_IFU_MERGE1D_DATA_OBJ_VIS_MOLECFIT_READY.fits',
                'PNIR': '/home/gmansir/Thesis/Uranus1/reflex_end_products/2023-09-11T15:33:27/Group4/XSHOO.2021-09-28T06:34:32.223/MOV_Uranus_2_SCI_IFU_MERGE1D_DATA_OBJ_NIR_MOLECFIT_READY.fits',
                'DIUVB': '/home/gmansir/Thesis/Uranus1/reflex_end_products/2023-09-11T15:33:27/Group4/MOV_Disk_Integrated_Uranus_4_SCI_IFU_MERGE1D_DATA_OBJ_UVB_MOLECFIT_READY.fits',
                'DIVIS': '/home/gmansir/Thesis/Uranus1/reflex_end_products/2023-09-11T15:33:27/Group4/MOV_Disk_Integrated_Uranus_4_SCI_IFU_MERGE1D_DATA_OBJ_VIS_MOLECFIT_READY.fits',
                'DINIR': '/home/gmansir/Thesis/Uranus1/reflex_end_products/2023-09-11T15:33:27/Group4/MOV_Disk_Integrated_Uranus_4_SCI_IFU_MERGE1D_DATA_OBJ_NIR_MOLECFIT_READY.fits'
            },
            'saturn1': {
                #'UVB': '/home/gmansir/Thesis/Saturn/reflex_end_products/2023-05-30T15:26:37/XSHOO.2019-09-26T01:01:18.794_tpl/MOV_Saturn_SCI_IFU_MERGE1D_DATA_OBJ_UVB_sum_sum.fits',
                #'VIS': '/home/gmansir/Thesis/Saturn/reflex_tmp_products/molecfit/XSHOOTER_IFU/molecfit_correct_1/2023-06-01T16:19:40.288/SCIENCE_TELLURIC_CORR_MOV_Saturn_SCI_IFU_MERGE1D_DATA_OBJ_VIS_MOLECFIT_READY.fits',
                #'NIR': '/home/gmansir/Thesis/Saturn/reflex_tmp_products/molecfit/XSHOOTER_IFU/molecfit_correct_1/2023-06-06T20:34:25.589/SCIENCE_TELLURIC_CORR_MOV_Saturn_SCI_IFU_MERGE1D_DATA_OBJ_NIR_MOLECFIT_READY.fits'
                'UVB': '/home/gmansir/Thesis/Saturn1/Data/PreMolecfit/MOV_DiskIntegrated_Saturn_1_SCI_IFU_MERGE1D_DATA_OBJ_UVB_MOLECFIT_READY.fits',
                'VIS': '/home/gmansir/Thesis/Saturn1/reflex_tmp_products/molecfit/XSHOOTER/molecfit_correct_1/2023-10-12T16:55:57.892/SCIENCE_TELLURIC_CORR_MOV_DiskIntegrated_Saturn_1_SCI_IFU_MERGE1D_DATA_OBJ_VIS_MOLECFIT_READY.fits',
                'NIR': '/home/gmansir/Thesis/Saturn1/reflex_tmp_products/molecfit/XSHOOTER/molecfit_correct_1/2023-10-12T16:31:27.165/SCIENCE_TELLURIC_CORR_MOV_DiskIntegrated_Saturn_1_SCI_IFU_MERGE1D_DATA_OBJ_NIR_MOLECFIT_READY.fits',
                'PUVB': '/home/gmansir/Thesis/Saturn1/Data/PreMolecfit/MOV_Saturn_SCI_IFU_MERGE1D_DATA_OBJ_UVB_MOLECFIT_READY.fits',
                'PVIS': '/home/gmansir/Thesis/Saturn1/Data/PreMolecfit/MOV_Saturn_SCI_IFU_MERGE1D_DATA_OBJ_VIS_MOLECFIT_READY.fits',
                'PNIR': '/home/gmansir/Thesis/Saturn1/Data/PreMolecfit/MOV_Saturn_SCI_IFU_MERGE1D_DATA_OBJ_NIR_MOLECFIT_READY.fits',
                'DIUVB': '/home/gmansir/Thesis/Saturn1/Data/PreMolecfit/MOV_DiskIntegrated_Saturn_1_SCI_IFU_MERGE1D_DATA_OBJ_UVB_MOLECFIT_READY.fits',
                'DIVIS': '/home/gmansir/Thesis/Saturn1/Data/PreMolecfit/MOV_DiskIntegrated_Saturn_1_SCI_IFU_MERGE1D_DATA_OBJ_VIS_MOLECFIT_READY.fits',
                'DINIR': '/home/gmansir/Thesis/Saturn1/Data/PreMolecfit/MOV_DiskIntegrated_Saturn_1_SCI_IFU_MERGE1D_DATA_OBJ_NIR_MOLECFIT_READY.fits'
            },
            'saturn2': {
                'UVB': '/home/gmansir/Thesis/Saturn2/Data/PreMolecfit/MOV_DiskIntegrated_Saturn_2_SCI_IFU_MERGE1D_DATA_OBJ_UVB_MOLECFIT_READY.fits',
                'VIS': '/home/gmansir/Thesis/Saturn2/reflex_tmp_products/molecfit/XSHOOTER/molecfit_correct_1/2023-10-12T19:47:44.623/SCIENCE_TELLURIC_CORR_MOV_DiskIntegrated_Saturn_2_SCI_IFU_MERGE1D_DATA_OBJ_VIS_MOLECFIT_READY.fits',
                'NIR': '/home/gmansir/Thesis/Saturn2/reflex_tmp_products/molecfit/XSHOOTER/molecfit_correct_1/2023-10-12T19:11:41.387/SCIENCE_TELLURIC_CORR_MOV_DiskIntegrated_Saturn_2_SCI_IFU_MERGE1D_DATA_OBJ_NIR_MOLECFIT_READY.fits',
                'PUVB': '/home/gmansir/Thesis/Saturn2/Data/PreMolecfit/MOV_Saturn_SCI_IFU_MERGE1D_DATA_OBJ_UVB_MOLECFIT_READY.fits',
                'PVIS': '/home/gmansir/Thesis/Saturn2/Data/PreMolecfit/MOV_Saturn_SCI_IFU_MERGE1D_DATA_OBJ_VIS_MOLECFIT_READY.fits',
                'PNIR': '/home/gmansir/Thesis/Saturn2/Data/PreMolecfit/MOV_Saturn_SCI_IFU_MERGE1D_DATA_OBJ_NIR_MOLECFIT_READY.fits',
                'DIUVB': '/home/gmansir/Thesis/Saturn2/Data/PreMolecfit/MOV_DiskIntegrated_Saturn_2_SCI_IFU_MERGE1D_DATA_OBJ_UVB_MOLECFIT_READY.fits',
                'DIVIS': '/home/gmansir/Thesis/Saturn2/Data/PreMolecfit/MOV_DiskIntegrated_Saturn_2_SCI_IFU_MERGE1D_DATA_OBJ_VIS_MOLECFIT_READY.fits',
                'DINIR': '/home/gmansir/Thesis/Saturn2/Data/PreMolecfit/MOV_DiskIntegrated_Saturn_2_SCI_IFU_MERGE1D_DATA_OBJ_NIR_MOLECFIT_READY.fits'
            },
            'saturn3': {
                'UVB': '/home/gmansir/Thesis/Saturn3/Data/PreMolecfit/MOV_DiskIntegrated_Saturn_3_SCI_IFU_MERGE1D_DATA_OBJ_UVB_MOLECFIT_READY.fits',
                'VIS': '/home/gmansir/Thesis/Saturn/reflex_tmp_products/molecfit/XSHOOTER/molecfit_correct_1/2023-09-25T16:52:42.066/SCIENCE_TELLURIC_CORR_MOV_DiskIntegrated_Saturn_3_SCI_IFU_MERGE1D_DATA_OBJ_VIS_MOLECFIT_READY.fits',
                'NIR': '/home/gmansir/Thesis/Saturn/reflex_tmp_products/molecfit/XSHOOTER/molecfit_correct_1/2023-09-25T16:28:26.734/SCIENCE_TELLURIC_CORR_MOV_DiskIntegrated_Saturn_3_SCI_IFU_MERGE1D_DATA_OBJ_NIR_MOLECFIT_READY.fits',
                'PUVB': '/home/gmansir/Thesis/Saturn3/Data/PreMolecfit/MOV_Saturn_SCI_IFU_MERGE1D_DATA_OBJ_UVB_MOLECFIT_READY.fits',
                'PVIS': '/home/gmansir/Thesis/Saturn3/Data/PreMolecfit/MOV_Saturn_SCI_IFU_MERGE1D_DATA_OBJ_VIS_MOLECFIT_READY.fits',
                'PNIR': '/home/gmansir/Thesis/Saturn3/Data/PreMolecfit/MOV_Saturn_SCI_IFU_MERGE1D_DATA_OBJ_NIR_MOLECFIT_READY.fits',
                'DIUVB': '/home/gmansir/Thesis/Saturn3/Data/PreMolecfit/MOV_DiskIntegrated_Saturn_3_SCI_IFU_MERGE1D_DATA_OBJ_UVB_MOLECFIT_READY.fits',
                'DIVIS': '/home/gmansir/Thesis/Saturn3/Data/PreMolecfit/MOV_DiskIntegrated_Saturn_3_SCI_IFU_MERGE1D_DATA_OBJ_VIS_MOLECFIT_READY.fits',
                'DINIR': '/home/gmansir/Thesis/Saturn3/Data/PreMolecfit/MOV_DiskIntegrated_Saturn_3_SCI_IFU_MERGE1D_DATA_OBJ_NIR_MOLECFIT_READY.fits'
            },
            'saturn4': {
                'UVB': '/home/gmansir/Thesis/Saturn4/Data/PreMolecfit/MOV_DiskIntegrated_Saturn_4_SCI_IFU_MERGE1D_DATA_OBJ_UVB_MOLECFIT_READY.fits',
                'VIS': '/home/gmansir/Thesis/Saturn4/reflex_tmp_products/molecfit/XSHOOTER/molecfit_correct_1/2023-10-12T14:24:56.084/SCIENCE_TELLURIC_CORR_MOV_DiskIntegrated_Saturn_4_SCI_IFU_MERGE1D_DATA_OBJ_VIS_MOLECFIT_READY.fits',
                'NIR': '/home/gmansir/Thesis/Saturn4/reflex_tmp_products/molecfit/XSHOOTER/molecfit_correct_1/2023-10-12T13:52:04.985/SCIENCE_TELLURIC_CORR_MOV_DiskIntegrated_Saturn_4_SCI_IFU_MERGE1D_DATA_OBJ_NIR_MOLECFIT_READY.fits',
                'PUVB': '/home/gmansir/Thesis/Saturn4/Data/PreMolecfit/MOV_Saturn_SCI_IFU_MERGE1D_DATA_OBJ_UVB_MOLECFIT_READY.fits',
                'PVIS': '/home/gmansir/Thesis/Saturn4/Data/PreMolecfit/MOV_Saturn_SCI_IFU_MERGE1D_DATA_OBJ_VIS_MOLECFIT_READY.fits',
                'PNIR': '/home/gmansir/Thesis/Saturn4/Data/PreMolecfit/MOV_Saturn_SCI_IFU_MERGE1D_DATA_OBJ_NIR_MOLECFIT_READY.fits',
                'DIUVB': '/home/gmansir/Thesis/Saturn4/Data/PreMolecfit/MOV_DiskIntegrated_Saturn_4_SCI_IFU_MERGE1D_DATA_OBJ_UVB_MOLECFIT_READY.fits',
                'DIVIS': '/home/gmansir/Thesis/Saturn4/Data/PreMolecfit/MOV_DiskIntegrated_Saturn_4_SCI_IFU_MERGE1D_DATA_OBJ_VIS_MOLECFIT_READY.fits',
                'DINIR': '/home/gmansir/Thesis/Saturn4/Data/PreMolecfit/MOV_DiskIntegrated_Saturn_4_SCI_IFU_MERGE1D_DATA_OBJ_NIR_MOLECFIT_READY.fits'
            },
            'saturn5': {
                'UVB': '/home/gmansir/Thesis/Saturn5/Data/PreMolecfit/MOV_DiskIntegrated_Saturn_5_SCI_IFU_MERGE1D_DATA_OBJ_UVB_MOLECFIT_READY.fits',
                'VIS': '/home/gmansir/Thesis/Saturn/reflex_tmp_products/molecfit/XSHOOTER/molecfit_correct_1/2023-09-25T19:01:06.016/SCIENCE_TELLURIC_CORR_MOV_DiskIntegrated_Saturn_5_SCI_IFU_MERGE1D_DATA_OBJ_VIS_MOLECFIT_READY.fits',
                'NIR': '/home/gmansir/Thesis/Saturn/reflex_tmp_products/molecfit/XSHOOTER/molecfit_correct_1/2023-09-25T18:47:09.771/SCIENCE_TELLURIC_CORR_MOV_DiskIntegrated_Saturn_5_SCI_IFU_MERGE1D_DATA_OBJ_NIR_MOLECFIT_READY.fits',
                'PUVB': '/home/gmansir/Thesis/Saturn5/Data/PreMolecfit/MOV_Saturn_SCI_IFU_MERGE1D_DATA_OBJ_UVB_MOLECFIT_READY.fits',
                'PVIS': '/home/gmansir/Thesis/Saturn5/Data/PreMolecfit/MOV_Saturn_SCI_IFU_MERGE1D_DATA_OBJ_VIS_MOLECFIT_READY.fits',
                'PNIR': '/home/gmansir/Thesis/Saturn5/Data/PreMolecfit/MOV_Saturn_SCI_IFU_MERGE1D_DATA_OBJ_NIR_MOLECFIT_READY.fits',
                'DIUVB': '/home/gmansir/Thesis/Saturn5/Data/PreMolecfit/MOV_DiskIntegrated_Saturn_5_SCI_IFU_MERGE1D_DATA_OBJ_UVB_MOLECFIT_READY.fits',
                'DIVIS': '/home/gmansir/Thesis/Saturn5/Data/PreMolecfit/MOV_DiskIntegrated_Saturn_5_SCI_IFU_MERGE1D_DATA_OBJ_VIS_MOLECFIT_READY.fits',
                'DINIR': '/home/gmansir/Thesis/Saturn5/Data/PreMolecfit/MOV_DiskIntegrated_Saturn_5_SCI_IFU_MERGE1D_DATA_OBJ_NIR_MOLECFIT_READY.fits'
            },
            'saturn6': {
                'UVB': '/home/gmansir/Thesis/Saturn6/Data/PreMolecfit/MOV_DiskIntegrated_Saturn_6_SCI_IFU_MERGE1D_DATA_OBJ_UVB_MOLECFIT_READY.fits',
                'VIS': '/home/gmansir/Thesis/Saturn6/reflex_tmp_products/molecfit/XSHOOTER/molecfit_correct_1/2023-10-11T17:29:54.151/SCIENCE_TELLURIC_CORR_MOV_DiskIntegrated_Saturn_6_SCI_IFU_MERGE1D_DATA_OBJ_VIS_MOLECFIT_READY.fits',
                'NIR': '/home/gmansir/Thesis/Saturn6/reflex_tmp_products/molecfit/XSHOOTER/molecfit_correct_1/2023-10-11T17:15:35.700/SCIENCE_TELLURIC_CORR_MOV_DiskIntegrated_Saturn_6_SCI_IFU_MERGE1D_DATA_OBJ_NIR_MOLECFIT_READY.fits',
                'PUVB': '/home/gmansir/Thesis/Saturn6/Data/PreMolecfit/MOV_Saturn_SCI_IFU_MERGE1D_DATA_OBJ_UVB_MOLECFIT_READY.fits',
                'PVIS': '/home/gmansir/Thesis/Saturn6/Data/PreMolecfit/MOV_Saturn_SCI_IFU_MERGE1D_DATA_OBJ_VIS_MOLECFIT_READY.fits',
                'PNIR': '/home/gmansir/Thesis/Saturn6/Data/PreMolecfit/MOV_Saturn_SCI_IFU_MERGE1D_DATA_OBJ_NIR_MOLECFIT_READY.fits',
                'DIUVB': '/home/gmansir/Thesis/Saturn6/Data/PreMolecfit/MOV_DiskIntegrated_Saturn_6_SCI_IFU_MERGE1D_DATA_OBJ_UVB_MOLECFIT_READY.fits',
                'DIVIS': '/home/gmansir/Thesis/Saturn6/Data/PreMolecfit/MOV_DiskIntegrated_Saturn_6_SCI_IFU_MERGE1D_DATA_OBJ_VIS_MOLECFIT_READY.fits',
                'DINIR': '/home/gmansir/Thesis/Saturn6/Data/PreMolecfit/MOV_DiskIntegrated_Saturn_6_SCI_IFU_MERGE1D_DATA_OBJ_NIR_MOLECFIT_READY.fits'
            },
            'saturn7': {
                'UVB': '/home/gmansir/Thesis/Saturn7/Data/PreMolecfit/MOV_DiskIntegrated_Saturn_7_SCI_IFU_MERGE1D_DATA_OBJ_UVB_MOLECFIT_READY.fits',
                'VIS': '/home/gmansir/Thesis/Saturn7/reflex_tmp_products/molecfit/XSHOOTER/molecfit_correct_1/2023-10-11T01:55:31.106/SCIENCE_TELLURIC_CORR_MOV_DiskIntegrated_Saturn_7_SCI_IFU_MERGE1D_DATA_OBJ_VIS_MOLECFIT_READY.fits',
                'NIR': '/home/gmansir/Thesis/Saturn7/reflex_tmp_products/molecfit/XSHOOTER/molecfit_correct_1/2023-10-11T01:17:53.612/SCIENCE_TELLURIC_CORR_MOV_DiskIntegrated_Saturn_7_SCI_IFU_MERGE1D_DATA_OBJ_NIR_MOLECFIT_READY.fits',
                'PUVB': '/home/gmansir/Thesis/Saturn7/Data/PreMolecfit/MOV_Saturn_SCI_IFU_MERGE1D_DATA_OBJ_UVB_MOLECFIT_READY.fits',
                'PVIS': '/home/gmansir/Thesis/Saturn7/Data/PreMolecfit/MOV_Saturn_SCI_IFU_MERGE1D_DATA_OBJ_VIS_MOLECFIT_READY.fits',
                'PNIR': '/home/gmansir/Thesis/Saturn7/Data/PreMolecfit/MOV_Saturn_SCI_IFU_MERGE1D_DATA_OBJ_NIR_MOLECFIT_READY.fits',
                'DIUVB': '/home/gmansir/Thesis/Saturn7/Data/PreMolecfit/MOV_DiskIntegrated_Saturn_7_SCI_IFU_MERGE1D_DATA_OBJ_UVB_MOLECFIT_READY.fits',
                'DIVIS': '/home/gmansir/Thesis/Saturn7/Data/PreMolecfit/MOV_DiskIntegrated_Saturn_7_SCI_IFU_MERGE1D_DATA_OBJ_VIS_MOLECFIT_READY.fits',
                'DINIR': '/home/gmansir/Thesis/Saturn7/Data/PreMolecfit/MOV_DiskIntegrated_Saturn_7_SCI_IFU_MERGE1D_DATA_OBJ_NIR_MOLECFIT_READY.fits'
            },
            'saturn8': {
                'UVB': '/home/gmansir/Thesis/Saturn8/Data/PreMolecfit/MOV_DiskIntegrated_Saturn_8_SCI_IFU_MERGE1D_DATA_OBJ_UVB_MOLECFIT_READY.fits',
                'VIS': '/home/gmansir/Thesis/Saturn8/reflex_tmp_products/molecfit/XSHOOTER/molecfit_correct_1/2023-10-11T16:24:22.507/SCIENCE_TELLURIC_CORR_MOV_DiskIntegrated_Saturn_8_SCI_IFU_MERGE1D_DATA_OBJ_VIS_MOLECFIT_READY.fits',
                'NIR': '/home/gmansir/Thesis/Saturn8/reflex_tmp_products/molecfit/XSHOOTER/molecfit_correct_1/2023-10-11T16:08:39.207/SCIENCE_TELLURIC_CORR_MOV_DiskIntegrated_Saturn_8_SCI_IFU_MERGE1D_DATA_OBJ_NIR_MOLECFIT_READY.fits',
                'PUVB': '/home/gmansir/Thesis/Saturn8/Data/PreMolecfit/MOV_Saturn_SCI_IFU_MERGE1D_DATA_OBJ_UVB_MOLECFIT_READY.fits',
                'PVIS': '/home/gmansir/Thesis/Saturn8/Data/PreMolecfit/MOV_Saturn_SCI_IFU_MERGE1D_DATA_OBJ_VIS_MOLECFIT_READY.fits',
                'PNIR': '/home/gmansir/Thesis/Saturn8/Data/PreMolecfit/MOV_Saturn_SCI_IFU_MERGE1D_DATA_OBJ_NIR_MOLECFIT_READY.fits',
                'DIUVB': '/home/gmansir/Thesis/Saturn8/Data/PreMolecfit/MOV_DiskIntegrated_Saturn_8_SCI_IFU_MERGE1D_DATA_OBJ_UVB_MOLECFIT_READY.fits',
                'DIVIS': '/home/gmansir/Thesis/Saturn8/Data/PreMolecfit/MOV_DiskIntegrated_Saturn_8_SCI_IFU_MERGE1D_DATA_OBJ_VIS_MOLECFIT_READY.fits',
                'DINIR': '/home/gmansir/Thesis/Saturn8/Data/PreMolecfit/MOV_DiskIntegrated_Saturn_8_SCI_IFU_MERGE1D_DATA_OBJ_NIR_MOLECFIT_READY.fits'
            },
            'saturn9': {
                'UVB': '/home/gmansir/Thesis/Saturn9/Data/PreMolecfit/MOV_DiskIntegrated_Saturn_9_SCI_IFU_MERGE1D_DATA_OBJ_UVB_MOLECFIT_READY.fits',
                'VIS': '/home/gmansir/Thesis/Saturn/reflex_tmp_products/molecfit/XSHOOTER/molecfit_correct_1/2023-09-26T16:15:57.896/SCIENCE_TELLURIC_CORR_MOV_DiskIntegrated_Saturn_9_SCI_IFU_MERGE1D_DATA_OBJ_VIS_MOLECFIT_READY.fits',
                'NIR': '/home/gmansir/Thesis/Saturn/reflex_tmp_products/molecfit/XSHOOTER/molecfit_correct_1/2023-09-26T15:58:57.545/SCIENCE_TELLURIC_CORR_MOV_DiskIntegrated_Saturn_9_SCI_IFU_MERGE1D_DATA_OBJ_NIR_MOLECFIT_READY.fits',
                'PUVB': '/home/gmansir/Thesis/Saturn9/Data/PreMolecfit/MOV_Saturn_SCI_IFU_MERGE1D_DATA_OBJ_UVB_MOLECFIT_READY.fits',
                'PVIS': '/home/gmansir/Thesis/Saturn9/Data/PreMolecfit/MOV_Saturn_SCI_IFU_MERGE1D_DATA_OBJ_VIS_MOLECFIT_READY.fits',
                'PNIR': '/home/gmansir/Thesis/Saturn9/Data/PreMolecfit/MOV_Saturn_SCI_IFU_MERGE1D_DATA_OBJ_NIR_MOLECFIT_READY.fits',
                'DIUVB': '/home/gmansir/Thesis/Saturn9/Data/PreMolecfit/MOV_DiskIntegrated_Saturn_9_SCI_IFU_MERGE1D_DATA_OBJ_UVB_MOLECFIT_READY.fits',
                'DIVIS': '/home/gmansir/Thesis/Saturn9/Data/PreMolecfit/MOV_DiskIntegrated_Saturn_9_SCI_IFU_MERGE1D_DATA_OBJ_VIS_MOLECFIT_READY.fits',
                'DINIR': '/home/gmansir/Thesis/Saturn9/Data/PreMolecfit/MOV_DiskIntegrated_Saturn_9_SCI_IFU_MERGE1D_DATA_OBJ_NIR_MOLECFIT_READY.fits'
            },
            'saturn10': {
                'UVB': '/home/gmansir/Thesis/Saturn10/Data/PreMolecfit/MOV_DiskIntegrated_Saturn_10_SCI_IFU_MERGE1D_DATA_OBJ_UVB_MOLECFIT_READY.fits',
                'VIS': '/home/gmansir/Thesis/Saturn/reflex_tmp_products/molecfit/XSHOOTER/molecfit_correct_1/2023-09-26T17:02:43.762/SCIENCE_TELLURIC_CORR_MOV_DiskIntegrated_Saturn_10_SCI_IFU_MERGE1D_DATA_OBJ_VIS_MOLECFIT_READY.fits',
                'NIR': '/home/gmansir/Thesis/Saturn/reflex_tmp_products/molecfit/XSHOOTER/molecfit_correct_1/2023-09-26T16:48:35.158/SCIENCE_TELLURIC_CORR_MOV_DiskIntegrated_Saturn_10_SCI_IFU_MERGE1D_DATA_OBJ_NIR_MOLECFIT_READY.fits',
                'PUVB': '/home/gmansir/Thesis/Saturn10/Data/PreMolecfit/MOV_Saturn_SCI_IFU_MERGE1D_DATA_OBJ_UVB_MOLECFIT_READY.fits',
                'PVIS': '/home/gmansir/Thesis/Saturn10/Data/PreMolecfit/MOV_Saturn_SCI_IFU_MERGE1D_DATA_OBJ_VIS_MOLECFIT_READY.fits',
                'PNIR': '/home/gmansir/Thesis/Saturn10/Data/PreMolecfit/MOV_Saturn_SCI_IFU_MERGE1D_DATA_OBJ_NIR_MOLECFIT_READY.fits',
                'DIUVB': '/home/gmansir/Thesis/Saturn10/Data/PreMolecfit/MOV_DiskIntegrated_Saturn_10_SCI_IFU_MERGE1D_DATA_OBJ_UVB_MOLECFIT_READY.fits',
                'DIVIS': '/home/gmansir/Thesis/Saturn10/Data/PreMolecfit/MOV_DiskIntegrated_Saturn_10_SCI_IFU_MERGE1D_DATA_OBJ_VIS_MOLECFIT_READY.fits',
                'DINIR': '/home/gmansir/Thesis/Saturn10/Data/PreMolecfit/MOV_DiskIntegrated_Saturn_10_SCI_IFU_MERGE1D_DATA_OBJ_NIR_MOLECFIT_READY.fits'
            },
            'saturn11': {
                'UVB': '/home/gmansir/Thesis/Saturn11/Data/PreMolecfit/MOV_DiskIntegrated_Saturn_11_SCI_IFU_MERGE1D_DATA_OBJ_UVB_MOLECFIT_READY.fits',
                'VIS': '/home/gmansir/Thesis/Saturn11/reflex_tmp_products/molecfit/XSHOOTER/molecfit_correct_1/2023-10-10T20:31:34.654/SCIENCE_TELLURIC_CORR_MOV_DiskIntegrated_Saturn_11_SCI_IFU_MERGE1D_DATA_OBJ_VIS_MOLECFIT_READY.fits',
                'NIR': '/home/gmansir/Thesis/Saturn11/reflex_tmp_products/molecfit/XSHOOTER/molecfit_correct_1/2023-10-10T20:03:43.766/SCIENCE_TELLURIC_CORR_MOV_DiskIntegrated_Saturn_11_SCI_IFU_MERGE1D_DATA_OBJ_NIR_MOLECFIT_READY.fits',
                'PUVB': '/home/gmansir/Thesis/Saturn11/Data/PreMolecfit/MOV_Saturn_SCI_IFU_MERGE1D_DATA_OBJ_UVB_MOLECFIT_READY.fits',
                'PVIS': '/home/gmansir/Thesis/Saturn11/Data/PreMolecfit/MOV_Saturn_SCI_IFU_MERGE1D_DATA_OBJ_VIS_MOLECFIT_READY.fits',
                'PNIR': '/home/gmansir/Thesis/Saturn11/Data/PreMolecfit/MOV_Saturn_SCI_IFU_MERGE1D_DATA_OBJ_NIR_MOLECFIT_READY.fits',
                'DIUVB': '/home/gmansir/Thesis/Saturn11/Data/PreMolecfit/MOV_DiskIntegrated_Saturn_11_SCI_IFU_MERGE1D_DATA_OBJ_UVB_MOLECFIT_READY.fits',
                'DIVIS': '/home/gmansir/Thesis/Saturn11/Data/PreMolecfit/MOV_DiskIntegrated_Saturn_11_SCI_IFU_MERGE1D_DATA_OBJ_VIS_MOLECFIT_READY.fits',
                'DINIR': '/home/gmansir/Thesis/Saturn11/Data/PreMolecfit/MOV_DiskIntegrated_Saturn_11_SCI_IFU_MERGE1D_DATA_OBJ_NIR_MOLECFIT_READY.fits'
            },
            'saturn12': {
                'UVB': '/home/gmansir/Thesis/Saturn12/Data/PreMolecfit/MOV_DiskIntegrated_Saturn_12_SCI_IFU_MERGE1D_DATA_OBJ_UVB_MOLECFIT_READY.fits',
                'VIS': '/home/gmansir/Thesis/Saturn12/reflex_tmp_products/molecfit/XSHOOTER/molecfit_correct_1/2023-10-10T22:47:59.442/SCIENCE_TELLURIC_CORR_MOV_DiskIntegrated_Saturn_12_SCI_IFU_MERGE1D_DATA_OBJ_VIS_MOLECFIT_READ.fits',
                'NIR': '/home/gmansir/Thesis/Saturn12/reflex_tmp_products/molecfit/XSHOOTER/molecfit_correct_1/2023-10-10T22:30:42.253/SCIENCE_TELLURIC_CORR_MOV_DiskIntegrated_Saturn_12_SCI_IFU_MERGE1D_DATA_OBJ_NIR_MOLECFIT_READY.fits',
                'PUVB': '/home/gmansir/Thesis/Saturn12/Data/PreMolecfit/MOV_Saturn_SCI_IFU_MERGE1D_DATA_OBJ_UVB_MOLECFIT_READY.fits',
                'PVIS': '/home/gmansir/Thesis/Saturn12/Data/PreMolecfit/MOV_Saturn_SCI_IFU_MERGE1D_DATA_OBJ_VIS_MOLECFIT_READY.fits',
                'PNIR': '/home/gmansir/Thesis/Saturn12/Data/PreMolecfit/MOV_Saturn_SCI_IFU_MERGE1D_DATA_OBJ_NIR_MOLECFIT_READY.fits',
                'DIUVB': '/home/gmansir/Thesis/Saturn12/Data/PreMolecfit/MOV_DiskIntegrated_Saturn_12_SCI_IFU_MERGE1D_DATA_OBJ_UVB_MOLECFIT_READY.fits',
                'DIVIS': '/home/gmansir/Thesis/Saturn12/Data/PreMolecfit/MOV_DiskIntegrated_Saturn_12_SCI_IFU_MERGE1D_DATA_OBJ_VIS_MOLECFIT_READ.fits',
                'DINIR': '/home/gmansir/Thesis/Saturn12/Data/PreMolecfit/MOV_DiskIntegrated_Saturn_12_SCI_IFU_MERGE1D_DATA_OBJ_NIR_MOLECFIT_READY.fits'
            },
            'sunlike': {
                'UVB': '',
                'VIS': '/home/gmansir/Thesis/Sunlike_Star/reflex_tmp_products/molecfit/XSHOOTER/molecfit_correct_1/2023-08-15T13:20:06.230/SCIENCE_TELLURIC_CORR_Hip098197_2_TELL_SLIT_FLUX_MERGE1D_VIS.fits',
                'NIR': '/home/gmansir/Thesis/Sunlike_Star/reflex_tmp_products/molecfit/XSHOOTER/molecfit_correct_1/2023-08-15T10:28:05.222/SCIENCE_TELLURIC_CORR_Hip098197_1_TELL_SLIT_FLUX_MERGE1D_NIR.fits'
            }
        }

        # Define the data types and their prefixes
        body_files = file_paths[self.body]
        data_keys = ['UVB', 'VIS', 'NIR', 'PUVB', 'PVIS', 'PNIR', 'DIUVB', 'DIVIS', 'DINIR']

        # Initialize dictionaries to store the wave and data for each data type
        wave_data = {}
        for key in data_keys:
            wave, data = self.fits_get_wave_data(body_files[key])
            wave_data[key] = [wave, data]

        # Get the file paths for the current body
        self.DIUVBwave, self.DIUVdata = wave_data['DIUVB']
        self.DIVISwave, self.DIVISdata = wave_data['DIVIS']
        self.DINIRwave, self.DINIRdata = wave_data['DINIR']

        # Define the data types and their clip ranges as tuples (start, end)
        data_clip_ranges = {
            'UVB': (0.552, 0.555926),
            'VIS': (0.552, 0.555926),
            'VIS2': (0.994165, 1.01988),
            'NIR': (0.994165, 1.01988),
        }
        prefixes = ['', 'P', 'DI']

        # Initialize dictionaries to store the clipped wave and data for each data type
        if normalize_only == False:
            clipped_data = {}
            for p in prefixes:
                for range in data_clip_ranges.keys():
                    data_key = p+range[0:3]
                    clip_key = p+range
                    wave, data = self.clip_data(wave_data[data_key][0], wave_data[data_key][1],
                                            data_clip_ranges[range][0], data_clip_ranges[range][1])
                    clipped_data[clip_key] = (wave, data)

        if bonus_plots == True:
            fig1 = plt.figure(1)
            ax1 = fig1.add_subplot(111)
            ax1.plot(clipped_data['UVB'], linewidth=0.5)
            ax1.plot(clipped_data['VIS'], linewidth=0.5)
            ax1.set_title('UV (blue) - VIS (orange) Overlap')

            fig2 = plt.figure(2)
            ax2 = fig2.add_subplot(111)
            ax2.plot(clipped_data['NIR'], linewidth=0.5)
            ax2.plot(clipped_data['VIS2'], linewidth=0.5)
            ax2.set_title('VIS (orange) - NIR (blue) Overlap')

        if dichroic == True:

            # Read the CSV file into a DataFrame
            dc = pd.read_csv('/home/gmansir/Thesis/Dichroic/D1_and_D2_final.csv')
            dc.columns.values[0] = 'Column1'
            dc.Wave = dc.Column1*0.001
            UV_low = (np.abs(dc.Wave - wave_data['UV'][0][0])).argmin()
            UV_high = (np.abs(dc.Wave - wave_data['UV'][0][-1])).argmin()
            UV_weights= np.interp(wave_data['UV'][0], dc.Wave[UV_low:UV_high], dc.UVB[UV_low:UV_high])
            VIS_low = (np.abs(dc.Wave - wave_data['VIS'][0][0])).argmin()
            VIS_high = (np.abs(dc.Wave - wave_data['VIS'][0][-1])).argmin()
            VIS_weights= np.interp(wave_data['VIS'][0], dc.Wave[VIS_low:VIS_high], dc.VIS[VIS_low:VIS_high])
            NIR_low = (np.abs(dc.Wave - wave_data['NIR'][0][0])).argmin()
            NIR_high = (np.abs(dc.Wave - wave_data['NIR'][0][-1])).argmin()
            dc.NIR = dc.NIR.fillna(1).clip(upper=100)
            NIR_weights= np.interp(wave_data['NIR'][0], dc.Wave[NIR_low:NIR_high], dc.NIR[NIR_low:NIR_high])
            wave_data['UV'][1] = wave_data['UV'][1] / np.array(UV_weights*0.01)
            wave_data['VIS'][1] = wave_data['VIS'][1] / np.array(VIS_weights*0.01)
            wave_data['NIR'][1] = np.nan_to_num(wave_data['NIR'][1], nan=0)
            wave_data['NIR'][1] = wave_data['NIR'][1] / np.array(NIR_weights*0.01)

        # Bin data to lower resolution
        num_bins = 8

        # Initialize dictionaries to store the binned wave and data for each data type
        if normalize_only == False:
            binned_data = {}
            for key in clipped_data.keys():
                wave, data = self.bin_data(clipped_data[key][0], clipped_data[key][1], num_bins)
                binned_data[key] = (wave, data)

        if bonus_plots == True:
            ax1.plot(binned_data['UVB'], linewidth=0.5)
            ax1.plot(binned_data['VIS'], linewidth=0.5)
            ax2.plot(binned_data['NIR'], linewidth=0.5)
            ax2.plot(binned_data['VIS2'], linewidth=0.5)

        # Use linear regression to find the scale factors between the two spectra and adjust accordingly
        if normalize_only == False:
            UVIS = np.polyfit(binned_data['UVB'][1][1:-1], binned_data['VIS'][1][1:-1], 1)
            VIR = np.polyfit(binned_data['NIR'][1][1:-1], binned_data['VIS2'][1][1:-1], 1)
            PUVIS = np.polyfit(binned_data['PUVB'][1][1:-1], binned_data['PVIS'][1][1:-1], 1)
            PVIR = np.polyfit(binned_data['PNIR'][1][1:-1], binned_data['PVIS2'][1][1:-1], 1)
            DIUVIS = np.polyfit(binned_data['DIUVB'][1][1:-1], binned_data['DIVIS'][1][1:-1], 1)
            DIVIR = np.polyfit(binned_data['DINIR'][1][1:-1], binned_data['DIVIS2'][1][1:-1], 1)

        if bonus_plots == True:
            y = np.polyval(UVIS, np.array(binned_data['UVB'][1]))
            fig4 = plt.figure(4)
            ax4 = fig4.add_subplot(111)
            ax4.plot(binned_data['UVB'][1][1:-1], binned_data['VIS'][1][1:-1], 'o')
            ax4.plot(binned_data['UVB'][1], y)
            ax4.set_title('UV - VIS Linear Regression')

            y2 = np.polyval(VIR, np.array(binned_data['NIR'][1]))
            fig5 = plt.figure(5)
            ax5 = fig5.add_subplot(111)
            ax5.plot(binned_data['NIR'][1], binned_data['VIS2'][1], 'o')
            ax5.plot(binned_data['NIR'][1], y2)
            ax5.set_title('VIS - NIR Linear Regression')

        if normalize_only == True:
            for key in data_keys:
                wave_data[key][1] = wave_data[key][1] / np.percentile(np.nan_to_num(wave_data[key][1], nan=0.0), 99)
        else:
            wave_data['UVB'][1] = np.polyval(UVIS, np.array(wave_data['UVB'][1]))
            wave_data['NIR'][1] = np.polyval(VIR, np.array(wave_data['NIR'][1]))
            wave_data['PUVB'][1] = np.polyval(PUVIS, np.array(wave_data['PUVB'][1]))
            wave_data['PNIR'][1] = np.polyval(PVIR, np.array(wave_data['PNIR'][1]))
            wave_data['DIUVB'][1] = np.polyval(DIUVIS, np.array(wave_data['DIUVB'][1]))
            wave_data['DINIR'][1] = np.polyval(DIVIR, np.array(wave_data['DINIR'][1]))

        # Define the bands and corresponding mask ranges
        mask_ranges = {
            'UVB': [[wave_data['UVB'][0][0], 0.30501], [0.544649, wave_data['UVB'][0][-1]]],
            'VIS': [[wave_data['VIS'][0][0], 0.544649], [1.01633, wave_data['VIS'][0][-1]]],
            'NIR': [[wave_data['NIR'][0][0], 1.01633], [2.192, wave_data['NIR'][0][-1]]],
            'PUVB': [[wave_data['PUVB'][0][0], 0.30501], [0.544649, wave_data['PUVB'][0][-1]]],
            'PVIS': [[wave_data['PVIS'][0][0], 0.544649], [1.01633, wave_data['PVIS'][0][-1]]],
            'PNIR': [[wave_data['PNIR'][0][0], 1.01633], [2.192, wave_data['PNIR'][0][-1]]],
            'DIUVB': [[wave_data['DIUVB'][0][0], 0.30501], [0.544649, wave_data['DIUVB'][0][-1]]],
            'DIVIS': [[wave_data['DIVIS'][0][0], 0.544649], [1.01633, wave_data['DIVIS'][0][-1]]],
            'DINIR': [[wave_data['DINIR'][0][0], 1.01633], [2.192, wave_data['DINIR'][0][-1]]],
        }

        # Initialize lists for wave and spec
        wave = []
        spec = []
        pwave = []
        pspec = []
        diwave = []
        dispec = []

        # Loop through the bands and mask data accordingly
        for band in data_keys:
            wave_data_band = wave_data[band]
            mask_range = mask_ranges[band]

            # Mask and compress data
            wave_band, spec_band = self.mask_and_compress(wave_data_band[0], wave_data_band[1], mask_range)

            setattr(self, band + 'wave', wave_band)
            setattr(self, band + 'data', spec_band)

            # Append data to the appropriate lists
            if 'VIS' in band:
                spec_band += 0.5
            if band in ['UVB', 'VIS', 'NIR']:
                wave.append(wave_band)
                spec.append(spec_band)
            elif band in ['PUVB', 'PVIS', 'PNIR']:
                pwave.append(wave_band)
                pspec.append(spec_band)
            else:
                diwave.append(wave_band)
                dispec.append(spec_band)
        # The wave and spec lists now contain the masked and compressed data for each band

        wave = np.array([i for band in wave for i in band])
        spec = np.array([i for band in spec for i in band])
        self.pwave = np.array([i for band in pwave for i in band])
        self.pspec = np.array([i for band in pspec for i in band])
        self.diwave = np.array([i for band in diwave for i in band])
        self.dispec = np.array([i for band in dispec for i in band])

        # Uses the edges of the full spectrum to remove any artificial incline  or below zero value
        # due to the edge-matching (move to linear regression only)
        if normalize_only == False:
            wave_edges = [i for edge in [wave[0:1000], wave[-1000:-1]] for i in edge]
            spec_edges = [i for edge in [spec[0:1000], spec[-1000:-1]] for i in edge]
            pwave_edges = [i for edge in [self.pwave[0:1000], self.pwave[-1000:-1]] for i in edge]
            pspec_edges = [i for edge in [self.pspec[0:1000], self.pspec[-1000:-1]] for i in edge]
            diwave_edges = [i for edge in [self.diwave[0:1000], self.diwave[-1000:-1]] for i in edge]
            dispec_edges = [i for edge in [self.dispec[0:1000], self.dispec[-1000:-1]] for i in edge]
            flatten_coeff = np.polyfit(wave_edges, spec_edges, 1)
            flatten_fit = np.polyval(flatten_coeff, wave)
            spec = spec - flatten_fit
            spec += np.abs(np.min(spec)) + 0.0001
            pflatten_coeff = np.polyfit(pwave_edges, pspec_edges, 1)
            pflatten_fit = np.polyval(pflatten_coeff, self.pwave)
            self.pspec = self.pspec - pflatten_fit
            self.pspec += np.abs(np.min(self.pspec)) + 0.0001
            diflatten_coeff = np.polyfit(diwave_edges, dispec_edges, 1)
            diflatten_fit = np.polyval(diflatten_coeff, self.diwave)
            self.dispec = self.dispec - diflatten_fit
            self.dispec += np.abs(np.min(self.dispec)) + 0.0001

        #if wave_include == True:
        #    windows = [[1.05, 1.08], [1.12, 1.13], [1.5, 1.58], [1.75, 1.77], [2.025, 2.050], [2.23, 2.38]]
        #    for w in windows:
        #        plt.axvspan(w[0], w[1], facecolor='olivedrab', alpha=0.4)

        # Saves the spectrum for further analysis
        clean_spec_file = self.work_dir + f'PostMolecfit/MOV_{self.body.title()}_SCI_IFU_FULL_SPECTRUM.fits'
        hdu = fits.open(clean_spec_file, mode='update')
        flux = np.array(spec)
        wav = np.array(wave)
        hdu[0].data = flux
        hdu[1].data = wav
        hdu.close()
        print(f'File Saved to: ' + self.work_dir + f'PostMolecfit/MOV_{self.body.title()}_SCI_IFU_FULL_SPECTRUM.fits')

    def compare_fit(self, annotate=False, **kwargs):

        pre = '/home/gmansir/Thesis/Titan_old/Data/PostMolecfit/MOV_Titan_SCI_IFU_FULL_SPECTRUM_PREMOLECFIT.fits'
        #default = '/home/gmansir/Thesis/Titan/Data/PostMolecfit/MOV_Titan_SCI_IFU_FULL_SPECTRUM_BETA_DEFAULTS.fits'
        #adjust = '/home/gmansir/Thesis/Titan/Data/PostMolecfit/MOV_Titan_SCI_IFU_FULL_SPECTRUM_BETA_ADJUSTED.fits'
        post = '/home/gmansir/Thesis/Titan/Data/PostMolecfit/MOV_Titan_SCI_IFU_FULL_SPECTRUM.fits'

        # Plot the results!!
        w, h = plt.figaspect(0.25)
        fig = plt.figure(1, figsize=(w, h))
        ax = fig.add_subplot(111)
        pre_hdu = fits.open(pre)
        pre_flux = pre_hdu[0].data
        pre_wav = pre_hdu[1].data
        pre_hdu.close()
        ax.plot(pre_wav, pre_flux, linewidth=0.5, color='black', label='PreMolecfit')
        for i in [[0.686294, 0.691341], [0.759164, 0.787264], [0.822648, 0.822888], [0.93059, 0.955106], [1.11, 1.16],
                  [1.33, 1.48912], [2.41, 2.48], [1.7846, 1.9654]]:
            lidx = (np.abs(pre_wav - i[0])).argmin()
            hidx = (np.abs(pre_wav - i[1])).argmin()
            ax.plot(pre_wav[lidx:hidx], pre_flux[lidx:hidx], linewidth=0.5, color='silver')
        def_hdu = fits.open(post)
        def_flux = def_hdu[0].data
        def_wav = def_hdu[1].data
        def_hdu.close()
        ax.plot(def_wav, def_flux-0.03, linewidth=0.5, label='PostMolecfit')
        for i in [[0.686294, 0.691341], [0.759164, 0.787264], [0.822648, 0.822888], [0.93059, 0.955106], [1.11, 1.16],
                  [1.33, 1.48912], [2.41, 2.48], [1.7846, 1.9654]]:
            lidx = (np.abs(def_wav - i[0])).argmin()
            hidx = (np.abs(def_wav - i[1])).argmin()
            ax.plot(def_wav[lidx:hidx], def_flux[lidx:hidx]-0.03, linewidth=0.5, color='silver')
        #adj_hdu = fits.open(adjust)
        #adj_flux = adj_hdu[0].data
        #adj_wav = adj_hdu[1].data
        #adj_hdu.close()
        #ax.plot(adj_wav, adj_flux, linewidth=0.5, label='Adjusted')
        #for i in [[0.686294, 0.691341], [0.759164, 0.787264], [0.822648, 0.822888], [0.93059, 0.955106], [1.11, 1.16],
        #          [1.33, 1.48912], [2.41, 2.48], [1.7846, 1.9654]]:
        #    lidx = (np.abs(adj_wav - i[0])).argmin()
        #    hidx = (np.abs(adj_wav - i[1])).argmin()
        #    ax.plot(adj_wav[lidx:hidx], adj_flux[lidx:hidx], linewidth=0.5, color='silver')

        ax.set_xscale('log')
        ax.set_ylim(top=1.75)
        ax.set_title(f'{self.body.title()} Fit Comparison')

        # Set x-axis tick labels to non-scientific notation
        if annotate == True:
            self.annotate_plot_clean(**kwargs)
        ax.xaxis.set_major_formatter(StrMethodFormatter('{x:.3f}'))
        ax.xaxis.set_minor_formatter(StrMethodFormatter('{x:.3f}'))
        ax.legend()
        plt.show()

    def load_eso_data(self, eso_path, m, b):
        eso_data = []
        with open(eso_path, 'r') as file:
            for line in file:
                columns = line.strip().split()
                if len(columns) == 2:
                    eso_data.append((float(columns[0]), float(columns[1])))
        eso_x = np.array([x * 0.0001 for x, _ in eso_data])
        eso_y = np.array([y * m + b for _, y in eso_data])
        return eso_x, eso_y

    def telluric_standard_solutuion(self, bonus_plots=False, *args, **kwargs):

        file_paths = {
            'enceladus': {
                'UVB': '/home/gmansir/Thesis/Telluric/Data/MOV_DiskIntegrated_LTT7987_Saturn_SCI_IFU_MERGE1D_DATA_OBJ_UVB_MOLECFIT_READY.fits',
                'VIS': '/home/gmansir/Thesis/Telluric/reflex_tmp_products/molecfit/XSHOOTER/molecfit_correct_1/2023-10-18T09:54:07.666/SCIENCE_TELLURIC_CORR_MOV_DiskIntegrated_LTT7987_Titan_SCI_IFU_MERGE1D_DATA_OBJ_VIS_MOLECFIT_READY.fits',
                'NIR': '/home/gmansir/Thesis/Telluric/reflex_tmp_products/molecfit/XSHOOTER/molecfit_correct_1/2023-10-18T09:39:43.837/SCIENCE_TELLURIC_CORR_MOV_DiskIntegrated_LTT7987_Titan_SCI_IFU_MERGE1D_DATA_OBJ_NIR_MOLECFIT_READY.fits'
            },
            'titan': {
                'UVB': '/home/gmansir/Thesis/Telluric/Data/MOV_DiskIntegrated_LTT7987_Saturn_SCI_IFU_MERGE1D_DATA_OBJ_UVB_MOLECFIT_READY.fits',
                'VIS': '/home/gmansir/Thesis/Telluric/reflex_tmp_products/molecfit/XSHOOTER/molecfit_correct_1/2023-10-18T09:54:07.666/SCIENCE_TELLURIC_CORR_MOV_DiskIntegrated_LTT7987_Titan_SCI_IFU_MERGE1D_DATA_OBJ_VIS_MOLECFIT_READY.fits',
                'NIR': '/home/gmansir/Thesis/Telluric/reflex_tmp_products/molecfit/XSHOOTER/molecfit_correct_1/2023-10-18T09:39:43.837/SCIENCE_TELLURIC_CORR_MOV_DiskIntegrated_LTT7987_Titan_SCI_IFU_MERGE1D_DATA_OBJ_NIR_MOLECFIT_READY.fits'
            },
            'neptune': {
                'UVB': '/home/gmansir/Thesis/Telluric/Data/MOV_DiskIntegrated_FIEGE-110_SCI_IFU_MERGE1D_DATA_OBJ_UVB_MOLECFIT_READY.fits',
                'VIS': '/home/gmansir/Thesis/Telluric/reflex_tmp_products/molecfit/XSHOOTER/molecfit_correct_1/2023-10-15T22:59:10.824/SCIENCE_TELLURIC_CORR_MOV_DiskIntegrated_FIEGE-110_SCI_IFU_MERGE1D_DATA_OBJ_VIS_MOLECFIT_READY.fits',
                'NIR': '/home/gmansir/Thesis/Telluric/reflex_tmp_products/molecfit/XSHOOTER/molecfit_correct_1/2023-10-15T22:34:09.646/SCIENCE_TELLURIC_CORR_MOV_DiskIntegrated_FIEGE-110_SCI_IFU_MERGE1D_DATA_OBJ_NIR_MOLECFIT_READY.fits'
            },
            'uranus': {
                'UVB':  '/home/gmansir/Thesis/Telluric/Data/MOV_DiskIntegrated_GD71_SCI_IFU_MERGE1D_DATA_OBJ_UVB_MOLECFIT_READY.fits',
                'VIS': '/home/gmansir/Thesis/Telluric/reflex_tmp_products/molecfit/XSHOOTER/molecfit_correct_1/2023-10-16T23:22:42.732/SCIENCE_TELLURIC_CORR_MOV_DiskIntegrated_GD71_SCI_IFU_MERGE1D_DATA_OBJ_VIS_MOLECFIT_READY.fits',
                'NIR': '/home/gmansir/Thesis/Telluric/reflex_tmp_products/molecfit/XSHOOTER/molecfit_correct_1/2023-10-16T22:27:59.279/SCIENCE_TELLURIC_CORR_MOV_DiskIntegrated_GD71_SCI_IFU_MERGE1D_DATA_OBJ_NIR_MOLECFIT_READY.fits'
            },
            'saturn': {
                'UVB': '/home/gmansir/Thesis/Telluric/Data/MOV_DiskIntegrated_LTT7987_Saturn_SCI_IFU_MERGE1D_DATA_OBJ_UVB_MOLECFIT_READY.fits',
                'VIS': '/home/gmansir/Thesis/Telluric/reflex_tmp_products/molecfit/XSHOOTER/molecfit_correct_1/2023-10-17T11:39:30.833/SCIENCE_TELLURIC_CORR_MOV_DiskIntegrated_LTT7987_Saturn_SCI_IFU_MERGE1D_DATA_OBJ_VIS_MOLECFIT_READY.fits',
                'NIR': '/home/gmansir/Thesis/Telluric/reflex_tmp_products/molecfit/XSHOOTER/molecfit_correct_1/2023-10-17T10:49:51.325/SCIENCE_TELLURIC_CORR_MOV_DiskIntegrated_LTT7987_Saturn_SCI_IFU_MERGE1D_DATA_OBJ_NIR_MOLECFIT_READY.fits'
            },
        }

        match = re.match(r"([a-z]+)", self.body)
        planet = match.group(1)

        body_files = file_paths[planet]
        UVBwave, UVBdata  = self.fits_get_wave_data(body_files['UVB'])
        VISwave, VISdata = self.fits_get_wave_data(body_files['VIS'])
        NIRwave, NIRdata = self.fits_get_wave_data(body_files['NIR'])
        UVBdata = UVBdata / np.percentile(UVBdata, 99)
        VISdata = VISdata / np.percentile(VISdata, 99)
        NIRdata = NIRdata / np.percentile(NIRdata, 99)

        #***remove the edge matching here***
        # Clip out edge regions for scaling:
        #UVwave_clip1, UVdata_clip1 = self.clip_data(UVwave, UVdata, 0.552, 0.555926)
        #VISwave_clip1, VISdata_clip1 = self.clip_data(VISwave, VISdata, 0.552, 0.555926)
        #VISwave_clip2, VISdata_clip2 = self.clip_data(VISwave, VISdata, 0.994165, 1.01988)
        #NIRwave_clip1, NIRdata_clip1 = self.clip_data(NIRwave, NIRdata, 0.994165, 1.01988)
        # UVlow - 0.545284

        # Bin data to lower resolution
        #num_bins = 8
        #UVwave_binned, UVdata_binned = self.bin_data(UVwave_clip1, UVdata_clip1, num_bins)
        #VISwave_binned, VISdata_binned = self.bin_data(VISwave_clip1, VISdata_clip1, num_bins)
        #VISwave_binned2, VISdata_binned2 = self.bin_data(VISwave_clip2, VISdata_clip2, num_bins)
        #NIRwave_binned, NIRdata_binned = self.bin_data(NIRwave_clip1, NIRdata_clip1, num_bins)

        # Use linear regression to find the scale factors between the two spectra and adjust accordingly
        #UVIS = np.polyfit(UVdata_binned[1:-1], VISdata_binned[1:-1], 1)
        #VIR = np.polyfit(NIRdata_binned[1:-1], VISdata_binned2[1:-1], 1)
        #UVdata = np.polyval(UVIS, np.array(UVdata))
        #NIRdata = np.polyval(VIR, np.array(NIRdata))
        #******

        # Mask telluric noise
        UVBmask_ranges = [[UVBwave[0], 0.30501], [0.544649, UVBwave[-1]]]
        VISmask_ranges = [[VISwave[0], 0.544649], [1.01633, VISwave[-1]]]
        NIRmask_ranges = [[NIRwave[0], 1.01633], [2.192, NIRwave[-1]]]
        UVBwave, UVBdata = self.mask_and_compress(UVBwave, UVBdata, UVBmask_ranges)
        VISwave, VISdata = self.mask_and_compress(VISwave, VISdata, VISmask_ranges)
        NIRwave, NIRdata = self.mask_and_compress(NIRwave, NIRdata, NIRmask_ranges)

        if self.body == 'enceladus' or self.body == 'titan' or 'saturn' in self.body:
            eso_path = '/home/gmansir/Thesis/Telluric/Data/fLTT7987.dat'
            m = 10 ** 13
            b = 0.2
        elif 'neptune' in self.body:
            eso_path = '/home/gmansir/Thesis/Telluric/Data/fFeige110.dat'
            m = 3 * 10 ** 12
            b = 0.2
        elif 'uranus' in self.body:
            eso_path = '/home/gmansir/Thesis/Telluric/Data/fGD71.dat'
            m = 10 ** 13
            b = 0.2

        eso_x, eso_y = self.load_eso_data(eso_path, m, b)
        if bonus_plots == True:
            w, h = plt.figaspect(0.25)
            fig4 = plt.figure(4, figsize=(w, h))
            ax4 = fig4.add_subplot(111)
            ax4.plot(eso_x, eso_y, color='indianred')

        UVBwave_binned = []
        VISwave_binned = []
        NIRwave_binned = []
        UVBdata_binned = []
        VISdata_binned = []
        NIRdata_binned = []
        UVB_coeffs = None
        VIS_coeffs = None
        NIR_coeffs = None
        UVB_yfit = None
        VIS_yfit = None
        NIR_yfit = None
        eso_xuvb_binned = []
        eso_xvis_binned = []
        eso_xnir_binned = []
        eso_yuvb_binned = []
        eso_yvis_binned = []
        eso_ynir_binned = []
        eso_uvb_coeffs = None
        eso_vis_coeffs = None
        eso_nir_coeffs = None
        eso_yuvb_fit = None
        eso_yvis_fit = None
        eso_ynir_fit = None
        UVB_results = []
        VIS_results = []
        NIR_results = []
        eso_UVB_results = []
        eso_VIS_results = []
        eso_NIR_results = []

        UVB_dict = {'wave':UVBwave, 'xbin':UVBwave_binned, 'data':UVBdata, 'ybin':UVBdata_binned,
                   'coeffs':UVB_coeffs, 'yfit':UVB_yfit, 'solution':UVB_results}
        VIS_dict = {'wave':VISwave, 'xbin':VISwave_binned, 'data':VISdata, 'ybin':VISdata_binned,
                    'coeffs':VIS_coeffs, 'yfit':VIS_yfit, 'solution':VIS_results}
        NIR_dict = {'wave':NIRwave, 'xbin':NIRwave_binned, 'data':NIRdata, 'ybin':NIRdata_binned,
                    'coeffs':NIR_coeffs, 'yfit':NIR_yfit, 'solution':NIR_results}
        eso_UVB_dict = {'wave':UVBwave, 'xbin':eso_xuvb_binned, 'ybin':eso_yuvb_binned, 'coeffs': eso_uvb_coeffs,
                       'yfit': eso_yuvb_fit, 'solution':eso_UVB_results}
        eso_VIS_dict = {'wave': VISwave, 'xbin': eso_xvis_binned, 'ybin': eso_yvis_binned,
                        'coeffs': eso_vis_coeffs, 'yfit': eso_yvis_fit, 'solution':eso_VIS_results}
        eso_NIR_dict = {'wave': NIRwave, 'xbin': eso_xnir_binned, 'ybin': eso_ynir_binned,
                        'coeffs': eso_nir_coeffs, 'yfit': eso_ynir_fit, 'solution':eso_NIR_results}
        dicts = {'Tell_UV':UVB_dict, 'Tell_VIS':VIS_dict, 'Tell_NIR':NIR_dict,
                 'eso_UV': eso_UVB_dict, 'eso_VIS': eso_VIS_dict, 'esoNIR': eso_NIR_dict}

        num_bins = 150
        # More bins!
        for label, info in dicts.items():
            wav = info['wave']
            if 'eso' in label:
                low_idx = (np.abs(eso_x - wav[0])).argmin()
                high_idx = (np.abs(eso_x - wav[-1])).argmin()
                wav = eso_x[low_idx:high_idx]
                data = eso_y[low_idx:high_idx]
            else:
                data = info['data']
            xbin = info['xbin']
            ybin = info['ybin']
            coeffs = info['coeffs']
            yfit = info['yfit']
            solution = info['solution']
            for axis in ['x', 'y']:
                if axis == 'x':
                    bin = xbin
                    dat = wav
                else:
                    bin = ybin
                    dat = data
                bin_width = len(dat) // num_bins
                for bin_start in np.arange(0, len(dat), bin_width):
                    bin_end = bin_start + bin_width
                    bin_data = dat[bin_start:bin_end]
                    bin_mean = np.median(bin_data)
                    bin.append(bin_mean)
            #if 'NIR' in label:
            #    xbin = np.array(xbin)
            #    ybin = np.array(ybin)
            #    mask = (xbin < 1.0207) | (xbin > 1.1141)
            #    xbin = xbin[mask]
            #    ybin = ybin[mask]
            #    mask2 = (xbin < 1.2) | (xbin > 1.32)
            #    xbin = xbin[mask2]
            #    ybin = ybin[mask2]
            coeffs = np.polyfit(xbin, ybin, 2)
            yfit = np.polyval(coeffs, wav)
            spl = uspline(xbin, ybin, k=3, s=0.01)
            wave = wav * un.um
            flux = data * (un.erg / (un.s * un.cm ** 2 * un.um))
            spectrum = Spectrum1D(spectral_axis=wave, flux=flux)
            continuum_fit = fit_generic_continuum(spectrum)
            if 'Tell' in label:
                solution.append(spl(wav))
                if bonus_plots == True:
                    ax4.plot(wav, data, color='grey', label='Disk Integrated Star Spectrum')
                    ax4.plot(xbin, ybin, 'o', color='steelblue', label='Binned Data')
            else:
                spline_resampled = np.interp(info['wave'], wav, spl(wav))
                solution.append(spline_resampled)
                if bonus_plots == True:
                    ax4.plot(xbin, ybin, 'o', color='firebrick', label='np.interp')
            if bonus_plots == True:
                ax4.plot(spectrum.spectral_axis, continuum_fit(spectrum.spectral_axis), label='Spectutils',
                         color='rebeccapurple')
                ax4.plot(wav, spl(wav), color='green', label='Spline')
                ax4.plot(wav, yfit, color='black', label='Polyfit')
                ax4.set_xscale('log')

        self.UVB_solution = np.array(eso_UVB_results)/np.array(UVB_results)
        self.VIS_solution = np.array(eso_VIS_results)/np.array(VIS_results)
        self.NIR_solution = np.array(eso_NIR_results)/np.array(NIR_results)

        if bonus_plots == True:
            plt.ylim(-0.5, 2.5)
            plt.show()

            w, h = plt.figaspect(0.25)
            fig2 = plt.figure(2, figsize=(w, h))
            ax2 = fig2.add_subplot(111)
            ax2.plot(UVBwave, self.UVB_solution[0], color='green')
            ax2.plot(VISwave, self.VIS_solution[0], color='green')
            ax2.plot(NIRwave, self.NIR_solution[0], color='green')
            ax2.set_xscale('log')

            w, h = plt.figaspect(0.25)
            fig7 = plt.figure(7, figsize=(w, h))
            ax7 = fig7.add_subplot(111)
            ax7.plot(UVBwave, UVBdata * self.UVB_solution[0])
            ax7.plot(VISwave, VISdata * self.VIS_solution[0])
            ax7.plot(NIRwave, NIRdata * self.NIR_solution[0])
            ax7.set_xscale('log')
            plt.title('Telluric Solution applied to telluric star data')

            plt.show()

    def test_solution(self):

        self.edge_matching(bonus_plots=False, annotate=False, threshold=0.3, dichroic=False)
        self.telluric_standard_solutuion(bonus_plots=True)

        UVB_object = self.UVBdata
        VIS_object = self.VISdata
        NIR_object = self.NIRdata
        UVBwave = self.UVBwave
        VISwave = self.VISwave
        NIRwave = self.NIRwave

        UVB_spec = UVB_object / self.UVB_solution[0]
        VIS_spec = VIS_object / self.VIS_solution[0]
        NIR_spec = NIR_object / self.NIR_solution[0]
        UVB_spec = UVB_spec / np.percentile(UVB_spec, 99)
        VIS_spec = VIS_spec / np.percentile(VIS_spec, 99) + 0.5
        NIR_spec = NIR_spec / np.percentile(NIR_spec, 99)

        # PreMolecfit
        #ax1 = fig1.add_subplot(411)
        #ax1.plot(self.pwave, self.pspec, color='black', linewidth=0.3)
        #for i in [[0.686294, 0.691341], [0.759164, 0.787264], [0.822648, 0.822888], [0.93059, 0.955106], [1.11, 1.16],
        #          [1.33, 1.48912], [2.41, 2.48], [1.7846, 1.9654]]:
        #    lidx = (np.abs(self.pwave - i[0])).argmin()
        #    hidx = (np.abs(self.pwave - i[1])).argmin()
        #    ax1.plot(self.pwave[lidx:hidx], self.pspec[lidx:hidx], linewidth=0.3, color='lightgrey')
        #ax1.set_title('Pre Molecfit')
        #ax1.set_xscale('log')
        #ax1.xaxis.set_major_formatter(StrMethodFormatter('{x:.3f}'))
        #ax1.xaxis.set_minor_formatter(StrMethodFormatter('{x:.3f}'))

        # Define a function for plotting the data and light grey regions
        def plot_data_and_regions(ax, wave, data, title):
            ax.plot(wave, data, color='black', linewidth=0.3)
            corrected_regions = [[0.58,0.60],[0.64,0.66],[0.68,0.75],[0.78,0.86],[0.88, 1.0],
                                 [1.062, 1.244],[1.26,1.57],[1.63,2.48]]
            untrustworthy_regions = [[0.50,0.51],[0.758,0.77],[0.685,0.695],[0.625,0.632],[0.72,0.73], [0.81,0.83], [0.89, 0.98],
                                     [1.107, 1.164],[1.3, 1.5], [1.73, 2.0], [2.38, 2.48], [1.946,1.978],
                                     [1.997,2.032],[2.043,2.080]]
            for region in corrected_regions:
                lidx = (np.abs(wave - region[0])).argmin()
                hidx = (np.abs(wave - region[1])).argmin()
                ax.plot(wave[lidx:hidx], data[lidx:hidx], linewidth=0.3, color='#121212')
            for region in untrustworthy_regions:
                lidx = (np.abs(wave - region[0])).argmin()
                hidx = (np.abs(wave - region[1])).argmin()
                ax.plot(wave[lidx:hidx], data[lidx:hidx], linewidth=0.3, color='lightgrey')
            ax.set_title(title)
            ax.set_xscale('log')
            ax.xaxis.set_major_formatter(StrMethodFormatter('{x:.3f}'))
            ax.xaxis.set_minor_formatter(StrMethodFormatter('{x:.3f}'))
            ax.set_ylim(0,1.5)

        # Create a figure
        w, h = plt.figaspect(1)
        fig1 = plt.figure(1, figsize=(w, h))
        plt.subplots_adjust(hspace=0.4)

        # Plot the Disk Integrated spectrum
        ax2 = fig1.add_subplot(311)
        plot_data_and_regions(ax2, self.diwave, self.dispec, 'Disk Integrated')

        # Plot the Disk Integrated spectrum after it's been run through Molecfit
        ax3 = fig1.add_subplot(312)
        ranges = [
            (self.UVBwave, self.UVBdata, 'Post Molecfit'),
            (self.VISwave, self.VISdata, 'Post Molecfit'),
            (self.NIRwave, self.NIRdata, 'Post Molecfit')
        ]
        for wave, data, title in ranges:
            plot_data_and_regions(ax3, wave, data, title)

        # Plot the Post Molecfit spectrum after applying the Telluric Solution
        ax4 = fig1.add_subplot(313)
        ranges = [
            (UVBwave, UVB_spec, 'Telluric Solution Applied'),
            (VISwave, VIS_spec, 'Telluric Solution Applied'),
            (NIRwave, NIR_spec, 'Telluric Solution Applied')
        ]
        for wave, data, title in ranges:
            plot_data_and_regions(ax4, wave, data, title)

        # Add a figure title
        current_date = datetime.now().strftime('%Y-%m-%d')
        fig1.suptitle(f'{self.body.title()}\nGenerated on: {current_date}, '
                      f'Molecfit Rerun, Removal of 2.35 Region and some molecules')

        #investigate nir 1.1 - 1.25 and snr difference between nir and vis/uv


def multi_object_handler(object_list, **kwargs):
    '''
    Wrapper outside of the class to work with multiple bodies at once

    :param object_list:
    :param kwargs:
    :return:
    '''
    for o in object_list:
        obj = DataAnalysis(o, **kwargs)
        # do analysis on the planet using the object

# einstein a with oscillator strength -> try to change the cross-section for each planet (linear?)
# level 1 einstein coefficient
# 2 correcting temperature and abundance for each planet (multiplicative)
# 3 full radiative transfer model

# browse planetary spectra modeling tools (google, ads, papers, etc (BART, FORMOSA (look at temperature ranges and other parameters))
# investigate which molecules are important for which planets (set defaults) by eye and lit

# numpy - digitize for binning
# Consider airmass when applying to objects
# (check if eso has more images so i can interpolate solutions for other airmasses, otherwise google for oother solutions)

# Look up saturn image positions to make sure molecfit is consistent for offsets that will be combined
# Look up XSHOOTER lit to see what other people do to deal with the telluric correction

# identify lines
    # ghost plot of the telluric in the background
    #print out neptune

# Thesis checklist/timeline:
# 2 months after submission for beuracracy
# 3 months intensive thesis writing + paper
# 4 months - basic modeling
# 2 months - final prep of data (visual and tables)

# Process Standard Stars through updated Molecfit routine
# Triple check RA/Dec and ESO data to make sure that the stars are the same YYYY
# Look at single pixels of our standard star images to see if the spectra is closer to ESO's anywhere
# Send updated plots by email
# Darker grey for telluric features that are corrected. Lighter grey for regions we can't trust

# Add a single value to see if we can align the different bands
    # try to come up with an instrumental reason as to why the telluric solution works so well for the
    # star and not for the extended sources and why a single value is appropriate if so

# Commit to Github
# Plot with pre and post molecfit to show correction of specific telluric molecules
# iraf documentation for response curve?
# Search ESO archive for more tellurics (XSHOOTER, IFU prefered) in the nights surrounding ours
