import numpy as np
import numpy.ma as ma
# import scipy as sp
from scipy.stats import chisquare
from scipy.signal import convolve  # , boxcar
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import glob
from astropy.table import Table  # , Column
from astropy.io import fits
from astropy.time import Time
import pdb
# from mpl_toolkits.basemap import Basemap
# from mpl_toolkits.mplot3d import Axes3D
import os, sys


class PlanetAnalysis:

    # TODO: Finish Molecfitting Saturn, run ifu_recovery and plot_ifu
    # TODO: Double check analysis software from proposal
    # TODO: Match up band edges
    # TODO: Email molecfit team for clarification on output files
    # TODO: Check single spacial pixel for sine shape (Saturn)
    # TODO: Compare results from molecfit by band and time for consistency
    # TODO: Add option to plot individual spec with averages

    def __init__(self, **kwargs):

        # Determine what dataset we will be working with (Planet, Pre/Post Molecfit, Bands)
        planets_ready = ['Moon', 'Saturn', 'Neptune', 'Enceladus-Titan']
        if 'planet' in kwargs.keys():
            self.planet = kwargs['planet']
            try:
                self.homedir = '/home/gmansir/Thesis/' + self.planet
            except OSError:
                print("Planet not recognized. Please choose from: ", planets_ready)
                self.planet = str()
                while True:
                    self.planet = input("Which planet would you like to work on (string, capitalized): ")
                    if self.planet not in planets_ready:
                        print("Planet not recognized. Please choose from: ", planets_ready)
                    else:
                        break
        else:
            self.planet = str()
            while True:
                self.planet = input("Which planet would you like to work on (string, capitalized): ")
                if self.planet not in planets_ready:
                    print("Planet not recognized. Please choose from: ", planets_ready)
                else:
                    self.homedir = '/home/gmansir/Thesis/' + self.planet

        if 'reduced' in kwargs.keys():
            self.reduced = kwargs['reduced']
            self.reduced = self.reduced[0].lower()
            if self.reduced not in ['r', 'c', 'm', 'i', 'a']:
                print("Reduction status not recognized. Please enter 'raw', 'collapsed', 'molecfit', 'ifu', or 'all'.")
                while True:
                    self.reduced = input("Reduction Status?: ")
                    self.reduced = self.reduced[0].lower()
                    if self.reduced == '' or self.reduced not in ['r', 'c', 'm', 'i', 'a']:
                        print("Please enter 'raw', 'collapsed', or 'molecfit', 'ifu', or 'all'.")
                    else:
                        break
        else:
            while True:
                self.reduced = input("Reduction Status?: ")
                self.reduced = self.reduced[0].lower()
                if self.reduced == '' or self.reduced not in ['r', 'c', 'm', 'i', 'a']:
                    print("Please enter 'raw', 'collapsed', or 'molecfit', 'ifu', or 'all'.")
                else:
                    break

        self.predir = self.homedir + '/Data/PreMolecfit'
        self.postdir = self.homedir + '/Data/PostMolecfit'

        if 'bands' in kwargs.keys():
            self.bands = kwargs['bands']
        else:
            self.bands = input("Which bands? 'uvb', 'vis', 'nir': ")
        if type(self.bands) == str:
            self.bands = [self.bands]

        # Organize raw data
        if self.reduced == 'r' or self.reduced == 'a':
            # add special case for moon
            self.RawUinfo = {}
            self.RawVinfo = {}
            self.RawNinfo = {}
            us = glob.glob(self.predir + '/**/MOV*MERGE3D_DATA*_UVB.fits')
            self.RawUinfo = dict.fromkeys(us, [])
            vs = glob.glob(self.predir + '/**/MOV*MERGE3D_DATA*_VIS.fits')
            self.RawVinfo = dict.fromkeys(vs, [])
            ns = glob.glob(self.predir + '/**/MOV*MERGE3D_DATA*_NIR.fits')
            self.RawNinfo = dict.fromkeys(ns, [])

            # try:
            #     self.headers = Table.read('Headers.tex')
            # except IOError:
            #     sys.exit("Header info missing. Please run Create_HeaderTable in this directory.")

            # Order data into dictionaries by arm
            # for i, arm in enumerate(self.headers['ESO SEQ ARM']):
            #     if any(s in self.headers['Filenames'][i] for s in ['2D', 'onoff']):
            #         pass
            #     elif arm == 'UVB':
            #         if 'TEL' in self.headers['Filenames'][i]:
            #             self.Utelinfo[self.headers['Filenames'][i]] = self.header_info(i)
            #         else:
            #             self.RawUinfo[self.headers['Filenames'][i]] = self.header_info(i)
            #     elif arm == 'VIS':
            #         if 'TEL' in self.headers['Filenames'][i]:
            #             self.Vtelinfo[self.headers['Filenames'][i]] = self.header_info(i)
            #         else:
            #             self.RawVinfo[self.headers['Filenames'][i]] = self.header_info(i)
            #     elif arm == 'NIR':
            #         if 'TEL' in self.headers['Filenames'][i]:
            #             self.Ntelinfo[self.headers['Filenames'][i]] = self.header_info(i)
            #         else:
            #             self.RawNinfo[self.headers['Filenames'][i]] = self.header_info(i)
            self.rawkeys = list(self.RawUinfo.keys())
            self.rawkeys += list(self.RawVinfo.keys())
            self.rawkeys += list(self.RawNinfo.keys())
            self.RawUdata = {}
            self.RawVdata = {}
            self.RawNdata = {}

            print("Raw data info collected")

        # Organize collapsed data
        if self.reduced == 'c' or self.reduced == 'a':
            self.PreUinfo = {}
            self.PreVinfo = {}
            self.PreNinfo = {}

            us = glob.glob(self.predir + '/**/Median*_UVB.fits')
            self.PreUinfo = dict.fromkeys(us, [])
            vs = glob.glob(self.predir + '/**/Median*_VIS.fits')
            self.PreVinfo = dict.fromkeys(vs, [])
            ns = glob.glob(self.predir + '/**/Median*_NIR.fits')
            self.PreNinfo = dict.fromkeys(ns, [])

            self.prekeys = list(self.PreUinfo.keys())
            self.prekeys += list(self.PreVinfo.keys())
            self.prekeys += list(self.PreNinfo.keys())
            self.PreUdata = {}
            self.PreVdata = {}
            self.PreNdata = {}

            print("Collapsed data info collected")

        # Organize postmolecfit data
        if self.reduced == 'm' or self.reduced == 'a':
            self.PostUinfo = {}
            self.PostVinfo = {}
            self.PostNinfo = {}

            notdata = ['atm', 'tac', 'res', 'gui', 'TAC', 'TRA', 'VIS_fit']
            us = glob.glob(self.postdir + '/uvb/2020*fit.fits')
            dus = [f for f in us if not any(nd in f for nd in notdata)]
            self.PostUinfo = dict.fromkeys(dus, [])
            vs = glob.glob(self.postdir + '/vis/2020*fit.fits')
            dvs = [f for f in vs if not any(nd in f for nd in notdata)]
            self.PostVinfo = dict.fromkeys(dvs, [])
            ns = glob.glob(self.postdir + '/nir/2020*fit.fits')
            dns = [f for f in ns if not any(nd in f for nd in notdata)]
            self.PostNinfo = dict.fromkeys(dns, [])

            self.postkeys = list(self.PostUinfo.keys())
            self.postkeys += list(self.PostVinfo.keys())
            self.postkeys += list(self.PostNinfo.keys())
            self.PostUdata = {}
            self.PostVdata = {}
            self.PostNdata = {}

            print("Post Molecfit data info collected")

        # Organize corrected IFU data
        if self.reduced == 'i' or self.reduced == 'a':
            self.IfuUinfo = {}
            self.IfuVinfo = {}
            self.IfuNinfo = {}

            us = glob.glob(self.postdir + '/uvb/*IFU.fits')
            self.IfuUinfo = dict.fromkeys(us, [])
            vs = glob.glob(self.postdir + '/vis/*IFU.fits')
            self.IfuVinfo = dict.fromkeys(vs, [])
            ns = glob.glob(self.postdir + '/nir/*IFU.fits')
            self.IfuNinfo = dict.fromkeys(ns, [])

            self.ifukeys = list(self.IfuUinfo.keys())
            self.ifukeys += list(self.IfuVinfo.keys())
            self.ifukeys += list(self.IfuNinfo.keys())
            self.IfuUdata = {}
            self.IfuVdata = {}
            self.IfuNdata = {}

            print("Corrected IFU data info collected")

        # Retrieve header, wavelength, and data info from files
        self.Uwave = {}
        self.Vwave = {}
        self.Nwave = {}
        if self.reduced == 'r' or self.reduced == 'a':
            for k in self.rawkeys:
                with fits.open(k) as hdul:
                    data = hdul[0].data
                    data = data.tolist()
                    CRVAL3 = hdul[0].header['CRVAL3']
                    CDELT3 = hdul[0].header['CDELT3']
                    NAXIS3 = hdul[0].header['NAXIS3']
                    wave = [CRVAL3+CDELT3*i for i in range(NAXIS3)]
                    header = hdul[0].header
                if np.shape(data) != ():
                    if k in self.RawUinfo:
                        self.RawUinfo[k] = header
                        self.RawUdata[k] = data
                        self.Uwave[k] = wave
                    elif k in self.RawVinfo:
                        self.RawVinfo[k] = header
                        self.RawVdata[k] = data
                        self.Vwave[k] = wave
                    elif k in self.RawNinfo:
                        self.RawNinfo[k] = header
                        self.RawNdata[k] = data
                        self.Nwave[k] = wave
                else:
                    pass
            print("Raw data collected")

        if self.reduced == 'c' or self.reduced == 'a':
            for k in self.prekeys:
                with fits.open(k) as hdul:
                    data = hdul[0].data
                    CRVAL1 = hdul[0].header['CRVAL1']
                    CDELT1 = hdul[0].header['CDELT1']
                    NAXIS1 = hdul[0].header['NAXIS1']
                    wave = [CRVAL1+CDELT1*i for i in range(NAXIS1)]
                    header = hdul[0].header
                    if np.shape(data) != ():
                        if k in self.PreUinfo:
                            self.PreUinfo[k] = header
                            self.PreUdata[k] = data
                            self.Uwave[k] = wave
                        elif k in self.PreVinfo:
                            self.PreVinfo[k] = header
                            self.PreVdata[k] = data
                            self.Vwave[k] = wave
                        elif k in self.PreNinfo:
                            self.PreNinfo[k] = header
                            self.PreNdata[k] = data
                            self.Nwave[k] = wave
            print("Collapsed data collected")

        if self.reduced == 'm' or self.reduced == 'a':
            for k in self.postkeys:
                with fits.open(k) as hdul:
                    data = hdul[1].data["flux"]
                    wave = hdul[1].data["lambda"]
                if np.shape(data) != ():
                    if k in self.PostUinfo:
                        self.PostUdata[k] = data
                        self.Uwave[k] = wave
                    elif k in self.PostVinfo:
                        self.PostVdata[k] = data
                        self.Vwave[k] = wave
                    elif k in self.PostNinfo:
                        self.PostNdata[k] = data
                        self.Nwave[k] = wave
                else:
                    pass
            print("Post Molecfit data collected")

        if self.reduced == 'i' or self.reduced == 'a':
            for k in self.ifukeys:
                with fits.open(k) as hdul:
                    data = hdul[0].data
                    data = data.tolist()
                    CRVAL3 = hdul[0].header['CRVAL3']
                    CDELT3 = hdul[0].header['CDELT3']
                    NAXIS3 = hdul[0].header['NAXIS3']
                    wave = [CRVAL3+CDELT3*i for i in range(NAXIS3)]
                    header = hdul[0].header
                if np.shape(data) != ():
                    if k in self.IfuUinfo:
                        self.IfuUinfo[k] = header
                        self.IfuUdata[k] = data
                        self.Uwave[k] = wave
                    elif k in self.IfuVinfo:
                        self.IfuVinfo[k] = header
                        self.IfuVdata[k] = data
                        self.Vwave[k] = wave
                    elif k in self.IfuNinfo:
                        self.IfuNinfo[k] = header
                        self.IfuNdata[k] = data
                        self.Nwave[k] = wave
                else:
                    pass
            print("Corrected IFU data collected")

        # Identify regions with strong telluric absorption (note that valid elements must be set to False)
        # Create masks of the telluric regions to use in each band and defines colors for visible light regions
        # if applicable
        if 'uvb' in self.bands:
            self.Uwave0 = self.Uwave[list(self.Uwave.keys())[0]]
            if self.Uwave0[0] >= 5.0:
                self.Uwave0 = np.array(self.Uwave0)*0.001
            self.Utels = [[.367, .370], [.396, .399], [.409, .414], [.429, .432], [.478, .480]]
            self.Umsk = np.ones(len(self.Uwave0))
            for low, high in self.Utels:
                for i, w in enumerate(self.Uwave0):
                    if not low < w < high:
                        self.Umsk[i] = False
            self.Umsk[0:800] = True
            self.Umsk[-150:] = True

            self.Ucolors = [[.380, .450], [.450, .485], [.485, .500], [.500, 0.556]]
            self.Uedges = [[self.Uwave0[0], self.Uwave0[800]], [self.Uwave0[-150], self.Uwave0[-1]]]
            self.Ucolornames = ['mediumorchid', 'mediumblue', 'cyan', 'green']

        if 'vis' in self.bands:
            self.Vwave0 = self.Vwave[list(self.Vwave.keys())[0]]
            if self.Vwave0[0] >= 5.0:
                self.Vwave0 = np.array(self.Vwave0)*0.001
            self.Vtels = [[.645, .649], [.687, .690], [.731, .738], [.756, .766], [.843, .863], [.874, .876],
                          [.878, .881], [.888, .890], [.902, .905], [.795, .811], [.915, .920], [.924, .934],
                          [.980, .990], [.972, .978], [.944, .967]]
            self.Vmsk = np.ones(len(self.Vwave0))
            for low, high in self.Vtels:
                for i, w in enumerate(self.Vwave0):
                    if not low < w < high:
                        self.Vmsk[i] = False
            self.Vmsk[0:100] = True
            self.Vmsk[-300:] = True

            self.Vcolors = [[.556, .565], [.565, .590], [.590, .625], [.625, .740]]
            self.Vedges = [[self.Vwave0[0], self.Vwave0[100]], [self.Vwave0[-300], self.Vwave0[-1]]]
            self.Vcolornames = ['green', 'gold', 'orangered', 'red']

        if 'nir' in self.bands:
            self.Nwave0 = self.Nwave[list(self.Nwave.keys())[0]]
            if self.Nwave0[0] >= 5.0:
                self.Nwave0 = np.array(self.Nwave0)*0.001
            self.Ntels = [[1.058, 1.066], [1.137, 1.200], [1.350, 1.530], [1.805, 1.991], [2.004, 2.044], [2.053, 2.088]]
            self.Nmsk = np.ones(len(self.Nwave0))
            for low, high in self.Ntels:
                for i, w in enumerate(self.Nwave0):
                    if not low < w < high:
                        self.Nmsk[i] = False
            self.Nmsk[0:200] = True
            self.Nmsk[-500:] = True

            self.Ncolors = []
            self.Nedges = [[self.Nwave0[0], self.Nwave0[200]], [self.Nwave0[-500], self.Nwave0[-1]]]
            self.Ncolornames = []

        self.aspecs = []
        self.sdspecs = []
        self.rchis = []

        self.hchis = []
        self.mchis = []
        self.dchis = []

    def raw_analysis(self):

        # Determines what data to use depending on the user input
        for band in self.bands:
            if band == 'uvb':
                specs = self.PreUdata
                wave = self.Uwave
                wave0 = self.Uwave0
                msk = self.Umsk
            elif band == 'vis':
                specs = self.PreVdata
                wave= self.Vwave
                wave0 = self.Vwave0
                msk = self.Vmsk
            elif band == 'nir':
                specs = self.PreNdata
                wave = self.Nwave
                wave0 = self.Nwave0
                msk = self.Nmsk
            else:
                specs = {}
                wave = {}
                wave0 = []
                msk = []

            # Plots each spectra in the given band and region, along with the average and stdev
            w, h = plt.figaspect(0.3)
            fig = plt.figure(1, figsize=(w, h))
            gspec = gridspec.GridSpec(ncols=1, nrows=len(specs)+2)
            for num, fname in enumerate(specs.keys()):
                if num == 0:
                    ax0 = fig.add_subplot(gspec[num, 0])
                    spec = self.normalize_spec(msk, specs[fname])
                    ax0.plot(wave[fname], spec)
                else:
                    ax = fig.add_subplot(gspec[num, 0], sharex=ax0)
                    spec = self.normalize_spec(msk, specs[fname])
                    ax.plot(wave[fname], spec)
            avax = fig.add_subplot(gspec[len(specs), 0])
            avax.plot(wave0, self.aspecs[band], color='orchid')
            sdax = fig.add_subplot(gspec[len(specs)+1, 0])
            sdax.plot(wave0, self.sdspecs[band], color='darkorchid')

            # remove tick labels
            # force ylims equal to double check outlier
            fig.subplots_adjust(hspace=0)
            fig.suptitle(band)
            plt.show()

    def averages(self):
        # Calculates the averages and standard deviations for each band of a planet
        # Calculates the reduced chi squareds for each spectrum compared to the average for the
        # region it is designated to.

        # Creates the dictionaries for the averages and standard deviations for the
        # UVB, VIS, and NIR bands, and Highlands, Maria, and Darkside regions.
        if self.planet == "Moon":
            regions = [' Highlands', ' Maria', ' Darkside']
        else:
            regions = [' Planet Body']
        keys = []
        for band in self.bands:
            for region in regions:
                keys.append(band + region)
        self.aspecs = dict.fromkeys(keys)
        self.sdspecs = dict.fromkeys(keys)
        self.rchis = dict.fromkeys(keys)

        # Retrieves the spectra and mask information for each band
        for band in self.bands:
            if band == 'uvb':
                specs = self.Udata
                msk = self.Umsk
            elif band == 'vis':
                specs = self.Vdata
                msk = self.Vmsk
            elif band == 'nir':
                specs = self.Ndata
                msk = self.Nmsk
            else:
                specs = {}
                msk = []

            # Determines the region for the Moon spectra in each band as listed in their filenames
            hlist = []
            mlist = []
            dlist = []
            for f in list(specs.keys()):
                if 'high' in f:
                    hlist.append(f)
                elif 'mar' in f:
                    mlist.append(f)
                elif 'dark' in f:
                    dlist.append(f)
                else:
                    pass
            regions = []
            if hlist != []:
                regions.append(hlist)
            if mlist != []:
                regions.append(mlist)
            if dlist != []:
                regions.append(dlist)

            # Applies the mask to each spectra per band and region, normalizes the spectrum using the mask,
            # then compiles all normalized spectra into a list to find the average and standard deviation
            # per band and region.
            for region in regions:
                data = {}
                for fname in rspecs[band + region]:
                    data[fname] = self.normalize_spec(msk, specs[fname])
                aspecs = np.median([*data.values()], axis=0)
                sdspecs = np.std([*data.values()], axis=0)

                # Avoids a divide by zero error
                for ii, ss in enumerate(sdspecs):
                    if ss == 0.:
                        sdspecs[ii] = 0.1

                # Computes the chi squared for each spectrum compared to the average for the band and region
                # Then divides that by the degrees of fredom (length of spectrum) for the reduced chi squared
                rchis = []
                for fname in region:
                    spec = data[fname]
                    chi = chisquare((spec - aspecs) / sdspecs)
                    rchis.append(abs(chi[0]) / len(spec))

                # Saves everything for future use
                self.aspecs[region + band] = aspecs
                self.sdspecs[region + band] = sdspecs
                self.rchis[region + band] = rchis

            # Applies the mask to each spectra per band, normalizes the spectrum using the mask,
            # then compiles all normalized spectra into a list to find the average and standard deviation
            # per band and region.
            data = {}
            for fname in specs:
                data[fname] = self.normalize_spec(msk, specs[fname])
            aspecs = np.median([*data.values()], axis=0)
            sdspecs = np.std([*data.values()], axis=0)

            # Avoids a divide by zero error
            for ii, ss in enumerate(sdspecs):
                if ss == 0.:
                    sdspecs[ii] = 0.1

            # Computes the chi squared for each spectrum compared to the average for the band and region
            # Then divides that by the degrees of fredom (length of spectrum) for the reduced chi squared
            rchis = []
            for fname in specs:
                spec = data[fname]
                chi = chisquare((spec - aspecs) / sdspecs)
                rchis.append(abs(chi[0]) / len(spec))

            # Saves everything for future use
            self.aspecs[band] = aspecs
            self.sdspecs[band] = sdspecs
            self.rchis[band] = rchis

    def sorting_hat(self, **kwargs):

        # Sorts spectra into regions based off of the similarities between them

        # Sets a maximum value for the reduced chi squared of a spectrum to be included in the existing regions
        # rather than becoming an independant region
        if 'rchi_max' in kwargs.keys():
            rchi_max = kwargs['rchi_max']
        else:
            rchi_max = 7.0

        # Retrieves the spectra and mask information for each band
        for band in self.bands:
            if band == 'uvb':
                specs = self.PostUdata
                wave = self.Uwave
                wave0 = self.Uwave0
                msk = self.Umsk
                tels = self.Utels
                colors = self.Ucolors
                colornames = self.Ucolornames
            elif band == 'vis':
                specs = self.PostVdata
                wave= self.Vwave
                wave0 = self.Vwave0
                msk = self.Vmsk
                tels = self.Vtels
                colors = self.Vcolors
                colornames = self.Vcolornames
            elif band == 'nir':
                specs = self.PostNdata
                wave = self.Nwave
                wave0 = self.Nwave0
                msk = self.Nmsk
                tels = self.Ntels
                colors = self.Ncolors
                colornames = self.Ncolornames
            else:
                specs = {}
                wave = {}
                wave0 = []
                msk = []
                tels = []
                colors = []
                colornames = []

            # Compares the rchis of one spectrum to all region averages to determine if the spectrum belongs in
            # any existing classification or if it should become it's own region based off of the maximum allowed
            # rchi value.
            speckeys = sorted(list(specs.keys()))
            self.regions = {}
            self.regions['Region_0'] = [speckeys[0]]
            rnum = 1
            for fname in speckeys[1:]:
                rchis = []
                for region in self.regions:
                    rspecs = []
                    for rs in self.regions[region]:
                        rss = self.normalize_spec(msk, specs[rs])
                        rspecs.append(rss)
                    med = np.median(rspecs)
                    stdev = np.std(rspecs)
                    spec = self.normalize_spec(msk, specs[fname])
                    chai = chisquare((spec - med) / stdev)
                    rchis.append(abs(chai[0]) / len(spec))
                best = np.min(rchis)
                if best <= rchi_max:
                    regnum = rchis.index(best)
                    self.regions['Region_' + str(regnum)].append(fname)
                else:
                    self.regions['Region_' + str(rnum)] = [fname]
                    rnum += 1

            print(band.upper() + ' band sorted into ' + str(rnum) + ' regions.')

    def averages_moon(self):

        # Calculates the averages and standard deviations for each region and band of the Moon
        # Calculates the reduced chi squareds for each spectrum compared to the average for the
        # region it is designated to.

        # Creates the dictionaries for the averages and standard deviations for the
        # UVB, VIS, and NIR bands, and Highland, Maria, and Darkside regions.
        keys = ['huvb', 'muvb', 'duvb', 'hvis', 'mvis', 'dvis', 'hnir', 'mnir']
        self.aspecs = dict.fromkeys(keys)
        self.sdspecs = dict.fromkeys(keys)
        self.rchis = dict.fromkeys(keys)

        # Retrieves the spectra and mask information for each band
        for band in self.bands:
            if band == 'uvb':
                specs = self.PostUdata
                msk = self.Umsk
            elif band == 'vis':
                specs = self.PostVdata
                msk = self.Vmsk
            elif band == 'nir':
                specs = self.PostNdata
                msk = self.Nmsk
            else:
                specs = {}
                msk = []

            # Determines the region for the spectra in each band as listed in their filenames
            hlist = []
            mlist = []
            dlist = []
            for f in list(specs.keys()):
                if 'high' in f:
                    hlist.append(f)
                elif 'mar' in f:
                    mlist.append(f)
                elif 'dark' in f:
                    dlist.append(f)
                else:
                    pass
            regions = []
            if hlist != []:
                regions.append(hlist)
            if mlist != []:
                regions.append(mlist)
            if dlist != []:
                regions.append(dlist)

            # Applies the mask to each spectra per band and region, normalizes the spectrum using the mask,
            # then compiles all normalized spectra into a list to find the average and standard deviation
            # per band and region.
            for li in regions:
                data = {}
                for fname in li:
                    data[fname] = self.normalize_spec(msk, specs[fname])
                aspecs = np.median([*data.values()], axis=0)
                sdspecs = np.std([*data.values()], axis=0)

                # Avoids a divide by zero error
                for ii, ss in enumerate(sdspecs):
                    if ss == 0.:
                        sdspecs[ii] = 0.1

                # Computes the chi squared for each spectrum compared to the average for the band and region
                # Then divides that by the degrees of fredom (length of spectrum) for the reduced chi squared
                rchis = []
                for fname in li:
                    spec = data[fname]
                    chi = chisquare((spec - aspecs) / sdspecs)
                    rchis.append(abs(chi[0]) / len(spec))

                # Saves everything for future use
                if li == hlist:
                    let = 'h'
                elif li == mlist:
                    let = 'm'
                else:
                    let = 'd'
                self.aspecs[let + band] = aspecs
                self.sdspecs[let + band] = sdspecs
                self.rchis[let + band] = rchis

    def averages_planet(self):

        # Calculates the averages and standard deviations for each band of a planet
        # Calculates the reduced chi squareds for each spectrum compared to the average for the
        # region it is designated to.

        # Creates the dictionaries for the averages and standard deviations for the
        # UVB, VIS, and NIR bands.
        keys = self.bands
        self.aspecs = dict.fromkeys(keys)
        self.sdspecs = dict.fromkeys(keys)
        self.rchis = dict.fromkeys(keys)

        # Retrieves the spectra and mask information for each band
        for band in self.bands:
            if band == 'uvb':
                specs = self.PostUdata
                msk = self.Umsk
            elif band == 'vis':
                specs = self.PostVdata
                msk = self.Vmsk
            elif band == 'nir':
                specs = self.PostNdata
                msk = self.Nmsk
            else:
                specs = {}
                msk = []

            # Applies the mask to each spectra per band, normalizes the spectrum using the mask,
            # then compiles all normalized spectra into a list to find the average and standard deviation
            # per band and region.
            data = {}
            for fname in specs:
                data[fname] = self.normalize_spec(msk, specs[fname])
            aspecs = np.median([*data.values()], axis=0)
            sdspecs = np.std([*data.values()], axis=0)

            # Avoids a divide by zero error
            for ii, ss in enumerate(sdspecs):
                if ss == 0.:
                    sdspecs[ii] = 0.1

            # Computes the chi squared for each spectrum compared to the average for the band and region
            # Then divides that by the degrees of fredom (length of spectrum) for the reduced chi squared
            rchis = []
            for fname in specs:
                spec = data[fname]
                chi = chisquare((spec - aspecs) / sdspecs)
                rchis.append(abs(chi[0]) / len(spec))

            # Saves everything for future use
            self.aspecs[band] = aspecs
            self.sdspecs[band] = sdspecs
            self.rchis[band] = rchis

    def plot_averages_planet(self, *args):

        # Plots the averages of the different bands for a given planet.
        # Includes an option for boxcar smoothing.
        # Highlights regions of strong telluric absorption, and regions masked from the average calculation
        # Saves figure.
        w, h = plt.figaspect(0.3)
        fig1 = plt.figure(1, figsize=(w, h))
        ax = fig1.add_subplot(111)
        for band in self.bands:
            if band == 'uvb':
                wave0 = self.Uwave0
                tels = self.Utels
                speccolor = 'royalblue'
                colors = self.Ucolors
                colornames = self.Ucolornames
                edges = self.Uedges
            elif band == 'vis':
                wave0 = self.Vwave0
                tels = self.Vtels
                speccolor = 'forestgreen'
                colors = self.Vcolors
                colornames = self.Vcolornames
                edges = self.Vedges
            elif band == 'nir':
                wave0 = self.Nwave0
                tels = self.Ntels
                speccolor = 'rebeccapurple'
                colors = self.Ncolors
                colornames = self.Ncolornames
                edges = self.Nedges
            else:
                wave0 = []
                tels = []
                speccolor = ''
                colors = []
                colornames = []
                edges = []

            ax.plot(wave0, self.aspecs[band], color=speccolor, label=band)
            for t in tels:
                ax.axvspan(t[0], t[1], facecolor='olivedrab', alpha=0.4)
            for e in edges:
                ax.axvspan(e[0], e[1], facecolor='firebrick', alpha=0.4)
            if 'colors' in args:
                for i, c in enumerate(colors):
                    ax.axvspan(c[0], c[1], facecolor=colornames[i], alpha=0.4)

        ax.legend()
        ax.set_title(self.planet + ' Averages')
        ax.set_xlabel('Wavelength (microns)')
        ax.set_ylabel('Normalized Flux')
        ax.set_ylim(bottom=0.0, top=1.0)

        plt.savefig('/home/gmansir/Thesis/'+self.planet+'/average_plot.png')
        plt.show()

    def plot_averages_moon(self, *args):

        # Plots the averages of the different regions of the moon.
        # Includes an option for boxcar smoothing.
        # Highlights regions of strong telluric absorption, and regions masked from the average calculation
        # Saves figure.

        w, h = plt.figaspect(0.3)
        fig1 = plt.figure(1, figsize=(w, h))
        ax = fig1.add_subplot(111)
        if 'uvb' in self.bands:
            udarf = self.aspecs['duvb']
            umarf = self.aspecs['muvb']
            uhigf = self.aspecs['huvb']
            ax.plot(self.Uwave0, udarf, color='midnightblue', label='Darkside (Maria)')
            ax.plot(self.Uwave0, umarf, color='royalblue', label='Maria')
            ax.plot(self.Uwave0, uhigf, color='powderblue', label='Highlands')
            for t in self.Utels:
                ax.axvspan(t[0], t[1], facecolor='olivedrab', alpha=0.4)
            for e in self.Uedges:
                ax.axvspan(e[0], e[1], facecolor='firebrick', alpha=0.4)
            if 'colors' in args:
                for i, c in enumerate(self.Ucolors):
                    ax.axvspan(c[0], c[1], facecolor=self.Ucolornames[i], alpha=0.4)
        if 'vis' in self.bands:
            vdarf = self.aspecs['dvis']
            vmarf = self.aspecs['mvis']
            vhigf = self.aspecs['hvis']
            ax.plot(self.Vwave0, vdarf, color='midnightblue')
            ax.plot(self.Vwave0, vmarf, color='royalblue')
            ax.plot(self.Vwave0, vhigf, color='powderblue')
            for t in self.Vtels:
                ax.axvspan(t[0], t[1], facecolor='olivedrab', alpha=0.4)
            for e in self.Vedges:
                ax.axvspan(e[0], e[1], facecolor='firebrick', alpha=0.4)
            if 'colors' in args:
                for i, c in enumerate(self.Vcolors):
                    ax.axvspan(c[0], c[1], facecolor=self.Vcolornames[i], alpha=0.4)
        if 'nir' in self.bands:
            nmarf = self.aspecs['mnir']
            nhigf = self.aspecs['hnir']
            ax.plot(self.Nwave0, nmarf, color='royalblue')
            ax.plot(self.Nwave0, nhigf, color='powderblue')
            for t in self.Ntels:
                ax.axvspan(t[0], t[1], facecolor='olivedrab', alpha=0.4)
            for e in self.Nedges:
                ax.axvspan(e[0], e[1], facecolor='firebrick', alpha=0.4)
        ax.set_ylim(bottom=0, top=1)
        ax.legend()
        ax.set_title('Moon Region Averages')
        ax.set_xlabel('Wavelength (microns)')
        ax.set_ylabel('Normalized Flux')

        plt.savefig('/home/gmansir/Thesis/Moon/aspec_plot.png')
        plt.show()

    def remove_bad_specs_moon(self):

        # Retrieves the spectra and mask information for each band
        for band in self.bands:
            if band == 'uvb':
                specs = self.PostUdata
                msk = self.Umsk
            elif band == 'vis':
                specs = self.PostVdata
                msk = self.Vmsk
            elif band == 'nir':
                specs = self.PostNdata
                msk = self.Nmsk
            else:
                specs = {}
                msk = []

            # Determines the region for the spectra in each band as listed in their filenames
            hlist = []
            mlist = []
            dlist = []
            for f in specs.keys():
                if 'high' in f:
                    hlist.append(f)
                elif 'mar' in f:
                    mlist.append(f)
                elif 'dark' in f:
                    dlist.append(f)
                else:
                    pass

            # Checks if a spectrum's rchi is more than 2 standard deviations away from the average in that
            # region. If it is, then it removes that spectrum from future analysis.
            for let in ['h', 'm', 'd']:
                key = let + band
                lst = let + 'list'
                data = 'self.' + key[1].upper() + 'data'
                bad = []
                if self.rchis[key] != None:
                    achi = np.mean(self.rchis[key])
                    sdchi = np.std(self.rchis[key])
                    hchi = achi + sdchi * 2
                    lchi = achi - sdchi * 2
                    for i, chi in enumerate(self.rchis[key]):
                        if hchi < chi or lchi > chi:
                            bad.append(locals()[lst][i])
                            print('Removed ', locals()[lst][i], ' from data. Average: ', achi, ', Spec: ', chi, '.')
                    for b in bad:
                        locals()[lst].remove(b)
                        eval(data).pop(b, None)

            # Reapplies the mask to each spectra per band and region, normalizes the spectrum using the mask,
            # then compiles all normalized spectra into an updated list to find the average and standard deviation
            # per band and region with the outliers removed.
            data = dict.fromkeys(list(specs.keys()))
            for li in [hlist, mlist, dlist]:
                for fname in li:
                    data[fname] = self.normalize_spec(msk, specs[fname])
                aspecs = np.median(data.values(), axis=0)
                sdspecs = np.std(data.values(), axis=0)

                # Avoids a divide by zero error
                for ii, ss in enumerate(sdspecs):
                    if ss == 0.:
                        sdspecs[ii] = 0.1

                # Computes the chi squared for each spectrum compared to the average for the band and region
                # Then divides that by the degrees of fredom (length of spectrum) for the reduced chi squared                rchis = []
                rchis = []
                for fname in li:
                    spec = specs[fname]
                    chi = chisquare((spec - aspecs) / sdspecs)
                    rchis.append(abs(chi[0]) / len(spec))

                # Saves everything for future use
                if li == hlist:
                    let = 'h'
                elif li == mlist:
                    let = 'm'
                else:
                    let = 'd'
                self.aspecs[let + band] = aspecs
                self.sdspecs[let + band] = sdspecs
                self.rchis[let + band] = rchis

    def rchi_compare_moon(self):

        # Compares each spectrum to the average of each region to help determine if the
        # spectrum is sorted properly using the filename.

        self.hchis = []
        self.mchis = []
        self.dchis = []

        # Retrieves the spectra and mask information for each band
        for band in self.bands:
            if band == 'uvb':
                specs = self.PostUdata
                msk = self.Umsk
            elif band == 'vis':
                specs = self.PostVdata
                msk = self.Vmsk
            elif band == 'nir':
                specs = self.PostNdata
                msk = self.Nmsk
            else:
                specs = {}
                msk = []

            # Determines the region for the spectra in each band as listed in their filenames
            hlist = []
            mlist = []
            dlist = []
            for f in specs.keys():
                if 'high' in f:
                    hlist.append(f)
                elif 'mar' in f:
                    mlist.append(f)
                elif 'dark' in f:
                    dlist.append(f)
                else:
                    pass

            # Computes the reduced chi squared of each spectrum vs the average of each region in a given band.
            for li in [hlist, mlist, dlist]:
                for fname in li:
                    ms = ma.masked_array(specs[fname], mask=msk)
                    spec = specs[fname] / ms.mean()
                    hc = chisquare((spec - self.aspecs['h' + band]) / self.sdspecs['h' + band])
                    self.hchis.append(abs(hc[0]) / len(spec))
                    mc = chisquare((spec - self.aspecs['m' + band]) / self.sdspecs['m' + band])
                    self.mchis.append(abs(mc[0]) / len(spec))
                    dc = chisquare((spec - self.aspecs['d' + band]) / self.sdspecs['d' + band])
                    self.dchis.append(abs(dc[0]) / len(spec))

    def plot_rchis_moon(self):

        # Plots a bar graph showing the rchis of each spectrum compared to each region average
        # as computed in self.moon_rchi_compare()

        w, h = plt.figaspect(0.9)
        fig = plt.figure(1, figsize=(w, h))
        ax = fig.add_subplot(111)
        bwidth = 0.3

        hax = ax.bar(np.arange(0, len(self.hchis)) - bwidth, self.hchis, bwidth, color='powderblue', label='Highlands')
        max = ax.bar(np.arange(0, len(self.mchis)), self.mchis, bwidth, color='royalblue', label='Maria')
        dax = ax.bar(np.arange(0, len(self.dchis)) + bwidth, self.dchis, bwidth, color='midnightblue', label='Darkside')

        ax.axvspan(0, 20, facecolor='green', alpha=0.1)
        ax.axvspan(59, 65, facecolor='green', alpha=0.1)
        ax.axvspan(76, 80, facecolor='green', alpha=0.1)
        ax.axvspan(20, 40, facecolor='turquoise', alpha=0.1)
        ax.axvspan(65, 71, facecolor='turquoise', alpha=0.1)
        ax.axvspan(80, 84, facecolor='turquoise', alpha=0.1)
        ax.axvspan(40, 59, facecolor='purple', alpha=0.1)
        ax.axvspan(71, 76, facecolor='purple', alpha=0.1)
        ax.axvspan(84, 88, facecolor='purple', alpha=0.1)

        ax.set_xlabel('Frame', fontsize=14)
        ax.set_ylabel('Reduced Chi Squared', fontsize=14)
        ax.legend()

        plt.savefig('/home/gmansir/Thesis/Moon/rchis_plot.png')
        plt.show()

    def plot_all_planet(self):

        # Plots all normalized spectra in each band, alongside the average and stdev

        # Determines what data to use depending on the user input
        for band in self.bands:
            if band == 'uvb':
                specs = self.PostUdata
                wave = self.Uwave
                wave0 = self.Uwave0
                msk = self.Umsk
                tels = self.Utels
                colors = self.Ucolors
                colornames = self.Ucolornames
                edges = self.Uedges
            elif band == 'vis':
                specs = self.PostVdata
                wave= self.Vwave
                wave0 = self.Vwave0
                msk = self.Vmsk
                tels = self.Vtels
                colors = self.Vcolors
                colornames = self.Vcolornames
                edges = self.Vedges
            elif band == 'nir':
                specs = self.PostNdata
                wave = self.Nwave
                wave0 = self.Nwave0
                msk = self.Nmsk
                tels = self.Ntels
                colors = self.Ncolors
                colornames = self.Ncolornames
                edges = self.Nedges
            else:
                specs = {}
                wave = {}
                wave0 = []
                msk = []
                tels = []
                colors = []
                colornames = []
                edges = []

            # Plots each spectra in the given band and region, along with the average and stdev
            w, h = plt.figaspect(0.5)
            fig = plt.figure(1, figsize=(w, h))
            gspec = gridspec.GridSpec(len(specs)+2, 1)
            fig.tight_layout()
            for num, fname in enumerate(specs):
                ax = fig.add_subplot(gspec[num, 0])
                spec = self.normalize_spec(msk, specs[fname])
                ax.plot(wave[fname], spec)
                ax.set_ylim(bottom=0, top=1)
                for t in tels:
                    ax.axvspan(t[0], t[1], facecolor='olivedrab', alpha=0.4)
                for e in edges:
                    ax.axvspan(e[0], e[1], facecolor='firebrick', alpha=0.4)
                for idx, c in enumerate(colors):
                    ax.axvspan(c[0], c[1], facecolor=colornames[idx], alpha=0.4)
                ax.tick_params(bottom=False, labelbottom=False)
            avax = fig.add_subplot(gspec[len(specs), 0])
            avax.plot(wave0, self.aspecs[band], color='orchid')
            avax.set_ylim(bottom=0, top=1)
            sdax = fig.add_subplot(gspec[len(specs)+1, 0])
            sdax.plot(wave0, self.sdspecs[band], color='darkorchid')
            sdax.set_ylim(bottom=0, top=1)
            fig.suptitle(band.upper())
            plt.show()

    def plot_all_moon(self):

        # Plots all normalized spectra in each band and region, alongside the average and stdev

        # Determines what data to use depending on the user input
        for band in self.bands:
            if band == 'uvb':
                specs = self.PostUdata
                wave = self.Uwave
                wave0 = self.Uwave0
                msk = self.Umsk
            elif band == 'vis':
                specs = self.PostVdata
                wave = self.Vwave
                wave0 = self.Vwave0
                msk = self.Vmsk
            elif band == 'nir':
                specs = self.PostNdata
                wave = self.Nwave
                wave0 = self.Nwave0
                msk = self.Nmsk
            else:
                specs = {}
                wave = {}
                wave0 = []
                msk = []

            # Determines the region for the spectra in each band as listed in their filenames
            hlist = []
            mlist = []
            dlist = []
            for f in list(specs.keys()):
                if 'high' in f:
                    hlist.append(f)
                elif 'mar' in f:
                    mlist.append(f)
                elif 'dark' in f:
                    dlist.append(f)
                else:
                    pass
            regions = []
            if hlist != []:
                regions.append(hlist)
            if mlist != []:
                regions.append(mlist)
            if dlist != []:
                regions.append(dlist)

            # Figures out which region of the Moon we are looking at for the user
            for r in regions:
                if r == hlist:
                    let = 'h'
                elif r == mlist:
                    let = 'm'
                else:
                    let = 'd'

            # Plots each spectra in the given band and region, along with the average and stdev
                fig = plt.figure()
                gspec = gridspec.GridSpec(ncols=1, nrows=len(r)+2)
                for num, fname in enumerate(r):
                    if num == 0:
                        ax0 = fig.add_subplot(gspec[num, 0])
                        spec = self.normalize_spec(msk, specs[fname])
                        ax0.plot(wave[fname], spec)
                    else:
                        ax = fig.add_subplot(gspec[num, 0], sharex=ax0)
                        spec = self.normalize_spec(msk, specs[fname])
                        ax.plot(wave[fname], spec)
                    avax = fig.add_subplot(gspec[len(r), 0])
                    avax.plot(wave0, self.aspecs[let + band], color='orchid')
                    sdax = fig.add_subplot(gspec[len(r)+1, 0])
                    sdax.plot(wave0, self.sdspecs[let + band], color='darkorchid')
                # remove tick labels
                # force ylims equal to double check outlier
                fig.subplots_adjust(hspace=0)
                fig.suptitle(let+band)
                plt.show()

    def plot_airmass(self):

        self.times = {}
        self.airmasses = {}
        airmasses = []
        times = []
        for k in self.PreUinfo.keys():
            air = self.PreUinfo[k][-2:]
            airavg = (air[0] + air[1]) / 2
            airmasses.append(airavg)
            t = self.PreUinfo[k][0]
            tsec = int(t[-2:]) / 60
            tmin = (int(t[-5:-3]) + tsec) / 60
            thour = int(t[-8:-6]) + tmin
            if thour < 10:
                thour += 24
            times.append(thour)
        plt.plot(times, airmasses, 'o', c='green', label='UVB Moon')
        self.times['uvb'] = times
        self.airmasses['uvb'] = airmasses
        airmasses = []
        times = []
        for k in self.PreVinfo.keys():
            air = self.PreVinfo[k][-2:]
            airavg = (air[0] + air[1]) / 2
            airmasses.append(airavg)
            t = self.PreVinfo[k][0]
            tsec = int(t[-2:]) / 60
            tmin = (int(t[-5:-3]) + tsec) / 60
            thour = int(t[-8:-6]) + tmin
            if thour < 10:
                thour += 24
            times.append(thour)
        plt.plot(times, airmasses, 'o', c='blue', label='VIS Moon')
        self.times['vis'] = times
        self.airmasses['vis'] = airmasses
        airmasses = []
        times = []
        for k in self.PreNinfo.keys():
            air = self.PreNinfo[k][-2:]
            airavg = (air[0] + air[1]) / 2
            airmasses.append(airavg)
            t = self.PreNinfo[k][0]
            tsec = int(t[-2:]) / 60
            tmin = (int(t[-5:-3]) + tsec) / 60
            thour = int(t[-8:-6]) + tmin
            if thour < 10:
                thour += 24
            times.append(thour)
        plt.plot(times, airmasses, 'o', c='purple', label='NIR Moon')
        self.times['nir'] = times
        self.airmasses['nir'] = airmasses
        airmasses = []
        times = []
        for k in self.Utelinfo.keys():
            air = self.Utelinfo[k][-2:]
            airavg = (air[0] + air[1]) / 2
            airmasses.append(airavg)
            t = self.Utelinfo[k][0]
            tsec = int(t[-2:]) / 60
            tmin = (int(t[-5:-3]) + tsec) / 60
            thour = int(t[-8:-6]) + tmin
            if thour < 10:
                thour += 24
            times.append(thour)
        plt.plot(times, airmasses, 'o', c='lightgreen', label='UVB Tel')
        airmasses = []
        times = []
        for k in self.Vtelinfo.keys():
            air = self.Vtelinfo[k][-2:]
            airavg = (air[0] + air[1]) / 2
            airmasses.append(airavg)
            t = self.Vtelinfo[k][0]
            tsec = int(t[-2:]) / 60
            tmin = (int(t[-5:-3]) + tsec) / 60
            thour = int(t[-8:-6]) + tmin
            if thour < 10:
                thour += 24
            times.append(thour)
        plt.plot(times, airmasses, 'o', c='lightblue', label='VIS Tel')
        airmasses = []
        times = []
        for k in self.Ntelinfo.keys():
            air = self.Ntelinfo[k][-2:]
            airavg = (air[0] + air[1]) / 2
            airmasses.append(airavg)
            t = self.Ntelinfo[k][0]
            tsec = int(t[-2:]) / 60
            tmin = (int(t[-5:-3]) + tsec) / 60
            thour = int(t[-8:-6]) + tmin
            if thour < 10:
                thour += 24
            times.append(thour)
            if airavg <= 1.6:
                print(k)
        plt.plot(times, airmasses, 'o', c='orchid', label='NIR Tel')
        plt.xlabel('Time (hours)')
        plt.ylabel('Airmass')
        plt.title('Airmass v Time')
        plt.legend(loc='upper left')
        plt.savefig('Airmass' + '.png')
        plt.show()

    def plot_times(self):

        hUs = []
        mUs = []
        dUs = []
        hVs = []
        mVs = []
        dVs = []
        hNs = []
        mNs = []
        dNs = []
        Usky = []
        Vsky = []
        Nsky = []

        openf = open('/home/gmansir/Thesis/Moon/fhv2.txt')
        ff = openf.readlines()

        for f in ff:
            ff[ff.index(f)] = [f[:36], f[37:].split()]

        for f in ff:
            if f[1][0] == 'BJECT':
                if f[1][1] == 'UVB':
                    if 'high' in f[1][3]:
                        hUs.append(float(f[1][2]))
                    elif 'mar' in f[1][3]:
                        mUs.append(float(f[1][2]))
                    elif 'dark' in f[1][3]:
                        dUs.append(float(f[1][2]))
                elif f[1][1] == 'VIS':
                    if 'high' in f[1][3]:
                        hVs.append(float(f[1][2]))
                    elif 'mar' in f[1][3]:
                        mVs.append(float(f[1][2]))
                    elif 'dark' in f[1][3]:
                        dVs.append(float(f[1][2]))
                elif f[1][1] == 'NIR':
                    if 'high' in f[1][3]:
                        hNs.append(float(f[1][2]))
                    elif 'mar' in f[1][3]:
                        mNs.append(float(f[1][2]))
                    elif 'dark' in f[1][3]:
                        dNs.append(float(f[1][2]))
            elif f[1][0] == 'KY':
                if f[1][1] == 'UVB':
                    Usky.append(float(f[1][2]))
                elif f[1][1] == 'VIS':
                    Vsky.append(float(f[1][2]))
                elif f[1][1] == 'NIR':
                    Nsky.append(float(f[1][2]))

        RUs = []
        RVs = []
        RNs = []

        for h in self.PreUinfo.keys():
            RUs.append(m.UVBs[h][0])
        for i in self.PreVinfo.keys():
            RVs.append(m.VISs[i][0])
        for j in self.PreNinfo.keys():
            RNs.append(m.NIRs[j][0])

        RUs = Time(RUs)
        RVs = Time(RVs)
        RNs = Time(RNs)

        offset = RUs.mjd[0] - hUs[0]

        plt.plot(hUs, np.repeat('UVB Raw', len(hUs)), 'o', color='lightgreen')
        plt.plot(hVs, np.repeat('VIS Raw', len(hVs)), 'o', color='lightblue')
        plt.plot(hNs, np.repeat('NIR Raw', len(hNs)), 'o', color='plum')
        plt.plot(mUs, np.repeat('UVB Raw', len(mUs)), 'o', color='green')
        plt.plot(mVs, np.repeat('VIS Raw', len(mVs)), 'o', color='blue')
        plt.plot(mNs, np.repeat('NIR Raw', len(mNs)), 'o', color='orchid')
        plt.plot(dUs, np.repeat('UVB Raw', len(dUs)), 'o', color='darkgreen')
        plt.plot(dVs, np.repeat('VIS Raw', len(dVs)), 'o', color='darkblue')
        plt.plot(dNs, np.repeat('NIR Raw', len(dNs)), 'o', color='darkorchid')
        plt.plot(Usky, np.repeat('UVB Raw', len(Usky)), 'o', color='lime')
        plt.plot(Vsky, np.repeat('VIS Raw', len(Vsky)), 'o', color='aquamarine')
        plt.plot(Nsky, np.repeat('NIR Raw', len(Nsky)), 'o', color='thistle')
        #        plt.plot(RUs.mjd-offset, np.repeat('UVB Reduced', len(RUs)), 'o', color='green')
        #        plt.plot(RVs.mjd-offset, np.repeat('VIS Reduced', len(RVs)), 'o', color='blue')
        #        plt.plot(RNs.mjd-offset, np.repeat('NIR Reduced', len(RNs)), 'o', color='purple')
        plt.savefig('timing.png', overwrite=True)
        plt.show()

    def plot_molecules(self):

        hlist = ['2020_highlands__1_fit.atm', '2020_highlands__2_fit.atm', '2020_highlands__3_fit.atm',
                 '2020_highlands__4_fit.atm', '2020_highlands__5_fit.atm', '2020_highlands__6_fit.atm',
                 '2020_highlands__7_fit.atm', '2020_highlands__8_fit.atm', '2020_highlands__9_fit.atm',
                 '2020_highlands__10_fit.atm', '2020_highlands__11_fit.atm', '2020_highlands__12_fit.atm',
                 '2020_highlands__13_fit.atm', '2020_highlands__14_fit.atm', '2020_highlands__15_fit.atm',
                 '2020_highlands__16_fit.atm', '2020_highlands__17_fit.atm', '2020_highlands__18_fit.atm',
                 '2020_highlands__19_fit.atm', '2020_highlands__20_fit.atm']

        mlist = ['2020_maria__1_fit.atm', '2020_maria__2_fit.atm', '2020_maria__3_fit.atm',
                 '2020_maria__4_fit.atm', '2020_maria__5_fit.atm', '2020_maria__6_fit.atm',
                 '2020_maria__7_fit.atm']

        dlist = ['2020_darkside__1_fit.atm', '2020_darkside__2_fit.atm', '2020_darkside__3_fit.atm',
                 '2020_darkside__4_fit.atm']

        hgt = dict((k, []) for k in ['uvb', 'vis', 'nir'])
        water = dict((k, []) for k in ['uvb', 'vis', 'nir'])
        co2 = dict((k, []) for k in ['uvb', 'vis', 'nir'])
        methane = dict((k, []) for k in ['uvb', 'vis', 'nir'])
        for li in [hlist, mlist, dlist]:
            for band in ['uvb', 'vis', 'nir']:
                for l in li:
                    if band == 'uvb':
                        l = 'Molecfit/UVB/' + l
                    elif band == 'vis':
                        l = 'Molecfit/VIS/' + l
                    else:
                        l = 'Molecfit/NIR/' + l
                        dlist = []
                    if li == hlist:
                        l = l[:28] + band + l[28:]
                    elif li == mlist:
                        l = l[:24] + band + l[24:]
                    else:
                        l = l[:27] + band + l[27:]
                    openf = open(l, 'r')
                    contents = openf.readlines()
                    hgt[band].append(contents[3:13])
                    water[band].append(contents[36:46])
                    co2[band].append(contents[47:57])
                    methane[band].append(contents[58:68])
                    openf.close()

                    for i in range(len(water[band])):
                        #                       hs = []
                        ws = []
                        cs = []
                        ms = []
                        #                       for j in range(len(hgt[band][i])):
                        #                           h = hgt[band][i][j].split()
                        #                           [hs.append(float(y)) for y in h]
                        #                       hgt[band][i] = hs
                        for j in range(len(water[band][i])):
                            w = water[band][i][j].split()
                            [ws.append(float(x)) for x in w]
                        water[band][i] = ws
                        for j in range(len(co2[band][i])):
                            c = co2[band][i][j].split()
                            [cs.append(float(x)) for x in c]
                        co2[band][i] = cs
                        for j in range(len(methane[band][i])):
                            m = methane[band][i][j].split()
                            [ms.append(float(x)) for x in m]
                        methane[band][i] = ms

                    if len(self.airmasses[band]) != len(water[band][0]):
                        air = self.airmasses[band]
                        for i in range(len(water[band][0]) - len(self.airmasses[band])):
                            air.append(self.airmasses[band][-1])
                    else:
                        air = self.airmasses[band]
                    if len(self.times[band]) != len(water[band][0]):
                        t = self.times[band]
                        for i in range(len(water[band][0]) - len(self.times[band])):
                            diff = self.times[band][-1] - self.times[band][-2]
                            t.append(t[-1] + diff)
                    else:
                        t = self.airmasses[band]

                    Z = np.array([air, t])

                    pdb.set_trace()

                    figure = plt.figure()
                    ax = plt.axes(projection='3d')
                    ax.plot_surface(np.array(water[band])[0], np.array(co2[band])[0], Z, cmap='viridis',
                                    edgecolor='none')
                    ax.set_title('water vs co2')
                    ax.set_xlabel('water')
                    ax.set_ylabel('co2')
                    ax.set_zlabel('Airmass')
                    plt.show()

                    figure = plt.figure()
                    ax = plt.axes(projection='3d')
                    ax.plot_surface(np.array(water[band])[0], np.array(methane[band])[0], Z, cmap='viridis',
                                    edgecolor='none')
                    ax.set_title('water vs methane')
                    ax.set_xlabel('water')
                    ax.set_ylabel('methane')
                    ax.set_zlabel('Airmass')
                    plt.show()

                    #                for i in range(len(hgt)):
                    #                    plt.plot(hgt[i], water[i], 'o', color = 'blue', label='water')
                    #                    plt.plot(hgt[i], co2[i], 'o', color = 'orange',label='co2')
                    #                    plt.plot(hgt[i], methane[i], 'o', color = 'green',label='methane')
                    #                plt.title(band + ' water, carbon dioxide, and methane abundances'+l[0][5:9])
                    #                plt.xlabel('Hight (km)')
                    #                plt.ylabel('ppmv')
                    #                plt.legend(['water', 'co2', 'methane'])
                    #                plt.show()

    def plot_locations(self):

        m = Basemap(projection='ortho', lon_0=0, lat_0=0, resolution=None, rsphere=1737000, satellite_height=384400000)
        m.warpimage('MoonSample.jpg')

        m.drawparallels(np.arange(-90., 120., 30.))
        m.drawmeridians(np.arange(0., 420., 30.))

        for i in range(len(self.prekeys)):
            ra = self.headers['ESO GEN MOON RA'][i] - self.headers['RA'][i]
            dec = self.headers['ESO GEN MOON DEC'][i] - self.headers['DEC'][i]

            x = np.sin(np.deg2rad(ra)) * 384400000
            y = np.sin(np.deg2rad(dec)) * 384400000

            lon, lat = m(x, y, inverse=True)
            m.plot(x, y, 'bo', markersize=3)

        plt.title('Observation Positions')
        plt.savefig('Obs_pos.png')
        plt.show()

        return self

    def save_simple_fits(self):
        for band in self.bands:
            if band == 'uvb':
                specs = self.PostUdata
                wave = self.Uwave0
                skipme = 4
            elif band == 'vis':
                specs = self.PostVdata
                wave = self.Vwave0
                skipme = 4
            elif band == 'nir':
                specs = self.PostNdata
                wave = self.Nwave0
                skipme = 5
            else:
                specs = {}
                wave = []
                skipme = 0

            for name in specs.keys():
                newname = 'Moon/Data/PostMolecfit/' + name[:skipme] + 'Reduced_' + name[skipme:-9] + '.txt'
                flux = np.array(specs[name])
                with open(newname, 'w+') as newf:
                    for idx in range(len(wave)):
                        l = str(wave[idx]) + ' ' + str(flux[idx]) + '\n'
                        newf.write(l)

    def collapse_planet(self):

        # Collapses 3D IFU data into 2D using medians. Save as new files

        for band in self.bands:
            if band == 'uvb':
                specs = self.RawUdata
                wave0 = self.Uwave0
                msk = self.Umsk
                tels = self.Utels
                edges = self.Uedges
            elif band == 'vis':
                specs = self.RawVdata
                wave0 = self.Vwave0
                msk = self.Vmsk
                tels = self.Vtels
                edges = self.Vedges
            elif band == 'nir':
                specs = self.RawNdata
                wave0 = self.Nwave0
                msk = self.Nmsk
                tels = self.Ntels
                edges = self.Nedges
            else:
                specs = {}
                wave0 = []
                msk = []
                tels = []
                edges = []

            # Finds the median of the IFU spectral data for an image and writes it to a new file
            # Removes the X and Y (spacial) axes information and moves the Z spectral axis to where
            # Programs like Molecfit expect to find it in the header
            for k in specs.keys():
                with fits.open(self.homedir + '/Data/PreMolecfit/' + k) as hdul:
                    data = hdul[0].data
                    meds = []
                    for i in range(np.shape(data)[-1]):
                        meds.append(np.median(data, axis=1)[:, i])
                    m = np.median(meds, axis=0)
                    hdul[0].header.remove('CRVAL1')
                    hdul[0].header.remove('CDELT1')
                    hdul[0].header.remove('CRVAL2')
                    hdul[0].header.remove('CDELT2')
                    hdul[0].header.rename_keyword('CRVAL3', 'CRVAL1')
                    hdul[0].header.rename_keyword('CDELT3', 'CDELT1')
                    newf = fits.PrimaryHDU(data=m, header=hdul[0].header)
                    newf.writeto(self.homedir + '/Data/PreMolecfit/' +k[:30] + 'Median' + k[30:], overwrite=True)

            # Plots the last file for the user to double check
            w, h = plt.figaspect(0.3)
            fig = plt.figure(22, figsize=(w, h))
            ax = fig.add_subplot(111)
            ax.plot(wave0, m)
            for t in tels:
                ax.axvspan(t[0], t[1], facecolor='olivedrab', alpha=0.4)
            for e in edges:
                ax.axvspan(e[0], e[1], facecolor='firebrick', alpha=0.4)
            plt.show()

    def normalize_spec(self, mask, spec):
        # Normalizes a spectrum using the mask defined in the init

        ms = ma.masked_array(spec, mask=mask)
        norm_spec = spec / ms.max()

        return norm_spec

    def ifu_recovery(self, *args):

        # Determines the telluric spectrum for each frame, then applies it to the premolecfit data to
        # recover the spacial information from the IFU by dividing each pixel with the telluric average

        # Retrieves the spectra and mask information for each band
        for band in self.bands:
            if band == 'uvb':
                raw = self.RawUdata
                info = self.RawUinfo
                collapsed = self.PreUdata
                post = self.PostUdata
                msk = self.Umsk
                wave0 = self.Uwave0
                tels = self.Utels
                colors = self.Ucolors
                colornames = self.Ucolornames
                edges = self.Uedges
            elif band == 'vis':
                raw = self.RawVdata
                info = self.RawVinfo
                collapsed = self.PreVdata
                post = self.PostVdata
                msk = self.Vmsk
                wave0 = self.Vwave0
                tels = self.Vtels
                colors = self.Vcolors
                colornames = self.Vcolornames
                edges = self.Vedges
            elif band == 'nir':
                raw = self.RawNdata
                info = self.RawNinfo
                collapsed = self.PreNdata
                post = self.PostNdata
                msk = self.Nmsk
                wave0 = self.Nwave0
                tels = self.Ntels
                colors = self.Ncolors
                colornames = self.Ncolornames
                edges = self.Nedges
            else:
                raw = {}
                info = {}
                collapsed = {}
                post = {}
                msk = []
                wave0 = []
                tels = []
                colors = []
                colornames = []
                edges = []

            # Applies the mask to each spectrum of the collapsed data and normalizes them
            # Orders the files so that they will match with the other dataset
            collnorm = {}
            for fname in collapsed:
                if band == 'vis:':
                    collnorm[fname] = self.normalize_spec(msk, collapsed[fname][:24317])
                else:
                    collnorm[fname] = self.normalize_spec(msk, collapsed[fname])
            collkeys = sorted(list(collnorm.keys()))

            # Applies the mask to each spectrum of the post-molecfit data and normalizes them
            postnorm = {}
            for fname in post:
                postnorm[fname] = self.normalize_spec(msk, post[fname])
            postkeys = sorted(list(postnorm.keys()))

            # Finds the ratio between the Post / Pre Molecfit data
            telluricspecs = dict.fromkeys(postkeys)
            for i, key in enumerate(postkeys):
                telluricspecs[key] = postnorm[key] / collnorm[collkeys[i]]

            # Plots the difference between the collapsed pre-molecfit data and the post molecfit data to show
            # the effects of earth's atmosphere on the image
            if 'plot_tellurics' in args:
                w, h = plt.figaspect(0.5)
                fig = plt.figure(1, figsize=(w, h))
                gspec = gridspec.GridSpec(len(collkeys), 1)
                fig.tight_layout()
                for i, fname in enumerate(postkeys):
                    ax = fig.add_subplot(gspec[i, 0])
                    ax.plot(wave0, telluricspecs[fname])
                    for t in tels:
                        ax.axvspan(t[0], t[1], facecolor='olivedrab', alpha=0.4)
                    for e in edges:
                        ax.axvspan(e[0], e[1], facecolor='firebrick', alpha=0.4)
                    for idx, c in enumerate(colors):
                        ax.axvspan(c[0], c[1], facecolor=colornames[idx], alpha=0.4)
                    if i != len(collkeys)-1:
                        ax.tick_params(bottom=False, labelbottom=False)
                fig.suptitle(self.planet + ' ' + band.upper() + ' Post / Pre Molecfit Ratio')
                plt.show()

            # Uses the ratio to remove telluric lines from IFU data
            rawkeys = sorted(list(raw.keys()))
            recovered = dict.fromkeys(rawkeys, np.zeros(np.shape(raw[rawkeys[0]])))
            for i, key in enumerate(rawkeys):
                ratio = telluricspecs[postkeys[i]]
                for x in range(np.shape(raw[key])[-2]):
                    for y in range(np.shape(raw[key])[-1]):
                        spec = np.array(raw[key])[:,x,y]
                        nspec = self.normalize_spec(msk, spec)
                        rec = ratio * nspec
                        recovered[key][:,x,y] = rec
                newf = fits.PrimaryHDU(data=recovered[key], header=info[key])
                newf.writeto(postkeys[i][:-5] + '_IFU' + postkeys[i][-5:], overwrite=True)
                print('Wrote file '+ postkeys[i][:-5] + '_IFU' + postkeys[i][-5:])

    def plot_ifu(self):

        # Plots all spectra for each pixel of each image

        # Determines what data to use depending on the user input
        for band in self.bands:
            if band == 'uvb':
                specs = self.IfuUdata
                wave = self.Uwave
                wave0 = self.Uwave0
                msk = self.Umsk
                tels = self.Utels
                colors = self.Ucolors
                colornames = self.Ucolornames
                edges = self.Uedges
            elif band == 'vis':
                specs = self.IfuVdata
                wave= self.Vwave
                wave0 = self.Vwave0
                msk = self.Vmsk
                tels = self.Vtels
                colors = self.Vcolors
                colornames = self.Vcolornames
                edges = self.Vedges
            elif band == 'nir':
                specs = self.IfuNdata
                wave = self.Nwave
                wave0 = self.Nwave0
                msk = self.Nmsk
                tels = self.Ntels
                colors = self.Ncolors
                colornames = self.Ncolornames
                edges = self.Nedges
            else:
                specs = {}
                wave = {}
                wave0 = []
                msk = []
                tels = []
                colors = []
                colornames = []
                edges = []

            # Plots each spectra in a grid cooresponding with i't position in the image
            for fname in specs:
                im = np.array(specs[fname])
                w, h = plt.figaspect(0.5)
                fig = plt.figure(1, figsize=(w, h))
                gspec = gridspec.GridSpec(np.shape(im)[-2], np.shape(im)[-1])
                # fig.tight_layout()
                gspec.update(wspace=0.025, hspace=0.05)
                for x in range(np.shape(im)[-2]):
                    for y in range(np.shape(im)[-1]):
                        ax = fig.add_subplot(gspec[x,y])
                        ax.plot(wave0, im[:,x,y])
                        ax.set_ylim(bottom=0, top=1)
                        for t in tels:
                            ax.axvspan(t[0], t[1], facecolor='olivedrab', alpha=0.4)
                        for e in edges:
                            ax.axvspan(e[0], e[1], facecolor='firebrick', alpha=0.4)
                        for idx, c in enumerate(colors):
                            ax.axvspan(c[0], c[1], facecolor=colornames[idx], alpha=0.4)
                        ax.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
                fig.suptitle(fname)
                plt.show()