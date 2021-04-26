import numpy as np
import numpy.ma as ma
import scipy as sp
from scipy import integrate
from scipy import interpolate
from scipy.stats import chisquare
from scipy.signal import convolve  # , boxcar
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib
#matplotlib.use('TkAgg')
#matplotlib.use('ps')
import glob
from sklearn.cluster import KMeans
from sklearn import metrics
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
from astropy.convolution import Gaussian1DKernel
from astropy.convolution import Gaussian2DKernel
from astropy.convolution import convolve
from astropy.time import Time
from astropy.stats import sigma_clip
import pdb
# from mpl_toolkits.basemap import Basemap
# from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import os, sys

class PlanetAnalysis:


    # TODO: Email Jason asking about solid state methane spectra references
    # TODO: Continuum fit (methane vs continuum kmeans)

    # TODO: Make plot about which wavelengths molecfit uses for the atmospheric profile (aka real green bars)
    # TODO: moving average/cosmic rays - mask out outliers in residuals (sigma clipping)
    # TODO: saturn position
    # TODO: Revisit moon
    # pre molecfit, post molecfit, rolling average if needed
    # batsignal


    def __init__(self, **kwargs):

        # Determine what dataset we will be working with (Planet, Pre/Post Molecfit, Bands)
        planets_ready = ['Moon', 'Saturn', 'Neptune', 'Enceladus', 'Titan', 'Sun', 'Pure_Saturn', 'Pure_Rings']
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
            if self.reduced not in ['r', 'p', 'c', 'm', 'i', 'a']:
                print("Reduction status not recognized. Please enter 'raw', 'post_xshooter_pipeline', 'collapsed', '"
                      "molecfit', 'ifu', or 'all'.")
                while True:
                    self.reduced = input("Reduction Status?: ")
                    self.reduced = self.reduced[0].lower()
                    if self.reduced == '' or self.reduced not in ['r', 'p', 'c', 'm', 'i', 'a']:
                        print("Please enter 'raw', 'post_xshooter_pipeline', 'collapsed', or 'molecfit', 'ifu',"
                              " or 'all'.")
                    else:
                        break
        else:
            while True:
                self.reduced = input("Reduction Status?: ")
                self.reduced = self.reduced[0].lower()
                if self.reduced == '' or self.reduced not in ['r', 'p', 'c', 'm', 'i', 'a']:
                    print("Please enter 'raw', 'post_xshooter_pipeline', 'collapsed', or 'molecfit', 'ifu', or 'all'.")
                else:
                    break

        self.rawdir = self.homedir + '/Data/Raw'
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
            if self.planet == "Moon":
                xs = glob.glob(self.rawdir + '/X*.fits')
                ms = glob.glob(self.rawdir + '/r*.fits')
            elif self.planet == 'Sun':
                xs = []
                ms = []
            else:
                xs = glob.glob(self.rawdir + '/X*.fits')
                ms = glob.glob(self.rawdir + '/M*.fits')
                if not xs:
                    xs = glob.glob(self.rawdir + '/data_with_raw_calibs/X*.fits')
                    ms = glob.glob(self.rawdir + '/data_with_raw_calibs/M*.fits')
            self.RawXinfo = dict.fromkeys(xs, [])
            self.RawMinfo = dict.fromkeys(ms, [])

            self.RawKeys = list(self.RawXinfo.keys())
            self.RawKeys += list(self.RawMinfo.keys())
            self.RawData = {}
            self.RawXdata = dict.fromkeys(xs, [])
            self.RawMdata = dict.fromkeys(ms, [])

            print("Raw data info collected")

        # Organize Post XSHOOTER Reduction Pipeline data
        if self.reduced =='p' or self.reduced =='a':
            if self.planet == "Moon":
                us = glob.glob(self.predir + '/**/MOV*FLUX_MERGE2D_UVB.fits')
                vs = glob.glob(self.predir + '/**/MOV*FLUX_MERGE2D_VIS.fits')
                ns = glob.glob(self.predir + '/**/MOV*FLUX_MERGE2D_NIR.fits')
            elif self.planet == 'Sun':
                us = []
                vs = glob.glob(self.predir + '/vis/*MERGE1D_VIS*.fits')
                ns = glob.glob(self.predir + '/nir/*MERGE1D_NIR*.fits')
            else:
                us = glob.glob(self.predir + '/Clean_XSHOOTER_Pipeline/**/MOV*MERGE3D_DATA*_UVB.fits')
                vs = glob.glob(self.predir + '/Clean_XSHOOTER_Pipeline/**/MOV*MERGE3D_DATA*_VIS.fits')
                ns = glob.glob(self.predir + '/Clean_XSHOOTER_Pipeline/**/MOV*MERGE3D_DATA*_NIR.fits')
            self.PostXPUinfo = dict.fromkeys(us, [])
            self.PostXPVinfo = dict.fromkeys(vs, [])
            self.PostXPNinfo = dict.fromkeys(ns, [])

            self.PostXPKeys = list(self.PostXPUinfo.keys())
            self.PostXPKeys += list(self.PostXPVinfo.keys())
            self.PostXPKeys += list(self.PostXPNinfo.keys())
            self.PostXPData = {}

            print("Post XSHOOTER Pipeline data info collected")

        # Organize collapsed data
        if self.reduced == 'c' or self.reduced == 'a':
            self.PreUinfo = {}
            self.PreVinfo = {}
            self.PreNinfo = {}

            if self.planet == "Moon":
                us = glob.glob(self.postdir + '/uvb/2020*tac.fits')
                vs = glob.glob(self.postdir + '/vis/2020*tac.fits')
                ns = glob.glob(self.postdir + '/nir/2020*tac.fits')
            elif self.planet == 'Sun':
                us = []
                vs = glob.glob(self.predir + '/vis/*MERGE1D_VIS*.fits')
                ns = glob.glob(self.predir + '/nir/*MERGE1D_NIR*.fits')
            else:
                #us = glob.glob(self.predir + '/**/MedianMOV*_UVB.fits')
                us = []
                #vs = glob.glob(self.predir + '/**/MedianMOV*_VIS.fits')
                vs = glob.glob(self.predir + '/vis/MOV_*sum_sum*.fits')
                #vs = glob.glob('/home/gmansir/Thesis/Neptune/Data/PreMolecfit/XSHOO.2019-08-30T04:48:43.725/MOV_*pixel_1_1*.fits')
                #ns = glob.glob(self.predir + '/**/MedianMOV*_NIR.fits')
                ns = glob.glob(self.predir + '/nir/MOV_*sum_sum*.fits')

            self.PreUinfo = dict.fromkeys(us, [])
            self.PreVinfo = dict.fromkeys(vs, [])
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

            notdata = ['atm', 'res', 'gui', 'TAC', 'TRA', 'VIS_fit']
            #if self.planet == 'Sun':
            #    us = []
            #    dus = []
            #    self.PostUinfo = {}
            #    vs = glob.glob(self.postdir + '/2020*vis*tac.fits')
            #    dvs = [f for f in vs if not any(nd in f for nd in notdata)]
            #    self.PostVinfo = dict.fromkeys(dvs, [])
            #    ns = glob.glob(self.postdir + '/2020*nir*tac.fits')
            #    dns = [f for f in ns if not any(nd in f for nd in notdata)]
            #    self.PostNinfo = dict.fromkeys(dns, [])

            #us = glob.glob(self.postdir + '/uvb/2020*tac.fits')
            #dus = [f for f in us if not any(nd in f for nd in notdata)]
            #self.PostUinfo = dict.fromkeys(dus, [])
            us = []
            dus = []
            if self.planet == 'Sun':
                vs = glob.glob(self.postdir + '/vis/*SCIENCE_TELLURIC_CORR*_row*.fits')
                ns = glob.glob(self.postdir + '/nir/*SCIENCE_TELLURIC_CORR*_row*.fits')
            else:
                vs = glob.glob(self.postdir + '/vis/*SCIENCE_TELLURIC_CORR*.fits')
                ns = glob.glob(self.postdir + '/nir/*SCIENCE_TELLURIC_CORR*.fits')
            dvs = [f for f in vs if not any(nd in f for nd in notdata)]
            self.PostVinfo = dict.fromkeys(dvs, [])
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

            us = glob.glob(self.postdir + '/uvb/*tac_IFU.fits')
            self.IfuUinfo = dict.fromkeys(us, [])
            vs = glob.glob(self.postdir + '/vis/*tac_IFU.fits')
            self.IfuVinfo = dict.fromkeys(vs, [])
            ns = glob.glob(self.postdir + '/nir/*tac_IFU.fits')
            self.IfuNinfo = dict.fromkeys(ns, [])

            self.ifukeys = list(self.IfuUinfo.keys())
            self.ifukeys += list(self.IfuVinfo.keys())
            self.ifukeys += list(self.IfuNinfo.keys())
            self.IfuUdata = {}
            self.IfuVdata = {}
            self.IfuNdata = {}

            print("Corrected IFU data info collected")

        # Organize Solar-like Star data
        if self.reduced == 's':
            self.RawUinfo = {}
            self.RawVinfo = {}
            self.RawNinfo = {}
            if self.planet == "Moon":
                us = glob.glob(self.predir + '/*tpl*/H*FLUX_MERGE1D_UVB.fits')
                vs = glob.glob(self.predir + '/*tpl*/H*FLUX_MERGE1D_VIS.fits')
                ns = glob.glob(self.predir + '/*tpl*/H*FLUX_MERGE1D_NIR.fits')
            else:
                us = glob.glob(self.predir + '/*tpl*/L*MERGE3D_DATA*OBJ_UVB.fits')
                us += glob.glob(self.predir + '/*tpl*/H*MERGE3D_DATA*OBJ_UVB.fits')
                us += glob.glob(self.predir + '/*tpl*/F*MERGE3D_DATA*OBJ_UVB.fits')
                vs = glob.glob(self.predir + '/*tpl*/L*MERGE3D_DATA*OBJ_VIS.fits')
                vs += glob.glob(self.predir + '/*tpl*/H*MERGE3D_DATA*OBJ_VIS.fits')
                vs += glob.glob(self.predir + '/*tpl*/F*MERGE3D_DATA*OBJ_VIS.fits')
                ns = glob.glob(self.predir + '/*tpl*/L*MERGE3D_DATA*OBJ_NIR.fits')
                ns += glob.glob(self.predir + '/*tpl*/H*MERGE3D_DATA*OBJ_NIR.fits')
                ns += glob.glob(self.predir + '/*tpl*/F*MERGE3D_DATA*OBJ_NIR.fits')


            self.RawUinfo = dict.fromkeys(us, [])
            self.RawVinfo = dict.fromkeys(vs, [])
            self.RawNinfo = dict.fromkeys(ns, [])

            self.rawkeys = list(self.RawUinfo.keys())
            self.rawkeys += list(self.RawVinfo.keys())
            self.rawkeys += list(self.RawNinfo.keys())
            self.RawUdata = {}
            self.RawVdata = {}
            self.RawNdata = {}

            print("Raw data info collected.")

            self.PreUinfo = {}
            self.PreVinfo = {}
            self.PreNinfo = {}

            if self.planet == "Moon":
                us = glob.glob(self.postdir + '/uvb/2020_solar*tac.fits')
                vs = glob.glob(self.postdir + '/vis/2020_solar*tac.fits')
                ns = glob.glob(self.postdir + '/nir/2020_solar*tac.fits')
            else:
                us = glob.glob(self.predir + '/*tpl*/Median*_OBJ_UVB.fits')
                vs = glob.glob(self.predir + '/*tpl*/Median*_OBJ_VIS.fits')
                ns = glob.glob(self.predir + '/*tpl*/Median*_OBJ_NIR.fits')

            self.PreUinfo = dict.fromkeys(us, [])
            self.PreVinfo = dict.fromkeys(vs, [])
            self.PreNinfo = dict.fromkeys(ns, [])
            self.prekeys = list(self.PreUinfo.keys())
            self.prekeys += list(self.PreVinfo.keys())
            self.prekeys += list(self.PreNinfo.keys())
            self.PreUdata = {}
            self.PreVdata = {}
            self.PreNdata = {}

            print("Collapsed data info collected")

            self.PostUinfo = {}
            self.PostVinfo = {}
            self.PostNinfo = {}

            notdata = ['atm', 'res', 'gui', 'TAC', 'TRA', 'VIS_fit']
            if self.planet == "Moon":
                us = glob.glob(self.postdir + '/uvb/2020_solar*tac.fits')
                vs = glob.glob(self.postdir + '/vis/2020_solar*tac.fits')
                ns = glob.glob(self.postdir + '/nir/2020_solar*tac.fits')
            else:
                us = glob.glob(self.postdir + '/uvb/Solar_2020*tac.fits')
                dus = [f for f in us if not any(nd in f for nd in notdata)]
                vs = glob.glob(self.postdir + '/vis/Solar_2020*tac.fits')
                dvs = [f for f in vs if not any(nd in f for nd in notdata)]
                ns = glob.glob(self.postdir + '/nir/Solar_2020*tac.fits')
                dns = [f for f in ns if not any(nd in f for nd in notdata)]

            self.PostUinfo = dict.fromkeys(dus, [])
            self.PostVinfo = dict.fromkeys(dvs, [])
            self.PostNinfo = dict.fromkeys(dns, [])

            self.postkeys = list(self.PostUinfo.keys())
            self.postkeys += list(self.PostVinfo.keys())
            self.postkeys += list(self.PostNinfo.keys())
            self.PostUdata = {}
            self.PostVdata = {}
            self.PostNdata = {}

            print("Post Molecfit data info collected")

            self.IfuUinfo = {}
            self.IfuVinfo = {}
            self.IfuNinfo = {}

            us = glob.glob(self.postdir + '/uvb/Solar*tac_IFU.fits')
            self.IfuUinfo = dict.fromkeys(us, [])
            vs = glob.glob(self.postdir + '/vis/Solar*tac_IFU.fits')
            self.IfuVinfo = dict.fromkeys(vs, [])
            ns = glob.glob(self.postdir + '/nir/Solar*tac_IFU.fits')
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
            for k in self.RawKeys:
                with fits.open(k) as hdul:
                    data = hdul[0].data
                    data = data.tolist()
                    if self.planet == 'Moon' or self.planet =='Sun' or self.planet == 'Neptune':
                        try:
                            CRVAL = hdul[0].header['CRVAL1']
                            CDELT = hdul[0].header['CDELT1']
                            NAXIS = hdul[0].header['NAXIS1']
                        except KeyError:
                            CRVAL = 1.0
                            CDELT = 1.0
                            NAXIS = 1
                    else:
                        CRVAL = hdul[0].header['CRVAL3']
                        CDELT = hdul[0].header['CDELT3']
                        NAXIS = hdul[0].header['NAXIS3']
                    wave = [CRVAL+CDELT*i for i in range(NAXIS)]
                    header = hdul[0].header
                if np.shape(data) != ():
                    if k in self.RawXinfo:
                        self.RawXinfo[k] = header
                        self.RawXdata[k] = data
                        self.Vwave[k] = wave
                    elif k in self.RawMinfo:
                        self.RawMinfo[k] = header
                        self.RawMdata[k] = data
                        self.Mwave[k] = wave
                    elif k in self.RawNinfo:
                        self.RawNinfo[k] = header
                        self.RawNdata[k] = data
                        self.Nwave[k] = wave
                else:
                    pass
            print("Raw data collected")

        if self.reduced == 'c' or self.reduced == 'a' or self.reduced == 's':
            for k in self.prekeys:
                with fits.open(k) as hdul:
                    if self.planet == "Moon":
                        data = hdul[1].data["mtrans"]
                        wave = hdul[1].data["mlambda"]
                        header = hdul[1].header
                    else:
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

        if self.reduced == 'm' or self.reduced == 'a' or self.reduced == 's':
            for k in self.postkeys:
                with fits.open(k) as hdul:
                    data = hdul[0].data
                    CRVAL1 = hdul[0].header['CRVAL1']
                    CDELT1 = hdul[0].header['CDELT1']
                    NAXIS1 = hdul[0].header['NAXIS1']
                    wave = [CRVAL1+CDELT1*i for i in range(NAXIS1)]
                    header = hdul[0].header
                    if np.shape(data) != ():
                        if k in self.PostUinfo:
                            self.PostUinfo[k] = header
                            self.PostUdata[k] = data
                            self.Uwave[k] = wave
                        elif k in self.PostVinfo:
                            self.PostVinfo[k] = header
                            self.PostVdata[k] = data
                            self.Vwave[k] = wave
                        elif k in self.PostNinfo:
                            self.PostNinfo[k] = header
                            self.PostNdata[k] = data
                            self.Nwave[k] = wave
                #with fits.open(k) as hdul:
                #    data = hdul[1].data["mtrans"]
                #    wave = hdul[1].data["mlambda"]
                #if np.shape(data) != ():
                #    if k in self.PostUinfo:
                #        self.PostUdata[k] = data
                #        self.Uwave[k] = wave
                #    elif k in self.PostVinfo:
                #        self.PostVdata[k] = data
                #        self.Vwave[k] = wave
                #    elif k in self.PostNinfo:
                #        self.PostNdata[k] = data
                #        self.Nwave[k] = wave
                    else:
                        pass
            print("Post Molecfit data collected")

        if self.reduced == 'i' or self.reduced == 'a' or self.reduced == 's':
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
            self.UtelsEye = [[0.397, 0.401], [0.416, 0.42], [0.44, 0.449], [0.467, 0.475],
                          [0.484, 0.490], [0.50, 0.514], [0.523, 0.53], [0.536, 0.552]]
            self.Utels = [[0.3403, 0.3457],[0.3542, 0.3643],[0.3765, 0.3834],[0.4455, 0.4475],[0.4675, 0.4839],
                          [0.5233, 0.5401],[0.5534, 0.556 ]]
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
            self.VtelsEye = [[0.54, 0.55], [0.567, 0.58], [0.585, 0.605], [0.626, 0.635], [0.643, 0.665], [0.685, 0.746], [0.757, 0.775],
                          [0.783, 0.86], [0.878, 1.02]]
            self.Vtels = [[0.5402, 0.5403],[0.5407, 0.541 ],[0.5412, 0.5421],[0.5423, 0.5425],[0.5429, 0.5432],
                               [0.5434, 0.5439],[0.5441, 0.5444],[0.5446, 0.5452],[0.5455, 0.5465],[0.5467, 0.5468],
                               [0.5477, 0.5479],[0.5494, 0.5495],[0.5666, 0.5667],[0.5674, 0.5675],[0.568 , 0.5694],
                               [0.5696, 0.5704],[0.5706, 0.5707],[0.5711, 0.5714],[0.5717, 0.5724],[0.5726, 0.573 ],
                               [0.5733, 0.5739],[0.5741, 0.5748],[0.575 , 0.5764],[0.5766, 0.5767],[0.5769, 0.5773],
                               [0.5779, 0.578 ],[0.5782, 0.5783],[0.5786, 0.5787],[0.5793, 0.5794],[0.5804, 0.5806],
                               [0.5822, 0.5823],[0.5841, 0.5843],[0.5845, 0.5846],[0.5858, 0.6009],[0.6011, 0.6013],
                               [0.6015, 0.6016],[0.6018, 0.6019],[0.6261, 0.6262],[0.6264, 0.6265],[0.6267, 0.627 ],
                               [0.6272, 0.6273],[0.6275, 0.63  ],[0.6302, 0.6307],[0.6309, 0.6333],[0.6335, 0.6338],
                               [0.6341, 0.6343],[0.6345, 0.6347],[0.635 , 0.6352],[0.6358, 0.6359],[0.6361, 0.6362],
                               [0.6368, 0.6373],[0.6381, 0.6383],[0.6385, 0.6386],[0.6389, 0.6394],[0.6396, 0.6401],
                               [0.6403, 0.6405],[0.6407, 0.6411],[0.6413, 0.6414],[0.6416, 0.6417],[0.6425, 0.6427],
                               [0.6429, 0.643 ],[0.6432, 0.6435],[0.6439, 0.644 ],[0.6442, 0.6455],[0.6457, 0.6528],
                               [0.653 , 0.6576],[0.6579, 0.6589],[0.6593, 0.6597],[0.6599, 0.6606],[0.6609, 0.661 ],
                               [0.6612, 0.6613],[0.6615, 0.6616],[0.6621, 0.6623],[0.6628, 0.6629],[0.6632, 0.6633],
                               [0.6641, 0.6644],[0.6824, 0.6825],[0.6842, 0.6844],[0.6853, 0.6854],[0.6859, 0.686 ],
                               [0.6863, 0.6966],[0.6968, 0.7032],[0.7035, 0.7116],[0.7118, 0.7418],[0.742 , 0.7426],
                               [0.7433, 0.7435],[0.7437, 0.7442],[0.7449, 0.7452],[0.7454, 0.7455],[0.7458, 0.746 ],
                               [0.7467, 0.7469],[0.7592, 0.7673],[0.7675, 0.7679],[0.7681, 0.7685],[0.7687, 0.7699],
                               [0.7701, 0.7704],[0.771 , 0.7712],[0.7716, 0.7718],[0.772 , 0.7721],[0.7725, 0.7726],
                               [0.7729, 0.773 ],[0.7734, 0.7735],[0.7739, 0.774 ],[0.7744, 0.7745],[0.7828, 0.7829],
                               [0.7839, 0.7841],[0.7845, 0.7847],[0.7849, 0.7851],[0.7853, 0.7856],[0.7858, 0.7877],
                               [0.7879, 0.7882],[0.7884, 0.7897],[0.7899, 0.7904],[0.7906, 0.7913],[0.7915, 0.7925],
                               [0.7927, 0.7933],[0.7939, 0.7943],[0.7945, 0.7948],[0.795 , 0.7954],[0.7956, 0.805 ],
                               [0.8052, 0.8391],[0.8393, 0.8455],[0.8459, 0.8464],[0.8468, 0.847 ],[0.8473, 0.8483],
                               [0.8486, 0.8489],[0.8491, 0.8494],[0.8496, 0.8497],[0.8499, 0.85  ],[0.8502, 0.8503],
                               [0.8505, 0.8506],[0.8509, 0.852 ],[0.8522, 0.8523],[0.8525, 0.8527],[0.853 , 0.8532],
                               [0.8534, 0.8537],[0.8539, 0.8541],[0.8546, 0.8547],[0.8549, 0.8559],[0.8562, 0.8563],
                               [0.8567, 0.8576],[0.859 , 0.8593],[0.8605, 0.8606],[0.8608, 0.8611],[0.8625, 0.8627],
                               [0.8689, 0.8691],[0.8735, 0.8736],[0.8758, 0.8759],[0.878 , 0.8782],[0.8785, 0.8788],
                               [0.8802, 0.8805],[0.8811, 0.8812],[0.8819, 0.8824],[0.8829, 0.8832],[0.8835, 0.8839],
                               [0.8848, 0.8849],[0.8851, 0.8853],[0.8856, 0.886 ],[0.8865, 0.8867],[0.8869, 0.887 ],
                               [0.8872, 0.8874],[0.8877, 0.8883],[0.8887, 0.8889],[0.8893, 0.8901],[0.8907, 0.8908],
                               [0.891 , 0.8912],[0.8915, 0.9902],[0.9904, 0.9917],[0.9919, 0.9937],[0.9939, 0.9942],
                               [0.9946, 0.9948],[0.995 , 0.9958],[0.9961, 0.9962],[0.9965, 0.9972],[0.9974, 0.998 ],
                               [0.9983, 0.9989],[0.9991, 0.9995],[0.9997, 1.    ],[1.001 , 1.0013],[1.0015, 1.0026],
                               [1.0031, 1.0032],[1.0036, 1.0038],[1.0041, 1.0042],[1.0046, 1.0055],[1.0058, 1.0059],
                               [1.0062, 1.0065],[1.0068, 1.007 ],[1.0073, 1.0075],[1.0077, 1.0079],[1.0081, 1.0083],
                               [1.0085, 1.0103],[1.0105, 1.0107],[1.0109, 1.011 ],[1.0115, 1.012 ],[1.0124, 1.0125],
                               [1.0129, 1.0131],[1.014 , 1.0142],[1.0144, 1.0146],[1.0152, 1.0155],[1.0165, 1.0168],
                               [1.017 , 1.0173],[1.0175, 1.0176],[1.0183, 1.0184],[1.0189, 1.0192]]

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
            self.OldNtels = [[1.058, 1.066], [1.137, 1.200], [1.350, 1.530], [1.805, 1.991], [2.004, 2.044],
                          [2.053, 2.088]]
            self.NtelsEye = [[1.058, 1.233], [1.240, 1.290], [1.300, 1.550], [1.560, 1.583], [1.589, 1.616],
                          [1.621, 1.695], [1.71, 2.03], [2.047, 2.082], [2.095, 2.136], [2.150, 2.45]]
            self.Ntels = [[1.0547, 1.0717],[1.0719, 1.0723],[1.0725, 1.0727],[1.0742, 1.0745],[1.0771, 1.0772],
                          [1.0798, 1.0799],[1.081 , 1.0811],[1.0815, 1.0816],[1.0831, 1.0834],[1.0842, 1.0843],
                          [1.0849, 1.085 ],[1.0856, 1.086 ],[1.0867, 1.0868],[1.0881, 1.0882],[1.0888, 1.0889],
                          [1.0891, 1.0892],[1.09  , 1.0901],[1.0921, 1.0925],[1.0941, 1.0947],[1.0949, 1.095 ],
                          [1.0954, 1.0961],[1.0972, 1.0978],[1.0981, 1.0988],[1.099 , 1.1009],[1.1012, 1.1034],
                          [1.1036, 1.1065],[1.1068, 1.1692],[1.1695, 1.1742],[1.1744, 1.1755],[1.1762, 1.1764],
                          [1.1767, 1.1783],[1.1789, 1.1797],[1.18  , 1.1801],[1.1803, 1.1829],[1.1831, 1.1858],
                          [1.186 , 1.1891],[1.1894, 1.19  ],[1.1902, 1.191 ],[1.1914, 1.1917],[1.1921, 1.1924],
                          [1.1928, 1.1933],[1.194 , 1.1943],[1.1946, 1.1948],[1.195 , 1.1955],[1.1959, 1.1962],
                          [1.1965, 1.1966],[1.1968, 1.1974],[1.198 , 1.1989],[1.1992, 1.2001],[1.2003, 1.2016],
                          [1.2022, 1.2052],[1.2055, 1.2072],[1.2074, 1.2082],[1.2084, 1.2092],[1.2094, 1.2097],
                          [1.2106, 1.2116],[1.2118, 1.2127],[1.2133, 1.2148],[1.215 , 1.2151],[1.2158, 1.2166],
                          [1.2175, 1.2201],[1.2209, 1.2215],[1.2217, 1.2218],[1.222 , 1.2224],[1.2226, 1.2227],
                          [1.223 , 1.2235],[1.2237, 1.2241],[1.2244, 1.2253],[1.2256, 1.2258],[1.2262, 1.2272],
                          [1.2274, 1.2276],[1.2284, 1.2286],[1.229 , 1.2292],[1.2296, 1.2299],[1.2308, 1.231 ],
                          [1.2313, 1.2314],[1.2322, 1.2331],[1.234 , 1.2342],[1.2344, 1.2872],[1.2874, 1.289 ],
                          [1.2894, 1.2901],[1.2904, 1.2907],[1.2911, 1.2919],[1.2921, 1.2923],[1.2925, 1.2926],
                          [1.2934, 1.2937],[1.2941, 1.2942],[1.2944, 1.2947],[1.295 , 1.2955],[1.2957, 1.2971],
                          [1.2973, 1.2976],[1.2978, 1.2979],[1.2989, 1.2994],[1.2999, 1.301 ],[1.3018, 1.3019],
                          [1.3022, 1.3025],[1.3028, 1.3042],[1.3045, 1.3046],[1.305 , 1.3055],[1.3059, 1.307 ],
                          [1.3073, 1.3109],[1.3115, 1.4996],[1.5001, 1.5055],[1.5057, 1.5058],[1.5062, 1.5063],
                          [1.5065, 1.5091],[1.5096, 1.5101],[1.5103, 1.5113],[1.5118, 1.5139],[1.5142, 1.5145],
                          [1.515 , 1.5151],[1.5155, 1.5175],[1.5177, 1.518 ],[1.5182, 1.5192],[1.5194, 1.52  ],
                          [1.5211, 1.5215],[1.5229, 1.523 ],[1.5239, 1.5242],[1.5244, 1.5248],[1.525 , 1.5254],
                          [1.5259, 1.5264],[1.5268, 1.527 ],[1.5281, 1.5283],[1.529 , 1.5293],[1.5297, 1.5299],
                          [1.5305, 1.5311],[1.532 , 1.5321],[1.5323, 1.5324],[1.5326, 1.5327],[1.5329, 1.533 ],
                          [1.5332, 1.5337],[1.5339, 1.534 ],[1.5342, 1.5347],[1.5349, 1.535 ],[1.5352, 1.5353],
                          [1.5355, 1.536 ],[1.5362, 1.5364],[1.5366, 1.5367],[1.537 , 1.5371],[1.5383, 1.5384],
                          [1.5386, 1.5388],[1.5391, 1.5392],[1.5398, 1.54  ],[1.5403, 1.5404],[1.5407, 1.5408],
                          [1.5411, 1.5412],[1.5415, 1.5417],[1.5419, 1.542 ],[1.5432, 1.5433],[1.5437, 1.5439],
                          [1.5674, 1.5676],[1.5678, 1.5812],[1.5814, 1.5815],[1.5818, 1.5821],[1.5824, 1.5826],
                          [1.5829, 1.583 ],[1.5835, 1.5836],[1.584 , 1.5841],[1.5845, 1.5846],[1.5857, 1.5858],
                          [1.5978, 1.6051],[1.6054, 1.6122],[1.6124, 1.6126],[1.6128, 1.6132],[1.6135, 1.6136],
                          [1.614 , 1.6142],[1.6146, 1.6148],[1.6152, 1.6153],[1.6157, 1.6159],[1.6161, 1.6164],
                          [1.6168, 1.6169],[1.6175, 1.6176],[1.6181, 1.6182],[1.6275, 1.6277],[1.628 , 1.6282],
                          [1.6298, 1.6303],[1.6315, 1.6316],[1.6323, 1.6327],[1.633 , 1.6331],[1.6348, 1.6351],
                          [1.6355, 1.6356],[1.6361, 1.6362],[1.637 , 1.6374],[1.6384, 1.6388],[1.639 , 1.6393],
                          [1.6395, 1.64  ],[1.6402, 1.6403],[1.6406, 1.6407],[1.641 , 1.6411],[1.6414, 1.6415],
                          [1.6418, 1.6419],[1.6421, 1.6428],[1.643 , 1.6431],[1.6433, 1.6435],[1.6438, 1.6439],
                          [1.6447, 1.6454],[1.6456, 1.6457],[1.6461, 1.6462],[1.6465, 1.6466],[1.6469, 1.6472],
                          [1.6474, 1.6481],[1.6483, 1.6484],[1.6486, 1.6493],[1.6496, 1.6497],[1.6499, 1.6508],
                          [1.651 , 1.6511],[1.6514, 1.6516],[1.6519, 1.652 ],[1.6529, 1.6534],[1.6544, 1.655 ],
                          [1.6552, 1.6553],[1.6558, 1.6562],[1.6565, 1.6568],[1.6588, 1.6589],[1.6607, 1.6609],
                          [1.6617, 1.6619],[1.6629, 1.663 ],[1.6646, 1.6669],[1.6673, 1.6679],[1.6681, 1.6683],
                          [1.6687, 1.669 ],[1.6692, 1.6693],[1.6699, 1.6703],[1.6707, 1.6709],[1.6738, 1.6741],
                          [1.6748, 1.675 ],[1.6768, 1.6772],[1.6778, 1.6779],[1.6801, 1.6805],[1.6833, 1.6837],
                          [1.6865, 1.687 ],[1.6883, 1.6885],[1.6899, 1.6905],[1.6908, 1.6911],[1.6914, 1.6915],
                          [1.6918, 1.6919],[1.6933, 1.6937],[1.6939, 1.6946],[1.696 , 1.697 ],[1.6977, 1.6978],
                          [1.699 , 1.6999],[1.7001, 1.7007],[1.701 , 1.7013],[1.702 , 1.7024],[1.7026, 1.7033],
                          [1.7035, 1.7042],[1.7045, 1.7051],[1.7053, 1.7084],[1.7089, 1.709 ],[1.7094, 1.7096],
                          [1.7098, 1.7102],[1.7104, 1.7106],[1.7108, 1.711 ],[1.7116, 1.7117],[1.7122, 1.7123],
                          [1.7126, 1.7129],[1.7131, 1.7142],[1.7147, 1.7148],[1.7154, 1.7158],[1.7164, 1.7165],
                          [1.7194, 1.7195],[1.7198, 1.7213],[1.7215, 1.7218],[1.7221, 1.7246],[1.7248, 1.7249],
                          [1.7251, 1.7259],[1.7264, 1.7291],[1.7296, 1.7306],[1.7309, 1.7327],[1.7332, 1.7335],
                          [1.7337, 1.7366],[1.7373, 1.738 ],[1.7383, 1.7384],[1.7388, 1.7417],[1.7422, 1.7426],
                          [1.7429, 1.7432],[1.7434, 1.7446],[1.7449, 1.7478],[1.748 , 1.7497],[1.7499, 1.7528],
                          [1.7533, 1.7588],[1.7591, 2.0956],[2.0958, 2.113 ],[2.1132, 2.1284],[2.1286, 2.129 ],
                          [2.1293, 2.1296],[2.1299, 2.1302],[2.1304, 2.1308],[2.1311, 2.1313],[2.1317, 2.1319],
                          [2.1322, 2.1325],[2.133 , 2.1331],[2.1336, 2.1337],[2.1343, 2.1344],[2.1349, 2.1356],
                          [2.1362, 2.1363],[2.1365, 2.137 ],[2.1378, 2.1379],[2.1389, 2.139 ],[2.1415, 2.1418],
                          [2.1422, 2.1424],[2.1426, 2.1429],[2.1439, 2.144 ],[2.1452, 2.1455],[2.1457, 2.1692],
                          [2.1694, 2.1702],[2.1704, 2.1727],[2.173 , 2.1749],[2.1752, 2.1759],[2.1763, 2.1764],
                          [2.1772, 2.1781],[2.1783, 2.1791],[2.182 , 2.1848],[2.1863, 2.1866],[2.1869, 2.1877],
                          [2.1888, 2.189 ],[2.1896, 2.19  ],[2.1938, 2.1946],[2.1955, 2.1962],[2.1964, 2.2055],
                          [2.206 , 2.2063],[2.2065, 2.207 ],[2.2096, 2.2098],[2.2103, 2.2107],[2.2133, 2.2134],
                          [2.2136, 2.2137],[2.214 , 2.2144],[2.215 , 2.2162],[2.2171, 2.2181],[2.2189, 2.2194],
                          [2.2198, 2.2217],[2.2219, 2.222 ],[2.2222, 2.2225],[2.2228, 2.2235],[2.2239, 2.2248],
                          [2.225 , 2.2278],[2.2283, 2.2286],[2.2288, 2.2289],[2.2293, 2.2294],[2.2296, 2.2326],
                          [2.2328, 2.234 ],[2.2343, 2.2378],[2.238 , 2.2381],[2.2384, 2.2392],[2.2394, 2.2412],
                          [2.2415, 2.2439],[2.2444, 2.3101],[2.3108, 2.3117],[2.3128, 2.313 ],[2.3139, 2.479 ]]

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

        self.aspecs = {}
        self.sdspecs = {}
        self.rchis = {}
        self.preaspecs = {}
        self.presdspecs = {}

        self.Uregions = {}
        self.Vregions = {}
        self.Nregions = {}

        self.sun_lamb = []
        self.sun_flux = []
        self.fluxclip = []

    def request(self, band, request):

        if band == 'uvb':
            if request == 'wave':
                return self.Uwave
            elif request == 'wave0':
                return self.Uwave0
            elif request == 'rawinfo':
                return self.RawXinfo
            elif request == 'raw':
                return self.RawXdata
            elif request == 'collapsed':
                return self.PreUdata
            elif request == 'molecfit':
                return self.PostUdata
            elif request == 'ifu':
                return self.IfuUdata
            elif request == 'msk':
                return self.Umsk
            elif request == 'tels':
                return self.Utels
            elif request == 'colors':
                return self.Ucolors
            elif request == 'colornames':
                return self.Ucolornames
            elif request == 'speccolor':
                return 'forestgreen'
            elif request == 'edges':
                return self.Uedges
            elif request == 'regions':
                return self.Uregions

        if band == 'vis':
            if request == 'wave':
                return self.Vwave
            elif request == 'wave0':
                return self.Vwave0
            elif request == 'rawinfo':
                return self.RawXinfo
            elif request == 'raw':
                return self.RawXdata
            elif request == 'collapsed':
                return self.PreVdata
            elif request == 'molecfit':
                return self.PostVdata
            elif request == 'ifu':
                return self.IfuVdata
            elif request == 'msk':
                return self.Vmsk
            elif request == 'tels':
                return self.Vtels
            elif request == 'colors':
                return self.Vcolors
            elif request == 'colornames':
                return self.Vcolornames
            elif request == 'speccolor':
                return 'royalblue'
            elif request == 'edges':
                return self.Vedges
            elif request == 'regions':
                return self.Vregions

        if band == 'nir':
            if request == 'wave':
                return self.Nwave
            elif request == 'wave0':
                return self.Nwave0
            elif request == 'rawinfo':
                return self.RawMinfo
            elif request == 'raw':
                return self.RawMdata
            elif request == 'collapsed':
                return self.PreNdata
            elif request == 'molecfit':
                return self.PostNdata
            elif request == 'ifu':
                return self.IfuNdata
            elif request == 'msk':
                return self.Nmsk
            elif request == 'tels':
                return self.Ntels
            elif request == 'colors':
                return self.Ncolors
            elif request == 'colornames':
                return self.Ncolornames
            elif request == 'speccolor':
                return 'rebeccapurple'
            elif request == 'edges':
                return self.Nedges
            elif request == 'regions':
                return self.Nregions

    def closest_index(self, arr, val):

        # Finds the index of the value closest to that requested in a given array
        close_func = lambda x: abs(x - val)
        close_val = min(arr, key=close_func)

        return arr.index(close_val)

    def raw_analysis(self):

        # Plots all spectra for each pixel of each image

        for band in self.bands:

            # Plots each spectra in a grid cooresponding with i't position in the image
            specs = self.request(band, 'collapsed')
            for fname in specs:
                im = np.array(specs[fname])
                w, h = plt.figaspect(0.5)
                fig = plt.figure(1, figsize=(w, h))
                if self.planet == "Moon" or self.planet == "Sun" or self.planet == "Neptune":
                    gspec = gridspec.GridSpec(len(specs), 1)
                    exs = len(specs)
                    ys = 1
                else:
                    gspec = gridspec.GridSpec(np.shape(im)[-2], np.shape(im)[-1])
                    exs = np.shape(im)[-2]
                    ys = np.shape(im)[-1]
                # fig.tight_layout()
                gspec.update(wspace=0.025, hspace=0.05)
                for x in range(exs):
                    for y in range(ys):
                        ax = fig.add_subplot(gspec[x,y])
                        if self.planet == "Sun" or self.planet == "Moon" or self.planet == "Neptune":
                            ax.plot(self.request(band, 'wave0'), im)
                        else:
                            ax.plot(self.request(band, 'wave0'), im[:,x,y])
                        ax.set_ylim(bottom=0, top=8000)
                        #for t in self.request(band, 'tels'):
                        #    ax.axvspan(t[0], t[1], facecolor='olivedrab', alpha=0.4)
                        #for e in self.request(band, 'edges'):
                        #    ax.axvspan(e[0], e[1], facecolor='firebrick', alpha=0.4)
                        #for idx, c in enumerate(self.request(band, 'colors')):
                        #    ax.axvspan(c[0], c[1], facecolor=self.request(band, 'colornames')[idx], alpha=0.4)
                        ax.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
                fig.suptitle('Neptune, Raw')
                plt.show()

    def collapse_planet(self):

        # Collapses 3D IFU data into 2D using medians. Save as new files

        for band in self.bands:

            # Finds the median of the IFU spectral data for an image and writes it to a new file
            # Removes the X and Y (spacial) axes information and moves the Z spectral axis to where
            # Programs like Molecfit expect to find it in the header
            specs = self.request(band, 'raw')
            for k in specs.keys():
                with fits.open(k) as hdul:
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
                    if self.planet == 'Saturn':
                        newf.writeto(k[:79] + 'Median' + k[79:], overwrite=True)
                    elif self.planet == 'Neptune':
                        newf.writeto(k[:76] + 'Median' + k[76:], overwrite=True)
                    elif self.planet == 'Enceladus-Titan':
                        newf.writeto(k[:84] + 'Median' + k[84:], overwrite=True)
                    else:
                        print("Go to the code of collapse planet and figure out where 'Median' should be entered "
                              "in the filename")

            # Plots the last file for the user to double check
            w, h = plt.figaspect(0.3)
            fig = plt.figure(22, figsize=(w, h))
            ax = fig.add_subplot(111)
            ax.plot(self.request(band, 'wave0'), m)
            for t in self.request(band, 'tels'):
                ax.axvspan(t[0], t[1], facecolor='olivedrab', alpha=0.4)
            for e in self.request(band, 'edges'):
                ax.axvspan(e[0], e[1], facecolor='firebrick', alpha=0.4)
            plt.show()

    def sorting_hat(self, *args, **kwargs):

        # Sorts spectra into regions based off of the similarities between them
        if 'prespecs' in args:
            self.averages_planet('prespecs')
        else:
            self.averages_planet()

        # Retrieves the spectra and mask information for each band
        for band in self.bands:
            if band == 'uvb':
                self.Uregions = {}
                regions = self.Uregions
                free = [0.3834, 0.4455]
            elif band == 'vis':
                self.Vregions = {}
                regions = self.Vregions
                free = [0.880, 0.890] # Methane
                free2 = [0.64, 0.66]  # Ammonia
            elif band == 'nir':
                self.Nregions = {}
                regions = self.Nregions
                free = [1.5439, 1.5674]
            else:
                regions = {}
                free = []

            pretty_colors = ['indigo', 'rebeccapurple', 'mediumpurple', 'darkblue', 'royalblue', 'skyblue',
                             'teal', 'mediumturquoise', 'paleturquoise', 'darkgreen', 'forestgreen',
                             'limegreen', 'goldenrod', 'gold', 'yellow', 'darkorange', 'orange',
                             'moccasin', 'darkred', 'red', 'lightcoral']

            if 'n_clusters' in kwargs.keys():
                n_clusters = kwargs['n_clusters']
            else:
                n_clusters = 3

            specs = self.request(band, 'molecfit')
            keys = sorted(list(specs.keys()))
            ordered_specs = []
            ordered_rchis = []
            wavelength = []
            wavelength2 = []
            for i, key in enumerate(keys):
                ordered_specs.append(specs[key])
                wavelength.append(self.spectral_index(self.request(band, 'wave0'), specs[key], free))
                if band =='vis':
                    wavelength2.append(self.spectral_index(self.request(band, 'wave0'), specs[key], free2))
                #wavelength.append(specs[key][7624])
                ordered_rchis.append(self.rchis[band][i])
            if band == 'vis':
                arr = np.array([wavelength, wavelength2]).transpose()
            else:
                arr = np.array([ordered_rchis, wavelength]).transpose()
            km = KMeans(init='k-means++', n_clusters=n_clusters).fit(arr)
            region_placement = km.labels_
            for i, r in enumerate(region_placement):
                region = 'Region_' + str(r)
                try:
                    regions[region].append(keys[i])
                except KeyError:
                    regions[region] = [keys[i]]

            centroids = km.cluster_centers_
            w, h = plt.figaspect(1.0)
            fig1 = plt.figure(1, figsize=(w, h))
            ax = fig1.add_subplot(111)
            for i,r in enumerate(region_placement):
                ax.plot(arr[i,0], arr[i,1], 'o', color=pretty_colors[r])
            ax.plot(centroids.transpose()[0], centroids.transpose()[1], 'o', color='mediumturquoise')
            ax.set_title(self.planet + ' Centroids for ' + band.upper() + ' Band')
            if band == 'vis':
                ax.set_xlabel('Methane')
                ax.set_ylabel('Ammonia')
            else:
                ax.set_xlabel('Reduced Chi Squared')
                ax.set_ylabel('Spectral Index')

            plt.savefig('/home/gmansir/Thesis/' + self.planet + '/' + band + '_centroid_plot.png')
            plt.show()

            # Finds the silhouette score for the region sorting. scores are bound between -1 and +1, with
            # values closer to +1 indicating better defined clusters
            silhouette = metrics.silhouette_score(ordered_specs, region_placement, metric='euclidean')
            print('The silhouette score for the ' + band.upper() + ' band is: ' + str(silhouette))

    def spectral_index(self, wave0, spec, free):

        abs_diff_func1 = lambda x : abs(x-free[0])
        close1 = min(wave0, key=abs_diff_func1)
        idx1 = list(wave0).index(close1)
        abs_diff_func2 = lambda x : abs(x-free[1])
        close2 = min(wave0, key=abs_diff_func2)
        idx2 = list(wave0).index(close2)

        x = wave0[idx1:idx2]
        y = spec[idx1:idx2]

        return sp.integrate.simps(1-y, x)

    def averages_regions(self, *args):
        # Calculates the averages and standard deviations for each band of a planet
        # Calculates the reduced chi squareds for each spectrum compared to the average for the
        # region it is designated to.

        # Retrieves the spectra and mask information for each band
        for band in self.bands:

            # Applies the mask to each spectra per band and region, normalizes the spectrum using the mask,
            # then compiles all normalized spectra into a list to find the average and standard deviation
            # per band and region.
            regions = self.request(band, 'regions')
            specs = self.request(band, 'molecfit')
            for region in regions:
                data = {}
                for fname in regions[region]:
                    data[fname] = self.normalize_spec(self.request(band, 'msk'), specs[fname])
                aspecs = np.median([*data.values()], axis=0)
                sdspecs = np.std([*data.values()], axis=0)

                # Avoids a divide by zero error
                for ii, ss in enumerate(sdspecs):
                    if ss == 0.:
                        sdspecs[ii] = 0.1

                # Computes the chi squared for each spectrum compared to the average for the band and region
                # Then divides that by the degrees of freedom (length of spectrum) for the reduced chi squared
                rchis = []
                for fname in sorted(regions[region]):
                    spec = data[fname]
                    chi = chisquare((spec - aspecs) / sdspecs)
                    rchis.append(abs(chi[0]) / len(spec))

                # Saves everything for future use
                self.aspecs[band + region] = aspecs
                self.sdspecs[band+ region] = sdspecs
                self.rchis[band + region] = rchis

                #if 'predata' in args:
                #    data = {}
                #    for fname in regions[region]:
                #        data[fname] = self.normalize_spec(msk, specs[fname])
                #    aspecs = np.median([*data.values()], axis=0)
                #    sdspecs = np.std([*data.values()], axis=0)

    def averages_planet(self, *args):

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

            # Applies the mask to each spectra per band, normalizes the spectrum using the mask,
            # then compiles all normalized spectra into a list to find the average and standard deviation
            # per band and region.
            data = {}
            specs = self.request(band, 'molecfit')
            for fname in specs:
                data[fname] = self.normalize_spec(self.request(band, 'msk'), specs[fname])
            aspecs = np.median([*data.values()], axis=0)
            if band == 'vis':
                aspecs -= 0.001
            if band == 'nir':
                aspecs -= 0.024
            sdspecs = np.std([*data.values()], axis=0)

            # Avoids a divide by zero error
            for ii, ss in enumerate(sdspecs):
                if ss == 0.:
                    sdspecs[ii] = 0.000001

            # Computes the chi squared for each spectrum compared to the average for the band and region
            # Then divides that by the degrees of fredom (length of spectrum) for the reduced chi squared
            rchis = []
            keys = sorted(list(specs.keys()))
            for fname in keys:
                spec = data[fname]
                chi = chisquare((spec - aspecs) / sdspecs)
                rchis.append(abs(chi[0]) / len(spec))

            if 'prespecs' in args:
                data = {}
                prespecs = self.request(band, 'collapsed')
                for fname in prespecs:
                    data[fname] = self.normalize_spec(self.request(band, 'msk'), prespecs[fname])
                preaspecs = np.median([*data.values()], axis=0)
                if band == 'vis':
                    preaspecs -= 0.001
                if band == 'nir':
                    preaspecs -= 0.024
                presdspecs = np.std([*data.values()], axis=0)
                self.preaspecs[band] = preaspecs
                self.presdspecs[band] = presdspecs

            # Saves everything for future use
            self.aspecs[band] = aspecs
            self.sdspecs[band] = sdspecs
            self.rchis[band] = rchis

    def moving_average(self, **kwargs):

        # Calculates the moving average of a spectrum to remove the "spikes" in the data
        if 'plot' in kwargs:
            plot = kwargs['plot']
        else:
            plot = False
        if 'window' in kwargs:
            window = kwargs['window']
        else:
            window = 25
        if 'sigval' in kwargs:
            sigval = kwargs['sigval']
        else:
            sigval = 3

        for band in self.bands:
            specdf = pd.DataFrame(zip(self.request(band, 'wave0'), self.aspecs[band]), columns=['wavelength','flux'])
            specdf.set_index('wavelength', inplace=True)
            specdf['rolling'] = specdf.rolling(window, center=True, min_periods=1).median()
            specdf['residuals'] = specdf['flux'] - specdf['rolling']
            sigclip = sigma_clip(specdf['residuals'], sigma=sigval, cenfunc='median')
            specdf['fluxclip'] = np.ma.masked_array(specdf['flux'], mask=sigclip.mask, fill_value=np.nan).filled()
            self.fluxclip = specdf['fluxclip'].values
            if plot == True:
                colors=['magenta', 'royalblue', 'mediumpurple', 'coral', 'skyblue']
                w, h = plt.figaspect(0.3)
                specdf.plot(color=colors, figsize=(w,h))
                plt.legend(labels = ['Flux', 'Rolling Average', 'Residuals', 'Clipped Rolling Average', 'Clipped Flux'])
                plt.title(self.planet + ' Rolling Average')
                plt.show()

    def plot_averages_planet(self, *args, **kwargs):

        # Plots the averages of the different bands for a given planet.
        # Includes an option for boxcar smoothing.
        # Highlights regions of strong telluric absorption, and regions masked from the average calculation
        # Saves figure.
        w, h = plt.figaspect(0.3)
        fig1 = plt.figure(1, figsize=(w, h))
        ax = fig1.add_subplot(111)
        if 'prespecs' in args:
            for band in self.bands:
                ax.plot(self.request(band, 'wave0'), self.preaspecs[band], color='mediumvioletred', label='premolecfit')

        for band in self.bands:
            ax.plot(self.request(band, 'wave0'), self.aspecs[band], color='gold', label=band)
            #ax.plot(self.request(band, 'wave0'), self.aspecs[band], color=self.request(band, 'speccolor'), label=band)
            #for t in self.request(band, 'tels'):
            #    ax.axvspan(t[0], t[1], facecolor='olivedrab', alpha=0.4)
            #for e in self.request(band, 'edges'):
            #    ax.axvspan(e[0], e[1], facecolor='firebrick', alpha=0.4)
            #for i, c in enumerate(self.request(band, 'colors')):
            #    ax.axvspan(c[0], c[1], facecolor=self.request(band, 'colornames')[i], alpha=0.4)

        ax.legend()
        ax.set_title(self.planet + '-like Star Averages')
        ax.set_xlabel('Wavelength (microns)')
        ax.set_ylabel('Normalized Flux')
        if 'xlims' in kwargs:
            ax.set_xlim(left=kwargs['xlims'][0], right=kwargs['xlims'][1])
        if 'ylims' in kwargs:
            ax.set_ylim(bottom=kwargs['ylims'][0], top=kwargs['ylims'][1])

        plt.savefig('/home/gmansir/Thesis/'+self.planet+'/average_plot.png')
        plt.show()

    def plot_averages_regions(self, *args):

        # Plots the averages of the different bands for a given planet.
        # Includes an option for boxcar smoothing.
        # Highlights regions of strong telluric absorption, and regions masked from the average calculation
        # Saves figure.

        pretty_colors = ['indigo', 'rebeccapurple', 'mediumpurple', 'darkblue', 'royalblue', 'skyblue',
                         'teal', 'mediumturquoise', 'paleturquoise', 'darkgreen', 'forestgreen',
                         'limegreen', 'goldenrod', 'gold', 'yellow', 'darkorange', 'orange',
                         'moccasin', 'darkred', 'red', 'lightcoral']
        count = 0

        w, h = plt.figaspect(0.3)
        fig1 = plt.figure(1, figsize=(w, h))
        ax = fig1.add_subplot(111)
        #ax1 = fig1.add_subplot(211)
        for band in self.bands:

            for region in self.request(band, 'regions'):
                ax.plot(self.request(band, 'wave0'), self.aspecs[band+region], label=band+region,
                        color=pretty_colors[count])
                #ax1.plot(self.request(band, 'wave0'), self.sdspecs[band+region], label=band+region,
                #         color=pretty_colors[count])
                count += 1
            for t in self.request(band, 'tels'):
                ax.axvspan(t[0], t[1], facecolor='olivedrab', alpha=0.4)
                #ax1.axvspan(t[0], t[1], facecolor='olivedrab', alpha=0.4)
            for e in self.request(band, 'edges'):
                ax.axvspan(e[0], e[1], facecolor='firebrick', alpha=0.4)
                #ax1.axvspan(e[0], e[1], facecolor='firebrick', alpha=0.4)
            for i, c in enumerate(self.request(band, 'colors')):
                ax.axvspan(c[0], c[1], facecolor=self.request(band, 'colornames')[i], alpha=0.4)
                #ax1.axvspan(c[0], c[1], facecolor=self.request(band, 'colornames')[i], alpha=0.4)

        ax.legend()
        ax.set_title(self.planet + ' Averages')
        ax.set_xlabel('Wavelength (microns)')
        ax.set_ylabel('Normalized Flux')
        ax.set_ylim(top=1.001) #, bottom=0.98

        plt.savefig('/home/gmansir/Thesis/' + self.planet + '/average_plot.png')
        plt.show()

    def collect_sun_data(self):

        #Gathers the data of the solar spectrum
        flist = glob.glob('/home/gmansir/Thesis/Sun/Data/PostMolecfit/file*')
        flist.sort()
        ws = []
        fs = []
        es = []
        for f in flist:
            data = np.loadtxt(f).transpose()
            ws.append(data[0])
            fs.append(data[1])
            es.append(data[2])
        lamb = []
        flux = []
        for i in range(len(ws[0:8])):
            i += 2
            lamb += ws[i].tolist()
            flux += fs[i].tolist()
        lamb = np.array(lamb)*0.0001
        flux = np.array(flux)/np.max(np.array(flux))

        #Remove the continuum from the solar spec
        pabu = pd.DataFrame(flux)
        windows = pd.DataFrame.rolling(pabu, 1000)
        smooth_flux = np.array(windows.max())
        smooth_flux = np.reshape(smooth_flux, np.shape(smooth_flux)[0])
        smooth_flux[0:999] = 1.0
        coeffs = np.polyfit(lamb, smooth_flux, 2)
        continuum = coeffs[0]*lamb**2+coeffs[1]*lamb + coeffs[2]

        self.sun_lamb = lamb.tolist()
        self.sun_flux = flux/continuum

    def convolve_sun(self, *args, **kwargs):

        if 'kernel_stddev' in kwargs:
            kernel_stddev = kwargs['kernel_stddev']
        else:
            kernel_stddev = 20
        kernel = Gaussian1DKernel(stddev=kernel_stddev)
        con_sun_flux = convolve(self.sun_flux, kernel)

        if 'plot_con_results' in args:
            w, h = plt.figaspect(0.3)
            fig1 = plt.figure(1, figsize=(w, h))
            plt.plot(self.sun_lamb, self.sun_flux, c='rebeccapurple', label='Sun')
            plt.plot(self.sun_lamb, con_sun_flux, c='turquoise', alpha=0.8, label='Convolved')
            plt.legend()
            plt.title('Convolution Results')
            plt.show()

        if 'overplot_results' in args:
            w, h = plt.figaspect(0.3)
            fig1 = plt.figure(1, figsize=(w, h))
            if 'offset' in kwargs:
                offset = kwargs['offset']
            else:
                offset = 0.0
            plt.plot(self.sun_lamb, (self.sun_flux)+offset, c='slateblue', label='Sun', alpha=0.8)
            plt.plot(self.sun_lamb, (con_sun_flux)+offset, c='turquoise', alpha=0.8, label='Convolved')


        return con_sun_flux

    def plot_rms(self, mini_sun_flux, mini_hip_flux, line, rms, l):

        plt.plot(mini_sun_flux, mini_hip_flux, 'o', color='rebeccapurple')
        plt.plot(mini_sun_flux, line, c='magenta')
        plt.title('Line ' + str(l) + ' Sun v Hipparcos, RMS = ' + str(rms))
        plt.xlabel('Sun Flux')
        plt.ylabel('Hipparcos Flux')
        plt.show()

    def plot_fwhm(self, mini_lamb, mini_sun_flux, line, mini_hip_flux, peaks, heights, xmins, xmaxes, l,
                  hip_peaks, hip_heights, hip_xmins, hip_xmaxes):

        # Plots the sun and hipparcos data with the peaks and fwhm identified
        mini_flux = mini_sun_flux * line
        plt.plot(mini_lamb, mini_flux, c='turquoise', label='Sun')
        plt.plot(mini_lamb, mini_hip_flux, c=self.request('vis', 'speccolor'), label='Hipparcos Star')
        for i in range(len(peaks)):
            plt.plot(mini_lamb[peaks[i]], mini_flux[peaks[i]], 'x')
            plt.hlines(heights[i], mini_lamb[int(xmins[i])], mini_lamb[int(xmaxes[i])], color='rebeccapurple')
        for i in range(len(hip_peaks)):
            plt.plot(mini_lamb[hip_peaks[i]], mini_hip_flux[hip_peaks[i]], 'x')
            plt.hlines(hip_heights[i], mini_lamb[int(hip_xmins[i])], mini_lamb[int(hip_xmaxes[i])], color='purple')
        plt.title('Line ' + str(l))
        plt.legend()
        plt.show()

    def plot_resolution(self, res, hip_res, l):

        # Plots the resolution of each identified peak (lambda/fwhm)

        for r in res:
            plt.plot(r[0], r[1], 'o', c='turquoise')
        for r in hip_res:
            plt.plot(r[0], r[1], 'o', c=self.request('vis', 'speccolor'))
        plt.title('Line ' + str(l) + ' Resolutions')
        plt.xlabel('Wavelength (microns)')
        plt.ylabel('Resolution using pixels')
        plt.show()

    def resolution_compare(self, *args):

        self.collect_sun_data()
        lines = [0.69385, 0.7446, 0.7925, 0.830, 0.899, 0.9875]
        convolution_kernels = [14.4, 10.5, 20.4, 17.9, 17.9, 16.9]
        self.averages_planet()
        hip_lamb = list(self.request('vis', 'wave0'))
        hip_flux = self.aspecs['vis']

        for l in lines:
            # Finds the index of wavelengths in my desired range
            con_sun_flux = self.convolve_sun(kernel_stddev=convolution_kernels[lines.index(l)])
            low = l - 0.0005
            high = l + 0.0005
            sun_low = self.closest_index(self.sun_lamb, low)
            sun_high = self.closest_index(self.sun_lamb, high)
            hip_low = self.closest_index(hip_lamb, low)
            hip_high = self.closest_index(hip_lamb, high)

            # Clips the working spectra to this range, interpolates the sun data to the
            # xshooter wavelength values, and inverts the flux to work with the scipy algorithms
            mini_sun_lamb = self.sun_lamb[sun_low:sun_high]
            mini_sun_flux = 1/con_sun_flux[sun_low:sun_high] - 1
            min_val = min(mini_sun_flux)
            if min_val >= 0.00:
                mini_sun_flux -= min_val
            int_func = interpolate.interp1d(mini_sun_lamb, mini_sun_flux, fill_value='extrapolate')
            mini_lamb = hip_lamb[hip_low:hip_high]
            mini_hip_flux = 1/hip_flux[hip_low:hip_high] - 1
            mini_sun_flux = int_func(mini_lamb)
            min_val = min(mini_hip_flux)
            if min_val != 0.00:
                mini_hip_flux -= min_val

            # Finds the linear scaling between the two data sets and the root-mean-square
            coeffs = np.polyfit(mini_sun_flux, mini_hip_flux, 1)
            line = mini_sun_flux*coeffs[0]+coeffs[1]
            ms = np.sum((mini_hip_flux-line)**2)/len(mini_hip_flux)
            rms = ms**0.5

            # Finds the peaks and the fwhm of the data within my working region
            peaks = list(sp.signal.find_peaks(mini_sun_flux * line, height=0.0015)[0])
            widths,heights,xmins,xmaxes = list(sp.signal.peak_widths(mini_sun_flux * line, peaks, rel_height=0.5))
            hip_peaks = list(sp.signal.find_peaks(mini_hip_flux, height=0.0015)[0])
            hip_widths,hip_heights,hip_xmins,hip_xmaxes = list(sp.signal.peak_widths(mini_hip_flux, hip_peaks, rel_height=0.5))

            # Finds the resolution of the peaks
            res = []
            for i in range(len(peaks)):
                res.append([mini_lamb[peaks[i]], peaks[i] / widths[i]])
            hip_res = []
            for i in range(len(hip_peaks)):
                hip_res.append([mini_lamb[hip_peaks[i]], hip_peaks[i] / hip_widths[i]])

            # Plots whatever was requested by the user
            if 'plot_rms' in args:
                self.plot_rms(mini_sun_flux, mini_hip_flux, line, rms, l)
            if 'plot_fwhm' in args:
                self.plot_fwhm(mini_lamb, mini_sun_flux, line, mini_hip_flux, peaks, heights, xmins, xmaxes, l,
                               hip_peaks, hip_heights, hip_xmins, hip_xmaxes)
            if 'plot_resolution' in args:
                self.plot_resolution(res, hip_res, l)

    def minimize_convolution_kernel(self):

        # Computes the rms for a range of convolution kernels and plots the results
        self.collect_sun_data()
        lines = [0.69385, 0.7446, 0.7925, 0.830, 0.899, 0.9875]
        test_convolution_kernels = list(np.linspace(1,100,200))
        self.averages_planet()
        self.convolve_sun()
        hip_lamb = list(self.request('vis', 'wave0'))
        hip_flux = self.aspecs['vis']

        for l in lines:
            low = l - 0.0005
            high = l + 0.0005
            sun_low = self.closest_index(self.sun_lamb, low)
            sun_high = self.closest_index(self.sun_lamb, high)
            hip_low = self.closest_index(hip_lamb, low)
            hip_high = self.closest_index(hip_lamb, high)
            mini_sun_lamb = self.sun_lamb[sun_low:sun_high]
            mini_lamb = hip_lamb[hip_low:hip_high]
            mini_hip_flux = 1 / hip_flux[hip_low:hip_high] - 1

            rmses = []
            for t in test_convolution_kernels:
                print('Testing kernel ' + str(test_convolution_kernels.index(t)+1) + ' of '
                      + str(len(test_convolution_kernels)))
                test_flux = self.convolve_sun(kernel_stddev=t)
                mini_test_flux = 1 / test_flux[sun_low:sun_high] - 1
                min_val = min(mini_test_flux)
                if min_val >= 0.00:
                    mini_test_flux -= min_val + 0.01
                int_func = interpolate.interp1d(mini_sun_lamb, mini_test_flux, fill_value='extrapolate')
                mini_test_flux = int_func(mini_lamb)
                coeffs = np.polyfit(mini_test_flux, mini_hip_flux, 1)
                line = mini_test_flux * coeffs[0] + coeffs[1]
                ms = np.sum((mini_hip_flux - line) ** 2) / len(mini_hip_flux)
                rms = ms ** 0.5
                rmses.append(rms)
            min_rms = rmses.index(min(rmses))
            plt.plot(test_convolution_kernels, rmses)
            plt.xlabel('Convolution Kernel Standard Deviation')
            plt.ylabel('RMS')
            plt.title('RMS Minimization, Line ' + str(l) + ', Min = ' + str(test_convolution_kernels[min_rms]))
            plt.show()

    def remove_outliers(self):

        # Retrieves the spectra and mask information for each band
        for band in self.bands:

            # Checks if a spectrum's rchi is more than 2 standard deviations away from the average in that
            # region. If it is, then it removes that spectrum from future analysis.
            outliers = []
            regions = self.request(band, 'regions')
            specs = self.request(band, 'molecfit')
            for region in regions:
                bad = []
                key = band+region
                achi = np.median(self.rchis[key])
                sdchi = np.std(self.rchis[key])
                hchi = achi + sdchi * 2
                lchi = achi - sdchi * 2
                for i, chi in enumerate(self.rchis[key]):
                    if hchi < chi or lchi > chi:
                        bad.append(sorted(regions[region])[i])
                        print('Removed ', sorted(regions[region])[i], ' from data. Average: ', achi, ', Spec: ', chi, '.')
                for b in bad:
                    regions[region].remove(b)
                    specs.pop(b)
                if len(regions[region]) == 1:
                    print('Removed ', regions[region][0], ' from data.')
                    specs.pop(regions[region][0])
                    outliers.append(region)
            for out in outliers:
                regions.pop(out)

            # Reapplies the mask to each spectra per band and region, normalizes the spectrum using the mask,
            # then compiles all normalized spectra into an updated list to find the average and standard deviation
            # per band and region with the outliers removed.
            for region in regions:
                data = {}
                for fname in regions[region]:
                    data[fname] = self.normalize_spec(self.request(band, 'msk'), specs[fname])
                aspecs = np.median(list(data.values()), axis=0)
                sdspecs = np.std(list(data.values()), axis=0)

                # Avoids a divide by zero error
                for ii, ss in enumerate(sdspecs):
                    if ss == 0.:
                        sdspecs[ii] = 0.1

                # Computes the chi squared for each spectrum compared to the average for the band and region
                # Then divides that by the degrees of fredom (length of spectrum) for the reduced chi squared                rchis = []
                rchis = []
                for fname in regions[region]:
                    spec = specs[fname]
                    chi = chisquare((spec - aspecs) / sdspecs)
                    rchis.append(abs(chi[0]) / len(spec))

                # Saves everything for future use
                self.aspecs[band+region] = aspecs
                self.sdspecs[band+region] = sdspecs
                self.rchis[band+region] = rchis

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
            # Plots each spectra in the given band and region, along with the average and stdev
            w, h = plt.figaspect(0.5)
            fig = plt.figure(1, figsize=(w, h))
            gspec = gridspec.GridSpec(len(self.request(band, 'molecfit'))+2, 1)
            fig.tight_layout()
            wave = self.request(band, 'wave')
            specs = self.request(band, 'molecfit')
            for num, fname in enumerate(specs):
                ax = fig.add_subplot(gspec[num, 0])
                spec = self.normalize_spec(self.request(band, 'msk'), specs[fname])
                ax.plot(wave[fname], spec)
                #ax.set_ylim(bottom=0, top=1)
                for t in self.request(band, 'tels'):
                    ax.axvspan(t[0], t[1], facecolor='olivedrab', alpha=0.4)
                for e in self.request(band, 'edges'):
                    ax.axvspan(e[0], e[1], facecolor='firebrick', alpha=0.4)
                for idx, c in enumerate(self.request(band, 'colors')):
                    ax.axvspan(c[0], c[1], facecolor=self.request(band, 'colornames')[idx], alpha=0.4)
                ax.tick_params(bottom=False, labelbottom=False)
            #avax = fig.add_subplot(gspec[len(specs), 0])
            #avax.plot(wave0, self.aspecs[band], color='orchid')
            #avax.set_ylim(bottom=0, top=1)
            #sdax = fig.add_subplot(gspec[len(specs)+1, 0])
            #sdax.plot(wave0, self.sdspecs[band], color='darkorchid')
            #sdax.set_ylim(bottom=0, top=1)
            fig.suptitle(band.upper())
            plt.show()

    def plot_all_moon(self):

        # Plots all normalized spectra in each band and region, alongside the average and stdev

        # Determines what data to use depending on the user input
        for band in self.bands:

            # Determines the region for the spectra in each band as listed in their filenames
            hlist = []
            mlist = []
            dlist = []
            specs = self.request(band, 'molecfit')
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
                wave = self.request(band, 'wave')
                for num, fname in enumerate(r):
                    if num == 0:
                        ax0 = fig.add_subplot(gspec[num, 0])
                        spec = self.normalize_spec(self.request(band, 'msk'), specs[fname])
                        ax0.plot(wave[fname], spec)
                    else:
                        ax = fig.add_subplot(gspec[num, 0], sharex=ax0)
                        spec = self.normalize_spec(self.request(band, 'msk'), specs[fname])
                        ax.plot(wave[fname], spec)
                    avax = fig.add_subplot(gspec[len(r), 0])
                    avax.plot(self.request(band, 'wave0'), self.aspecs[let + band], color='orchid')
                    sdax = fig.add_subplot(gspec[len(r)+1, 0])
                    sdax.plot(self.request(band, 'wave0'), self.sdspecs[let + band], color='darkorchid')
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

    def ifu_recovery(self, *args):

        # Determines the telluric spectrum for each frame, then applies it to the premolecfit data to
        # recover the spacial information from the IFU by dividing each pixel with the telluric average

        # Retrieves the spectra and mask information for each band
        for band in self.bands:

            # Applies the mask to each spectrum of the collapsed data and normalizes them
            # Orders the files so that they will match with the other dataset
            collnorm = {}
            collapsed = self.request(band, 'collapsed')
            for fname in collapsed:
                if band == 'vis:':
                    collnorm[fname] = self.normalize_spec(self.request(band, 'msk'), collapsed[fname][:24317])
                else:
                    collnorm[fname] = self.normalize_spec(self.request(band, 'msk'), collapsed[fname])
            collkeys = sorted(list(collnorm.keys()))

            # Applies the mask to each spectrum of the post-molecfit data and normalizes them
            postnorm = {}
            post = self.request(band, 'molecfit')
            for fname in post:
                postnorm[fname] = self.normalize_spec(self.request(band, 'msk'), post[fname])
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
                    ax.plot(self.request(band, 'wave0'), telluricspecs[fname])
                    ax.set_ylim(bottom=-50, top=50)
                    for t in self.request(band, 'tels'):
                        ax.axvspan(t[0], t[1], facecolor='olivedrab', alpha=0.4)
                    for e in self.request(band, 'edges'):
                        ax.axvspan(e[0], e[1], facecolor='firebrick', alpha=0.4)
                    for idx, c in enumerate(self.request(band, 'colors')):
                        ax.axvspan(c[0], c[1], facecolor=self.request(band, 'colornames')[idx], alpha=0.4)
                    if i != len(collkeys)-1:
                        ax.tick_params(bottom=False, labelbottom=False)
                fig.suptitle(self.planet + ' ' + band.upper() + ' Post / Pre Molecfit Ratio')
                plt.show()

            # Uses the ratio to remove telluric lines from IFU data
            raw = self.request(band, 'raw')
            rawkeys = sorted(list(raw.keys()))
            recovered = dict.fromkeys(rawkeys, np.zeros(np.shape(raw[rawkeys[0]])))
            for i, key in enumerate(rawkeys):
                ratio = telluricspecs[postkeys[i]]
                for x in range(np.shape(raw[key])[-2]):
                    for y in range(np.shape(raw[key])[-1]):
                        spec = np.array(raw[key])[:,x,y]
                        nspec = self.normalize_spec(self.request(band, 'msk'), spec)
                        rec = ratio * nspec
                        recovered[key][:,x,y] = rec
                newf = fits.PrimaryHDU(data=recovered[key], header=self.request(band, 'rawinfo')[key])
                newf.writeto(postkeys[i][:-5] + '_IFU' + postkeys[i][-5:], overwrite=True)
                print('Wrote file '+ postkeys[i][:-5] + '_IFU' + postkeys[i][-5:])

    def plot_ifu(self):

        # Plots all spectra for each pixel of each image

        # Determines what data to use depending on the user input
        for band in self.bands:

            # Plots each spectra in a grid cooresponding with i't position in the image
            specs = self.request(band, 'ifu')
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
                        ax.plot(self.request(band, 'wave0'), im[:,x,y])
                        ax.set_ylim(bottom=0.5, top=2.0)
                        for t in self.request(band, 'tels'):
                            ax.axvspan(t[0], t[1], facecolor='olivedrab', alpha=0.4)
                        for e in self.request(band, 'edges'):
                            ax.axvspan(e[0], e[1], facecolor='firebrick', alpha=0.4)
                        for idx, c in enumerate(self.request(band, 'colors')):
                            ax.axvspan(c[0], c[1], facecolor=self.request(band, 'colornames')[idx], alpha=0.4)
                        ax.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
                fig.suptitle(fname)
                plt.show()

    def normalize_spec(self, mask, spec):
        # Normalizes a spectrum using the mask defined in the init

        ms = ma.masked_array(spec, mask=mask)
        norm_spec = spec / np.median(ms)

        return norm_spec

    def save_simple_fits(self):
        for band in self.bands:
            if band == 'uvb':
                skipme = 4
            elif band == 'vis':
                skipme = 4
            elif band == 'nir':
                skipme = 5
            else:
                specs = {}
                wave = []
                skipme = 0

            specs = self.request(band, 'molecfit')
            for name in specs.keys():
                newname = 'Moon/Data/PostMolecfit/' + name[:skipme] + 'Reduced_' + name[skipme:-9] + '.txt'
                flux = np.array(specs[name])
                with open(newname, 'w+') as newf:
                    for idx in range(len(self.request(band, 'wave0'))):
                        l = str(wave[idx]) + ' ' + str(flux[idx]) + '\n'
                        newf.write(l)

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

            # Determines the region for the spectra in each band as listed in their filenames
            hlist = []
            mlist = []
            dlist = []
            specs = self.request(band, 'molecfit')
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
                    data[fname] = self.normalize_spec(self.request(band, 'msk'), specs[fname])
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

    def find_telluric_absorption(self, *args, **kwargs):
        # A temporary method to help me determine which bands have the most telluric absorption
        if 'prespecs' in args:
            self.sorting_hat_kmeans('prespecs')
        else:
            self.sorting_hat_kmeans()
        self.averages_regions()

        pretty_colors = ['indigo', 'rebeccapurple', 'mediumpurple', 'darkblue', 'royalblue', 'skyblue',
                         'teal', 'mediumturquoise', 'paleturquoise', 'darkgreen', 'forestgreen',
                         'limegreen', 'goldenrod', 'gold', 'yellow', 'darkorange', 'orange',
                         'moccasin', 'darkred', 'red', 'lightcoral']
        count = 0

        w, h = plt.figaspect(0.6)
        fig1 = plt.figure(1, figsize=(w, h))
        ax = fig1.add_subplot(111)
        ax1 = fig1.add_subplot(211)
        for band in self.bands:
            if band == 'uvb':
                strictness = 0.004
                strictness2 = 0.05
            elif band == 'vis':
                strictness = 0.002
                strictness2 = 0.07145
            elif band == 'nir':
                strictness = 0.05
                strictness2 = 0.999944
            else:
                strictness = 0.01
                strictness2 = 0.01

            if 'strictness' in kwargs.keys():
                strictness = kwargs['strictness']
            else:
                pass
            if 'strictness2' in kwargs.keys():
                strictness2 = kwargs['strictness2']
            else:
                pass

            total_telluric = []
            regions = self.request(band, 'regions')
            for region in regions:
                ax.plot(self.request(band, 'wave0'), self.sdspecs[band + region], label=band + region,
                        color=pretty_colors[count])
                stdmax = np.max(self.sdspecs[band+region])
                telluric = [self.request(band, 'wave0')[i] for i,x in enumerate(self.sdspecs[band+region]) if x >= stdmax*strictness]
                for t in telluric:
                    total_telluric.append(round(t, 4))
                #[total_telluric.append(round(t, 4)) for t in telluric]
                count += 1
            total_telluric = sorted(list(set(total_telluric)))
            start = []
            end = []
            for i in range(len(total_telluric)-1):
                diff = round(total_telluric[i+1] - total_telluric[i], 4)
                if diff == 0.0001:
                    start.append(total_telluric[i])
                    end.append(total_telluric[i+1])
            to_remove = []
            for s in start:
                if s in end:
                    to_remove.append(s)
            [start.remove(s) for s in to_remove]
            [end.remove(s) for s in to_remove]
            self.total_telluric = np.array([start, end]).transpose()
            for t in self.total_telluric:
                ax.axvspan(t[0], t[1], facecolor='olivedrab', alpha=0.4)
            for e in self.request(band, 'edges'):
                ax.axvspan(e[0], e[1], facecolor='firebrick', alpha=0.4)
            for i, c in enumerate(self.request(band, 'colors')):
                ax.axvspan(c[0], c[1], facecolor=self.request(band, 'colornames')[i], alpha=0.4)
            ax.legend()
            ax.set_title(self.planet + ' Standard Deviations')
            ax.set_xlabel('Wavelength (microns)')
            ax.set_ylabel('Normalized Flux')

            if 'prespecs' in args:
                total_telluric = []
                aspec = self.preaspecs[band]
                ax1.plot(self.request(band, 'wave0'), aspec, color=pretty_colors[1])
                stdmax = abs(np.min(aspec))
                telluric = [self.request(band, 'wave0')[i] for i, x in enumerate(aspec) if x <= stdmax / strictness2]
                for t in telluric:
                    total_telluric.append(round(t, 4))
                # [total_telluric.append(round(t, 4)) for t in telluric]
                total_telluric = sorted(list(set(total_telluric)))
                start = []
                end = []
                for i in range(len(total_telluric) - 1):
                    diff = round(total_telluric[i + 1] - total_telluric[i], 4)
                    if diff == 0.0001:
                        start.append(total_telluric[i])
                        end.append(total_telluric[i + 1])
                to_remove = []
                for s in start:
                    if s in end:
                        to_remove.append(s)
                [start.remove(s) for s in to_remove]
                [end.remove(s) for s in to_remove]
                self.total_telluric2 = np.array([start, end]).transpose()
                for t in self.total_telluric2:
                    ax1.axvspan(t[0], t[1], facecolor='olivedrab', alpha=0.4)
                for e in self.request(band, 'edges'):
                    ax1.axvspan(e[0], e[1], facecolor='firebrick', alpha=0.4)
                for i, c in enumerate(self.request(band, 'colors')):
                    ax1.axvspan(c[0], c[1], facecolor=self.request(band, 'colornames')[i], alpha=0.4)

        plt.savefig('/home/gmansir/Thesis/' + self.planet + '/stdev_plot.png')
        plt.show()
