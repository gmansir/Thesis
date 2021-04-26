import os
import pdb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from Planet_Analysis import PlanetAnalysis

class PlanetAnalysisWrapper:

    def __init__(self, **kwargs):

        # os.system('%run Planet_Analysis')
        self.planets = ['Neptune', 'Titan', 'Enceladus', 'Pure_Saturn', 'Sun']
        #sun = PlanetAnalysis(planet='Sun', reduced='a', bands=['vis'])
        #sun.collect_sun_data()
        #self.sun_lamb = sun.sun_lamb
        #self.sun_flux = sun.sun_flux*0.3+0.7

        # lines = [0.69385, 0.7446, 0.7925, 0.830, 0.899, 0.9875]
        if 'range' in kwargs:
            self.range = kwargs['range']
        else:
            self.range = [0.898, 0.9]
        if 'sep' in kwargs:
            self.sep = kwargs['sep']
        else:
            self.sep = 3.0

        self.waves = []
        self.fluxes = []


    def plot_planets(self):

        # Plots the specturm of every planet/object we have data for within a requested wavelength range for comparison
        uvb_min = 0.3
        uvb_max =  0.6
        vis_max = 1.024
        nir_max = 2.480
        colors = ['royalblue', 'rebeccapurple', 'mediumaquamarine',  'goldenrod', 'gold', 'gray']

        # Determines which band the minimum wavelength requested is in
        if self.range[0] < uvb_min:
            print("Declared range invalid. Please try again")
        elif self.range[0] <= uvb_max:
            band1 = 'uvb'
        elif self.range[0] <= vis_max:
            band1 = 'vis'
        elif self.range[0] <= nir_max:
            band1 = 'nir'
        else:
            print("Declared range invalid. Please try again")

        # Determines which band the maximum wavelength is in
        if self.range[1] > nir_max:
            print("Declared range invalid. Please try again")
        elif self.range[1] >= vis_max:
            band2 = 'nir'
        elif self.range[1] >= uvb_max:
            band2 = 'vis'
        elif self.range[1] >= uvb_min:
            band2 = 'uvb'
        else:
            print("Declared range invalid. Please try again")

        # Figures out which data bands we are working with
        if band1 == band2:
            bands = [band1]
        elif band1 == 'uvb' and band2 == 'nir':
            bands = ['uvb','vis','nir']
        else:
            bands = [band1, band2]

        # Collects the relevant wavelengths and data for each planet
        for p in self.planets:
            print(p + ' Data: ')
            Instance = PlanetAnalysis(planet=p, reduced='m', bands=bands)
            Instance.averages_planet()
            wave = []
            flux = []
            for b in bands:
                wave.append(Instance.request(band=b, request='wave0'))
                if p == 'Enceladus' or p == 'Titan' or p =='Pure_Saturn':
                    Instance.moving_average(window=10, sigval=5)
                    flux.append(Instance.fluxclip)
                else:
                    flux.append(Instance.aspecs[b])
            wave = list(np.reshape(wave, -1))
            flux = list(np.reshape(flux, -1))
            low_wave = Instance.closest_index(wave, self.range[0])
            high_wave = Instance.closest_index(wave, self.range[1])
            lilwave = wave[low_wave:high_wave]
            lilflux = flux[low_wave:high_wave]
            self.waves.append(lilwave)
            self.fluxes.append(lilflux)

        self.planets[4] = 'Sunlike_Star'
        w, h = plt.figaspect(0.5)
        fig = plt.figure(1, figsize=(w, h))
        fig.tight_layout()
        for num, wave in enumerate(self.waves):
            avflux = np.nanmedian(np.array(self.fluxes[num]))
            #for i,f in enumerate(np.array(self.fluxes[num])):
            #    if f >= avflux*4:
            #        np.array(self.fluxes[num])[i] = np.nan
            plt.plot(wave, np.array(self.fluxes[num])/avflux + (num*self.sep), color=colors[num], label=self.planets[num])
        plt.legend(loc='upper right')
        plt.title('Planet Comparison at ' + str(self.range))
        plt.ylim([0,14])
        plt.xlim([1.17, 1.02])
        plt.show()

