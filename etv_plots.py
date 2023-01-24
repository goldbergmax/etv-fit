import numpy as np
import pandas as pd
from chisq import EclipseFit
import matplotlib.pyplot as plt

class Plotter(EclipseFit):
    def __init__(self, system, **kwargs):
        EclipseFit.__init__(self, system, **kwargs)        

    def etv_residuals(self, x, star='A', phased=False, ecl_max=None, linearize=False):
        cmap = plt.cm.get_cmap('inferno')
        ecl_model, rv_model = self.get_residuals(x, tFin=ecl_max)
        if not ecl_max:
            ecl_max = self.ecl_data[star]['data_t'].max() + 50
        ecl_model[star] = ecl_model[star][ecl_model[star]['model_t'] < ecl_max]
        fig, ax = plt.subplots(2, figsize=(12, 8), gridspec_kw={'height_ratios':[3,1]},
                               sharex=True)
        T0, P = lsq_fit(ecl_model[star]['model_t'])
        Tp2 = x[6]
        lin_T0, lin_P = 0., 0.
        if linearize:
            lin_T0, lin_P = lsq_fit(ecl_model[star]['model_t'] - self.ecl_data[star]['data_t'])
        if self.system == '5095':
            P2 = 235.9 # Highest peak in Fourier transform of ETVs of KIC509
        else:
            P2 = x[5]

        ax[0].scatter(self.time_to_phase(ecl_model[star]['model_t'], Tp2, P2, phased),
                      86400*ecl_time_to_etv(ecl_model[star]['model_t'], P, T0),
                      c=ecl_model[star]['model_t'], cmap=cmap, marker='x', s=64)

        ax[0].errorbar(self.time_to_phase(self.ecl_data[star]['data_t'], Tp2, P2, phased),
                       86400*ecl_time_to_etv(self.ecl_data[star]['data_t'], P - lin_P, T0 - lin_T0), 
                       86400*self.ecl_data[star]['data_err'], linestyle='None', marker='o', 
                       markersize=6, color='0.5')

        ax[1].errorbar(self.time_to_phase(self.ecl_data[star]['data_t'], Tp2, P2, phased),
                       86400*ecl_time_to_etv(ecl_model[star]['res'], -lin_P, -lin_T0).dropna(),
                       86400*self.ecl_data[star]['data_err'], linestyle='None', 
                       marker='o', markersize=6, color='0.5')
        ax[1].axhline(0, linestyle='--', color='k')
        if phased:
            ax[1].set_xlabel('Planet Orbital Phase from Periapse')
        else:
            ax[1].set_xlabel('Time (BJD-2454900)')
        ax[0].set_ylabel('ETV (seconds)')
        plt.subplots_adjust(hspace=0)
        return ax
        
    def etv_together(self, x, ecl_max=None):
        ecl_model, rv_model = self.get_residuals(x, tFin=ecl_max)
        for i in self.ecl_stars:
            if not ecl_max:
                ecl_max = self.ecl_data[i]['data_t'].max() + 50
            ecl_model[i] = ecl_model[i][ecl_model[i]['model_t'] < ecl_max]
        fig, ax = plt.subplots(2, figsize=(12, 8), gridspec_kw={'height_ratios':[3,1]},
                               sharex=True)
        T0 = {}; P = {}
        T0['A'], P['A'] = lsq_fit(ecl_model['A']['model_t'])
        T0['B'], P['B'] = lsq_fit(ecl_model['B']['model_t'])
        P = (P['A'] + P['B'])/2
        colors = {'A':'r', 'B':'k'}
        for i in self.ecl_stars: 
            ax[0].plot(ecl_model[i]['model_t'],
                          86400*ecl_time_to_etv(ecl_model[i]['model_t'], P, T0[i]),
                          color=colors[i])

            ax[0].errorbar(self.ecl_data[i]['data_t'],
                           86400*ecl_time_to_etv(self.ecl_data[i]['data_t'], P, T0[i]), 
                           86400*self.ecl_data[i]['data_err'], linestyle='None', marker='o', 
                           markersize=6, color=colors[i])

            ax[1].errorbar(self.ecl_data[i]['data_t'],
                           86400*ecl_model[i]['res'].dropna(),
                           86400*self.ecl_data[i]['data_err'], linestyle='None', 
                           marker='o', markersize=6, color=colors[i])
        ax[1].axhline(0, linestyle='--', color='k')
        
    def cont_rv(self, x, rv_times):
        #gamma = x[-1]
        rv = {}
        for i in self.rv_stars:
            rv[i] = pd.DataFrame(index=range(len(rv_times)), columns=['time', 'rv'], dtype=float)
        rvcount = 0
        sim = self.set_up_sim(x)
        while rvcount < len(rv_times):
            sim.integrate(rv_times[rvcount])
            for i in self.rv_stars:
                rv[i].at[rvcount, 'time'] = sim.t
                rv[i].at[rvcount, 'rv'] = -sim.particles[i].vz*1731.45683681 #+ gamma
            rvcount += 1
        return rv

    def rv_residuals(self, x, phased=True):
        gamma = x[-self.num_rv_sources:]
        ecl_model, rv_model = self.get_residuals(x)
        T0, P = lsq_fit(ecl_model['A']['data_t'])
        start = self.rv_data['A']['time'].min()
        if phased:
            cont_rvs = self.cont_rv(x, np.linspace(start, start + P, 100, endpoint=False))
        else:
            stop = self.rv_data['A']['time'].max()
            start -= 5.; stop += 5.
            cont_rvs = self.cont_rv(x, np.linspace(start, stop, 1000))
        fig, ax = plt.subplots(2, figsize=(12,8), gridspec_kw={'height_ratios':[3,1]},
                               sharex=True)
        colors = {'A':'r', 'B':'k'}
        markers = {0:'o', 1:'^'}
        if self.system == '5095':
            labels = {0:'NOT-FIES', 1:'CAHA-CARMENES'}
        elif self.system == '782':
            labels = {0:'Tillinghast-TrES'}
        elif self.system == '3938':
            labels = {0:'Tillinghast-TrES'}     
        for i in self.rv_stars:
            arr_sort = np.argsort(self.time_to_phase(cont_rvs[i]['time'], T0, P, phased))
            ax[0].plot(self.time_to_phase(cont_rvs[i]['time'], T0, P, phased)[arr_sort], cont_rvs[i]['rv'][arr_sort], colors[i])
            for rv_idx, rv_data in self.rv_data[i].groupby('rv_idx'):
                ax[0].errorbar(self.time_to_phase(rv_data['time'], T0, P, phased), rv_data['rv'] - gamma[rv_idx], 
                               rv_data['rv_err'], linestyle='None', marker=markers[rv_idx], 
                               c=colors[i], label=labels[rv_idx])

                ax[1].errorbar(self.time_to_phase(rv_data['time'], T0, P, phased), rv_model[i][rv_model[i]['rv_idx'] == rv_idx]['res'], 
                               rv_data['rv_err'], linestyle='None', marker=markers[rv_idx], 
                               color=colors[i], label=labels[rv_idx])
        #ax[0].axhline(gamma, linestyle='--', color='k')
        ax[1].axhline(0, linestyle='--', color='k')
        if phased:
            ax[1].set_xlabel('Orbital Phase from Primary Eclipse')
        else:
            ax[1].set_xlabel('Time (BJD-2454900)')
        ax[0].set_ylabel('Radial Velocity (km/s)')
        plt.subplots_adjust(hspace=0)
        return ax
                       
    def time_to_phase(self, t, T0, P, phased=True):
        if not phased:
            return t
        #T0, P = lsq_fit(self.ecl_data['A']['data_t'])
        return np.remainder(t - T0, P)/P

    def impact_fit_quality(self, x):
        ecl_model, rv_model = self.get_residuals(x)
        dbdt, b0 = self.impact_regression(ecl_model)
        if not self.b:
            return
        print('Data:  {: 6.4} t + {:5.4}'.format(self.dbdt_data['A'], self.b0_data['A']))
        print('Model: {: 6.4} t + {:5.4}'.format(dbdt['A'], b0['A']))

def lsq_fit(ecl_time):
    df = ecl_time.dropna()
    A = np.vstack([np.ones(len(df)), df.index]).T
    return np.linalg.lstsq(A, df, rcond=None)[0]

def ecl_time_to_etv(ecl_time, P=None, T0=None):
    if P is None or T0 is None:
        T0, P = lsq_fit(ecl_time)
    return ecl_time - P*ecl_time.index.values - T0

def nice_units(x):
    y = x.copy()
    # convert angles to degrees
    y[...,2] *= 180/np.pi
    y[...,4] *= 180/np.pi
    y[...,9] *= 180/np.pi
    y[...,10] *= 180/np.pi
    # convert solar masses to jupiter masses
    y[...,13] *= 1047.58
    return y

def pretty_print(x):
    e2 = np.sqrt(x[7]**2 + x[8]**2)
    omega2 = np.degrees(np.arctan2(x[8], x[7]))
    retstring =  \
    '''
             P (d)    Epoch (d)   i (deg)        e    ω (deg)    Ω (deg)
Binary   {:>9.8}    {:>9.7}   {:>7.5}  {:>7.3}   {:> 8.4}   {:> 8.4}
Planet   {:>9.5}    {:>9.5}   {:>7.4}  {:>7.3}   {:> 8.4}   {:> 8.3}

M_A (Msolar) {:.4}
M_B (Msolar) {:.4}
M_p (Mjup)   {:.4}
k2           {:.4}
    '''.format(x[0], x[1], np.degrees(x[2]), x[3], np.degrees(x[4]), 0.,
               x[5], x[6], np.degrees(x[9]), e2, omega2, np.degrees(x[10]),
               x[11], x[12], 1047.5*x[13], x[14])
    for gamma in x[15:]:
        retstring += '\nγ (km/s)     {:.4}'.format(gamma)
    return retstring