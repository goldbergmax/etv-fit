import numpy as np
from scipy.stats import linregress
import pandas as pd
import rebound
import reboundx

class EclipseFit():
    def __init__(self, system, dt=None):
        self.system = system
        self.load_data()
        self.dt = dt
        self.R = {'A':1.2*0.00465, 'B':1.2*0.00465}

    def load_data(self):
        # system picker
        if self.system == '782':
            ecl_files = {'A':'../data/KID7821010/data782.b.tt.trans', 
                         'B':'../data/KID7821010/data782.c.tt.trans'}
            self.ecl_stars = ['A', 'B']
            rv_files = [{'A':'../data/KID7821010/kid007821010RVA.dat', 
                         'B':'../data/KID7821010/kid007821010RVB.dat'}]
            self.rv_stars = ['A', 'B']
            shift_index = {'A':33, 'B':33}
        elif self.system == '5095':
            ecl_files = {'A':'../data/KID5095269/koi509.tt.dan.db.try7.trans'}
            self.ecl_stars = ['A']
            rv_files = [{'A':'../data/KID5095269/kid005095269RVA.dat', 
                         'B':'../data/KID5095269/kid005095269RVB.dat'},
                        {'A':'../data/KID5095269/kid005095269RVA_carmenes.dat', 
                         'B':'../data/KID5095269/kid005095269RVB_carmenes.dat'}]
            self.rv_stars = ['A', 'B']
            shift_index = {'A':41}
        self.num_rv_sources = len(rv_files)
        self.ecl_data = {}
        for i in self.ecl_stars:
            self.ecl_data[i] = pd.read_csv(ecl_files[i], header=None, delim_whitespace=True, 
                                           index_col=0, names=['data_t', 'data_err'])
            self.ecl_data[i].index = self.ecl_data[i].index.astype(int)
            self.ecl_data[i].index = self.ecl_data[i].index + shift_index[i]

        self.rv_datas = {rv_star:[None for _ in range(self.num_rv_sources)] for rv_star in self.rv_stars}
        for rv_idx in range(self.num_rv_sources):
            for i in self.rv_stars:
                self.rv_datas[i][rv_idx] = pd.read_csv(rv_files[rv_idx][i], header=None, delim_whitespace=True, 
                                                      names=['time', 'rv', 'rv_err'])
                self.rv_datas[i][rv_idx]['rv_idx'] = rv_idx
                self.rv_datas[i][rv_idx]['time'] += 100
        self.rv_data = {i:pd.concat(self.rv_datas[i], ignore_index=True) for i in self.rv_stars}

        self.tFin = max(max(x['data_t'].max() for x in self.ecl_data.values()), self.rv_data['A']['time'].max()) + 100
        self.b = False
        if self.system == '5095':
            self.b = True
            self.b0_data = {'A':1.2954204562}
            self.b0_err = {'A':0.20507}
            self.dbdt_data = {'A':3.9914432001e-07}
            self.dbdt_err = {'A':1.6933e-07}

    # observer on the positive z-axis
    def get_residuals(self, els, safe=True, tFin=None):
        sim = self.set_up_sim(els)
        if sim is None:
            return None, None
        if not tFin or tFin < self.tFin:
            tFin = self.tFin
        pri_to_sec_gap, sec_to_pri_gap = self.est_ecl_steps(els)
        next_ecl = np.fmod(els[1], els[0])
        N = int(tFin/els[0])
        ecl_model = {i:pd.DataFrame(index=range(N), columns=['model_t', 'model_b'], dtype=float) for i in self.ecl_stars}
        rv_model = {i:pd.DataFrame(index=self.rv_data[i].index, columns=['time', 'rv_idx', 'rv'], dtype=float) for i in self.rv_stars}
        for i in self.rv_stars: rv_model[i][['time', 'rv_idx']] = self.rv_data[i][['time', 'rv_idx']]
        p = sim.particles
        def dotprod(params):
            terms = [(None, None), (None, None)]
            for i, param in enumerate(params):
                if param == 'x':
                    terms[i] = p[1].x - p[0].x, p[1].y - p[0].y
                elif param == 'v':
                    terms[i] = p[1].vx - p[0].vx, p[1].vy - p[0].vy
                elif param == 'a':
                    terms[i] = p[1].ax - p[0].ax, p[1].ay - p[0].ay
            return terms[0][0]*terms[1][0] + terms[0][1]*terms[1][1]

        def ps2():
            return (p['B'].x - p['A'].x)**2 + (p['B'].y - p['A'].y)**2   
        def ds2():
            return (p['B'].z - p['A'].z)**2
        
        ecl_count = {'A':0, 'B':0}
        rvcount = 0
        SPEEDFAC = 1731.45683681 # au/d to km/s
        C = 173.1 # speed of light in au/d

        while sim.t < tFin:
            # integrate to next eclipse, or next RV if closer
            while rvcount < len(self.rv_data['A']) and rv_model['A'].loc[rvcount, 'time'] < next_ecl:
                sim.integrate(rv_model['A'].loc[rvcount, 'time'])
                for i in self.rv_stars:
                    rv_model[i].loc[rvcount, 'rv'] = -p[i].vz*SPEEDFAC
                rvcount += 1
            sim.integrate(next_ecl)
            for i in range(10):
                # Newton-Raphson method to find the zero of x.v
                xdotv = dotprod('xv')
                xdota = dotprod('xa')
                vdotv = dotprod('vv')
                newton_step = xdotv/(xdota + vdotv)
                sim.integrate(sim.t - newton_step)
                if np.abs(newton_step) < 1e-10:
                    break
            if p['B'].z - p['A'].z > 0.: # primary eclipse
                ecl_type = 'A'
                next_ecl = sim.t + pri_to_sec_gap
            else: # secondary eclipse
                ecl_type = 'B'
                next_ecl = sim.t + sec_to_pri_gap
            #print(sim.t, ecl_type, p['B'].z - p['A'].z)
            if ecl_type in self.ecl_stars:
                com_diff = sim.calculate_com().z - sim.calculate_com(last=2).z
                t_ltte = com_diff/C
                ecl_model[ecl_type].loc[ecl_count[ecl_type], 'model_t'] = sim.t + t_ltte
                ecl_model[ecl_type].loc[ecl_count[ecl_type], 'model_b'] = np.sqrt(ps2())/self.R[ecl_type]
                ecl_count[ecl_type] += 1
            #rebound.OrbitPlot(sim, slices=True)
        gamma = els[-2:]
        for i in self.ecl_stars:
            ecl_model[i]['data_t'] = self.ecl_data[i]['data_t']
            ecl_model[i]['res'] = self.ecl_data[i]['data_t'] - ecl_model[i]['model_t']
            ecl_model[i]['data_err'] = self.ecl_data[i]['data_err']
        for i in self.rv_stars:
            rv_model[i]['rv'] += gamma[rv_model[i]['rv_idx']]
            rv_model[i]['res'] = self.rv_data[i]['rv'] - rv_model[i]['rv']
        # raise errors if eclipses or RVs not recorded correctly
        for i in self.rv_stars:
            if safe and np.any(rv_model[i].isna()):
                raise IndexError('Some RVs not recorded')
        for i in self.ecl_stars:
            if safe and np.any(ecl_model[i].dropna(subset=['data_t']).isna()):
                raise IndexError('Some eclipses not recorded')
        return ecl_model, rv_model

    def est_ecl_steps(self, els):
        # We must compute a time of secondary eclipse
        P1, T01, i1, e1, omega1, P2, Tp2, ecw2, esw2, i2, Omega2, mA, mB, mp, k1, *gamma = els
        E01 = 2*np.arctan(np.sqrt((1-e1)/(1+e1))*np.tan((np.pi/2 - omega1)/2))
        E02 = 2*np.arctan(np.sqrt((1-e1)/(1+e1))*np.tan((np.pi/2 - omega1 + np.pi)/2))
        M01 = E01 - e1*np.sin(E01)
        M02 = E02 - e1*np.sin(E02)
        pri_to_sec_gap = P1/(2*np.pi) * (M02 - M01)
        sec_to_pri_gap = P1 - pri_to_sec_gap
        return pri_to_sec_gap, sec_to_pri_gap
   
    def set_up_sim(self, els):
        P1, T01, i1, e1, omega1, P2, Tp2, ecw2, esw2, i2, Omega2, mA, mB, mp, k1, *gamma = els
        #i1 = np.pi/2
        e2 = np.sqrt(ecw2**2 + esw2**2)
        omega2 = np.arctan2(esw2, ecw2)
        if e1 < 0 or e1 > 0.8 or e2 > 0.5:
            return None
        if mA < 0 or mB < 0:
            return None
        E0 = 2*np.arctan(np.sqrt((1-e1)/(1+e1))*np.tan((np.pi/2 - omega1)/2))
        M0 = E0 - e1*np.sin(E0)
        M = M0 - 2*np.pi*T01/P1
        sim = rebound.Simulation()
        sim.integrator = 'ias15'
        sim.units = ('d', 'AU', 'Msun')
        if self.dt:
            sim.dt = self.dt
        sim.add(m=mA, hash='A')
        sim.add(m=mB, P=P1, e=e1, omega=omega1, inc=i1, Omega=0.0, M=M, hash='B')
        sim.add(m=mp, P=P2, e=e2, omega=omega2, inc=i2, Omega=Omega2, T=Tp2, hash='b')
        sim.move_to_com()
        p = sim.particles
        rebx = reboundx.Extras(sim)
        #gr = rebx.add('gr_full')
        #gr.params['C'] = 173.1
        rebx.add("tides_precession")
        for i in ['A', 'B']:
            p[i].params["R_tides"] = self.R[i]
            p[i].params["k1"] = k1
        return sim

    def impact_regression(self, ecl_model):
        b0 = {}; dbdt = {}
        for i in self.ecl_stars:
            dbdt[i], b0[i], _, _, _ = linregress(ecl_model[i].dropna()['model_t'], ecl_model[i].dropna()['model_b'])
        return dbdt, b0

    def get_chisq(self, ecl_model, rv_model, ecl=True, rv=True, b=True):
        ecl_chisq = {}
        rv_chisq = {}
        b_chisq = {}
        dbdt, b0 = self.impact_regression(ecl_model)
        for i in self.ecl_stars:
            ecl_chisq[i] = ((ecl_model[i]['res']/ecl_model[i]['data_err'])**2).sum()
            if self.b:
                b_chisq[i] = ((self.b0_data[i] - b0[i])/self.b0_err[i])**2 + ((self.dbdt_data[i] - dbdt[i])/self.dbdt_err[i])**2
        for i in self.rv_stars:
            rv_chisq[i] = ((rv_model[i]['res']/self.rv_data[i]['rv_err'])**2).sum()
        chisq_sum = 0.
        if rv:
            chisq_sum += sum(rv_chisq.values())
        if ecl:
            chisq_sum += sum(ecl_chisq.values())
        if b and self.b:
            chisq_sum += sum(b_chisq.values())
        return chisq_sum

    def evaluate(self, els, **kwargs):
        ecl_model, rv_model = self.get_residuals(els)
        if ecl_model is None or rv_model is None:
            return -np.inf
        return -0.5*self.get_chisq(ecl_model, rv_model, **kwargs)

def lsq_fit(ecl_time):
    df = ecl_time.dropna()
    A = np.vstack([np.ones(len(df)), df.index]).T
    return np.linalg.lstsq(A, df, rcond=None)[0]

def ecl_time_to_etv(ecl_time, P=None, T0=None):
    if P is None or T0 is None:
        T0, P = lsq_fit(ecl_time)
    return ecl_time - P*ecl_time.index.values - T0