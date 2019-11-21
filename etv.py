import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# pylint: disable=invalid-name,locally-disabled
#plt.style.use('seaborn')
plt.rc('font', size=16)

G = 2.9591220363E-4
c = 173.145
msomj = 1047.377014

def getM(p, tp, t):
    return np.fmod(2 * np.pi/p * (t - tp), 2*np.pi)

def getE(M, e):
    E = M
    for j in range(25):
        E = M + e*np.sin(E)
        return E

def getf(p, tp, e, t):
    Mf = getM(p, tp, t)
    Ef = getE(Mf, e)
    return 2.*np.arctan(np.sqrt((1.+e)/(1.-e))*np.tan(Ef/2.))

class AnalyticETV:
    def __init__(self, params):
        self.p_1 = params[0]
        self.t0_1 = params[1]
        self.i_1 = params[2]
        self.e_1 = params[3]
        self.lo_1 = params[4]
        self.p_2 = params[5]
        self.tp_2 = params[6]
        self.e_2 = np.sqrt(params[7]**2 + params[8]**2)
        self.i_2 = params[9]
        self.bo_2 = params[10]
        self.lo_2 = np.arctan2(params[8], params[7])
        self.m_a = params[11]
        self.m_b = params[12]
        self.m_c = params[13] 
        self.derived_parameters() 

    def derived_parameters(self):
        self.m_ab = self.m_a + self.m_b
        self.q_1 = self.m_a/self.m_b
        self.m_abc = self.m_ab + self.m_c
        self.a_1 = (G * self.m_ab * (self.p_1/(2*np.pi))**2)**(1./3.)
        self.a_2 = (G * self.m_abc * (self.p_2/(2*np.pi))**2)**(1./3.)
        self.a_ab = self.m_c * self.a_2 / self.m_abc

        self.A_L1 = 15 * self.m_c * self.p_1 * (1 - self.e_2**2)**(-3./2.)/(8 * self.m_abc * self.p_2)
        self.A_L2 = (1-self.q_1)/(1+self.q_1) * (1 - self.m_c/self.m_abc)**(1./3.) * (self.p_1/self.p_2)**(2./3.)*self.A_L1/(1-self.e_2**2)
        self.A_S = self.p_1/self.p_2 * self.A_L1/(1-self.e_2**2)**(3./2.)

        self.f_1 = 1 + 25*self.e_1**2/8 + 15*self.e_1**4/8 + 95*self.e_1**6/64

        self.K_1_prim = -self.e_1 * np.sin(self.lo_1) + 3*self.e_1**2/4*np.cos(2*self.lo_1)
        self.K_1_sec  =  self.e_1 * np.sin(self.lo_1) + 3*self.e_1**2/4*np.cos(2*self.lo_1)

        self.K_11_prim = 3*self.e_1**2/4 + 3*self.e_1**4/16 + 3*self.e_1**6/32 + (self.e_1 + self.e_1**3/2 + self.e_1**5/4)*np.sin(self.lo_1) +\
        (51*self.e_1**2/40 + 37*self.e_1**4/80 + 241*self.e_1**6/640)*np.cos(2*self.lo_1) - 3*self.e_1**3/16*np.sin(3*self.lo_1) -\
        (self.e_1**4/16 - self.e_1**6/16)*np.cos(4*self.lo_1) - self.e_1**5/16*np.sin(5*self.lo_1) + 3*self.e_1**6/64*np.cos(6*self.lo_1)
        self.K_11_sec  = 3*self.e_1**2/4 + 3*self.e_1**4/16 + 3*self.e_1**6/32 - (self.e_1 + self.e_1**3/2 + self.e_1**5/4)*np.sin(self.lo_1) +\
        (51*self.e_1**2/40 + 37*self.e_1**4/80 + 241*self.e_1**6/640)*np.cos(2*self.lo_1) + 3*self.e_1**3/16*np.sin(3*self.lo_1) -\
        (self.e_1**4/16 - self.e_1**6/16)*np.cos(4*self.lo_1) + self.e_1**5/16*np.sin(5*self.lo_1) + 3*self.e_1**6/64*np.cos(6*self.lo_1)

        self.K_12_prim = -(self.e_1 - self.e_1**3/2 - self.e_1**5/4)*np.cos(self.lo_1) + (51*self.e_1**2/40 + 87*self.e_1**4/80 + 541*self.e_1**6/640)*np.sin(2*self.lo_1) -\
        3*self.e_1**3/16*np.cos(3*self.lo_1) - (self.e_1**4/16 + 5*self.e_1**6/32)*np.sin(4*self.lo_1) + self.e_1**5/16*np.cos(5*self.lo_1) + 3*self.e_1**6/64*np.sin(6*self.lo_1)
        self.K_12_sec  =  (self.e_1 - self.e_1**3/2 - self.e_1**5/4)*np.cos(self.lo_1) + (51*self.e_1**2/40 + 87*self.e_1**4/80 + 541*self.e_1**6/640)*np.sin(2*self.lo_1) +\
        3*self.e_1**3/16*np.cos(3*self.lo_1) - (self.e_1**4/16 + 5*self.e_1**6/32)*np.sin(4*self.lo_1) - self.e_1**5/16*np.cos(5*self.lo_1) + 3*self.e_1**6/64*np.sin(6*self.lo_1)

        self.I = np.cos(self.i_1) * np.cos(self.i_2) + np.sin(self.i_1) * np.sin(self.i_2) * np.cos(self.bo_2)
        self.i_m = np.arccos(self.I)
        self.n_1 = np.arccos((np.cos(self.i_2) - np.cos(self.i_1) * np.cos(self.i_m))/(np.sin(self.i_1) * np.sin(self.i_m)))
        self.n_2 = np.arccos((-np.cos(self.i_1) + np.cos(self.i_2) * np.cos(self.i_m))/(np.sin(self.i_2) * np.sin(self.i_m)))
        self.alpha = self.n_2 - self.n_1
        self.beta  = self.n_2 + self.n_1
        self.g_1 = self.lo_1 - self.n_1
        self.g_2 = self.lo_2 - self.n_2 + np.pi


    def del_ltte(self, t):
        f_2 = getf(self.p_2, self.tp_2, self.e_2, t)
        return -self.a_ab * np.sin(self.i_2)/c * (1 - self.e_2**2) * np.sin(f_2 + self.lo_2)/(1 + self.e_2 * np.cos(f_2))
    
    def S(self, u):
        return np.sin(u) + self.e_2 * (np.sin(u/2. + self.lo_2) + np.sin(3*u/2. - self.lo_2)/3.)
    def C(self, u):
        return np.cos(u) + self.e_2 * (np.cos(u/2. + self.lo_2) + np.cos(3*u/2. - self.lo_2)/3.)
    
    def del_1_prim(self, t):
        M_2 = getM(self.p_2, self.tp_2, t)
        f_2 = getf(self.p_2, self.tp_2, self.e_2, t)
        M_2 = np.fmod(np.fmod(M_2, 2*np.pi) + 2*np.pi, 2*np.pi)
        f_2 = np.fmod(np.fmod(f_2, 2*np.pi) + 2*np.pi, 2*np.pi)
        M = f_2 - M_2 + self.e_2 * np.sin(f_2)
        u_2 = f_2 + self.lo_2
        return self.p_1*self.A_L1*(1-self.e_1**2)**(1./2.)/(2*np.pi) * (
            (8*self.f_1/15 + 4*self.K_1_prim/5) * M +
            (1+self.I)*(self.K_11_prim*self.S(2*u_2-2*self.alpha) - self.K_12_prim*self.C(2*u_2 - 2*self.alpha)) +
            (1-self.I)*(self.K_11_prim*self.S(2*u_2-2*self.beta ) + self.K_12_prim*self.C(2*u_2 - 2*self.beta )) +
            np.sin(self.i_m)**2 * (self.K_11_prim*np.cos(2*self.n_1) + self.K_12_prim*np.sin(2*self.n_1) - 2*self.f_1/5 - 3*self.K_1_prim/5) *
            (2*M - self.S(2*u_2 - 2*self.n_2)))   
    def del_1_sec(self, t):
        M_2 = getM(self.p_2, self.tp_2, t)
        f_2 = getf(self.p_2, self.tp_2, self.e_2, t)
        M_2 = np.fmod(np.fmod(M_2, 2*np.pi) + 2*np.pi, 2*np.pi)
        f_2 = np.fmod(np.fmod(f_2, 2*np.pi) + 2*np.pi, 2*np.pi)
        M = f_2 - M_2 + self.e_2 * np.sin(f_2)
        u_2 = f_2 + self.lo_2
        return self.p_1*self.A_L1*(1-self.e_1**2)**(1./2.)/(2*np.pi) * (
            (8*self.f_1/15 + 4*self.K_1_sec/5) * M +
            (1+self.I)*(self.K_11_sec*self.S(2*u_2-2*self.alpha) - self.K_12_sec*self.C(2*u_2 - 2*self.alpha)) +
            (1-self.I)*(self.K_11_sec*self.S(2*u_2-2*self.beta ) + self.K_12_sec*self.C(2*u_2 - 2*self.beta )) +
            np.sin(self.i_m)**2 * (self.K_11_sec*np.cos(2*self.n_1) + self.K_12_sec*np.sin(2*self.n_1) - 2*self.f_1/5 - 3*self.K_1_sec/5) *
            (2*M - self.S(2*u_2 - 2*self.n_2)))   
    def del_S_prim(self, t):
        f_2 = getf(self.p_2, self.tp_2, self.e_2, t)
        u_2 = f_2 + self.lo_2
        # coplanar eccentric approximation
        return self.p_1/np.pi*self.A_S*(1-self.e_1**2)**(1./2.)*(1+self.e_2*np.cos(f_2))**3 * (
            -11*np.sin(2*u_2)/30 + self.e_1*(np.cos(self.lo_1) + 4*np.cos(2*u_2 - self.lo_1)/5 + 8*np.cos(2*u_2 + self.lo_1)/15))
        # doubly circular inclined approximation
        #return 11*self.m_c*self.p_1**3/(32*np.pi*self.m_abc*self.p_2**2) * ( -(1 + self.I)*np.sin(2*u_2 - 2*self.alpha) + 
        #    (1 - self.I)*np.sin(2*u_2 - 2*self.beta) - np.sin(self.i_m)**2 * np.sin(2*self.n_1) * (1 + np.cos(2*u_2 - 2*self.n_2)))
    def del_S_sec(self, t):
        f_2 = getf(self.p_2, self.tp_2, self.e_2, t)
        u_2 = f_2 + self.lo_2
        # coplanar eccentric approximation
        return self.p_1/np.pi*self.A_S*(1-self.e_1**2)**(1./2.)*(1+self.e_2*np.cos(f_2))**3 * (
            -11*np.sin(2*u_2)/30 - self.e_1*(np.cos(self.lo_1) + 4*np.cos(2*u_2 - self.lo_1)/5 + 8*np.cos(2*u_2 + self.lo_1)/15))
        # doubly circular inclined approximation
        #return 11*self.m_c*self.p_1**3/(32*np.pi*self.m_abc*self.p_2**2) * ( -(1 + self.I)*np.sin(2*u_2 - 2*self.alpha) + 
        #    (1 - self.I)*np.sin(2*u_2 - 2*self.beta) - np.sin(self.i_m)**2 * np.sin(2*self.n_1) * (1 + np.cos(2*u_2 - 2*self.n_2)))
    def del_apse(self, lo_here):
        return self.p_1/(2*np.pi)*(2*np.arctan(self.e_1*np.cos(lo_here)/(1+np.sqrt(1-self.e_1**2)-self.e_1*np.sin(lo_here))) + np.sqrt(1-self.e_1**2)*self.e_1*np.cos(lo_here)/(1-self.e_1*np.sin(lo_here)))
    def del_apse_prim(self, t):
        lo_here = self.lo_1 + np.radians(1.7e-5)*(t-800)
        return self.del_apse(lo_here) - self.del_apse(self.lo_1)
    def del_tot_prim(self, t):
        return self.del_ltte(t) + self.del_1_prim(t) + self.del_S_prim(t)
    def del_tot_sec(self, t):
        return self.del_ltte(t) + self.del_1_sec(t) + self.del_S_sec(t)

class SystemClass(AnalyticETV):
    def __init__(self, system, params):
        AnalyticETV.__init__(self, params)
        self.system = system
        if system == 5095:
            self.data1 = np.genfromtxt('../data/KID5095269/koi509.tt.dan.db.try7.trans')
            self.data2 = np.empty(shape=(0, 0))
            self.longname = '5095269'
        elif system == 782:
            self.data1 = np.genfromtxt('../data/KID7821010/data782.b.tt.trans')
            self.data2 = np.genfromtxt('../data/KID7821010/data782.c.tt.trans')
            self.longname = '7821010'
        elif system == 3938:
            self.data1 = np.genfromtxt('../data/data3938.b.tt.trans')
            self.data2 = np.genfromtxt('../data/data3938.c.tt.trans')
            self.longname = '3938073'
        elif system == 8610:
            self.data1 = np.genfromtxt('../data/data8610.b.tt.trans')
            self.data2 = np.genfromtxt('../data/data8610.c.tt.trans')
            self.longname = '8610483'
        self.data1, self.data2 = self.data1.T, self.data2.T
        self.start_time = 0.
        self.end_time = 1600.
        if self.system == 782:
            self.end_time = 2000.
        self.times = np.linspace(self.start_time, self.end_time, 1000)

    def get_o_c_prim(self, x):
        return np.array([self.data1[1], self.data1[1] - x[1] - x[0]*self.data1[0], self.data1[2]])

    def get_o_c_sec(self, x):
        return np.array([self.data2[1], self.data2[1] - x[1] - x[0]*self.data2[0], self.data2[2]])

    def get_resids_prim(self, x):
        o_c = self.get_o_c_prim(x)
        return o_c[1] - self.del_tot_prim(o_c[0])
    def get_resids_sec(self, x):
        o_c = self.get_o_c_sec(x)
        return o_c[1] - self.del_tot_sec(o_c[0])

    def chisq_resid_prim(self, x):
        o_c = self.get_o_c_prim(x)
        return np.sum(((self.del_tot_prim(o_c[0]) - o_c[1])/o_c[2])**2)
    def chisq_resid_sec(self, x):
        o_c = self.get_o_c_sec(x)
        return np.sum(((self.del_tot_sec(o_c[0]) - o_c[1])/o_c[2])**2)

    def optimize(self):
        bounds = ((15.0, 50.0), (800., 840.))
        guess = [self.p_1, self.t0_1]
        self.opt_prim = minimize(self.chisq_resid_prim, guess, bounds=bounds)
        print('Primary fit:')
        print(self.opt_prim.x)
        print(self.opt_prim.fun)

        if self.data2.size:
            self.opt_sec = minimize(self.chisq_resid_sec, guess, bounds=bounds)
            print('Secondary fit:')
            print(self.opt_sec.x)
            print(self.opt_sec.fun)
            #avg_bin_per = (self.opt_prim.x[0] + self.opt_sec.x[0])/2
            #self.opt_prim.x[0] = avg_bin_per
            #self.opt_sec.x[0] = avg_bin_per

    def configure_plot(self, plot_secondary=True):
        fig = plt.figure(figsize=(10,5))
        if self.data2.size and plot_secondary:
            ax1 = plt.subplot2grid((2, 1), (0, 0))
            ax2 = plt.subplot2grid((2, 1), (1, 0), sharex=ax1)
            return fig, ax1, ax2
        else:
            ax1 = plt.subplot2grid((1, 1), (0, 0))
            return fig, ax1, None

    def model_plot(self, ax1, ax2, plot_tot=True, plot_ltte=False, plot_dyn=False, plot_apse=False):
        # Smooth model
        ltte_effect = self.del_ltte(self.times)
        dyn_effect_prim = self.del_1_prim(self.times)
        s_effect_prim = self.del_S_prim(self.times)
        tot_effect_prim = ltte_effect + dyn_effect_prim + s_effect_prim
        dyn_effect_sec = self.del_1_sec(self.times)
        s_effect_sec = self.del_S_sec(self.times)
        apse_effect_prim = self.del_apse_prim(self.times)
        tot_effect_sec = ltte_effect + dyn_effect_sec + s_effect_sec
        if plot_ltte:
            ax1.plot(self.times, 86400*ltte_effect, c='k')
            if ax2:
                ax2.plot(self.times, 86400*ltte_effect)
        if plot_dyn:
            ax1.plot(self.times, 86400*(dyn_effect_prim + s_effect_prim), c='k')
            if ax2:
                ax2.plot(self.times, 86400*(dyn_effect_sec + s_effect_sec))
        if plot_tot:
            ax1.plot(self.times, 86400*tot_effect_prim, c='k')
            if ax2:
                ax2.plot(self.times, 86400*tot_effect_sec)
        if plot_apse:
            ax1.plot(self.times, 86400*apse_effect_prim, c='k')

    def data_plot(self, ax1, ax2):
        self.optimize()
        # O-C data
        best_o_c_prim = self.get_o_c_prim(self.opt_prim.x)
        ax1.errorbar(best_o_c_prim[0], 86400*best_o_c_prim[1], 86400*best_o_c_prim[2], fmt='o', marker='x')
        if ax2 and self.data2.size:
            best_o_c_sec = self.get_o_c_sec(self.opt_sec.x)
            ax2.errorbar(best_o_c_sec[0], 86400*best_o_c_sec[1], 86400*best_o_c_sec[2], fmt='o', marker='x')

    def finalize_plot(self, fig, ax1, ax2):
        title = 'Eclipse Timing Variations of KIC ' + self.longname
        #ax1.set_title(title)
        #plt.xlabel('BJD-2454900 (days)')
        ax1.set_ylabel('Primary O-C (seconds)')
        if ax2:
            ax2.set_ylabel('Secondary O-C (seconds)')
        plt.ylim([-80.0, 120])
        plt.xlim([self.start_time, self.end_time])
        plt.subplots_adjust(hspace=0)
        plt.setp(ax1.get_xticklabels(), visible=False)
        # The y-ticks will overlap with "hspace=0", so we'll hide the bottom tick
        ax1.set_yticks(ax1.get_yticks()[1:])
        plt.show()

system_782 = SystemClass(782, [24.238217, 821.002012, np.radians(90.0), 0.678925, np.radians(239.23940),
                               990.1583246, 1184.69244, 0.362028*np.cos(123.056915), 0.362028*np.sin(123.056915), 80.8040823, -4.1679800,
                               1.282782, 1284.9546700/msomj, 2.04081/msomj])

system_5095 = SystemClass(5095, [18.61085, 66.86201, np.radians(85.02), 0.5092, 0.0, np.radians(180+108.56),
                                 236.8796, 95.923, 0.0376, -0.04030, 83.01, 0.77, -36.83,
                                 1.11, 1.14, 5.0/msomj])
