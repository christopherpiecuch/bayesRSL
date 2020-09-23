import os, os.path as op
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import h5py, pickle
from scipy.io import loadmat
import BZ
from BZ import bbTS, bbPlot
plt.style.use('seaborn-dark-palette')

class cmpPyMat(BZ.bzBase):
    def __init__(self, save_tag=2):
        super().__init__()
        self.path_res = op.join(self.path_Brsl, 'bayes_model_solutions')
        self.arrnames = 'MU NU PI_2 DELTA_2 SIGMA_2 TAU_2 PHI B L R Y_O Y TGDATA N K D'.split()
        self.dct_arrs, self.dct_hp = self.load_py_res(save_tag)
        self.dct_mat  = loadmat(op.join(self.path_res, f'experiment_{save_tag}'))
        self.cmp_y()
        self.cmp_parms()
        return

    def load_py_res(self, save_tag=0):
        dst_h5    = op.join(self.path_res, f'py_exp{save_tag}.h5')
        dct_arrs  = {}
        with h5py.File(dst_h5, 'r') as h5:
            for arrn in self.arrnames:
                try:
                    arr = h5[arrn][:]
                except:
                    arr = h5[arrn] # scalars
                dct_arrs[arrn] = arr

        dst_dct   = f'{op.splitext(dst_h5)[0]}.dct'
        with open(dst_dct, 'rb') as fh:
            dct_hp    = pickle.load(fh)
        return dct_arrs, dct_hp

    def y_stats(self, kind='py'):
        """ Units converted to mm """
        if kind == 'py':
            y  = self.dct_arrs['Y'] # simulation x year x tide gauge
        elif kind == 'mat':
            y  = self.dct_mat['Y']

        dates = pd.date_range('18930101', '20180101', freq='AS')
        decyr = bbTS.date2dec(dates)
        ## get a timeseries for each gauge
        y_res = {}
        rates_all = [] ## rates for all simulations, all gauges
        for tg in range(y.shape[-1]):
            rates = []
            for sim in range(y.shape[0]):
                ts     = y[sim, :, tg] * 100 # convert to mm
                coeffs = np.polyfit(decyr, ts, 1)
                rates.append(coeffs[0])
            rates_all.extend(rates)
            ser = pd.Series(rates)
            y_res[tg] = [ser.mean(), ser.std(), ser.sem()]
        df = pd.DataFrame(y_res, index=['avg', 'std', 'sem'])
        df = df[list(range(y.shape[-1]))] # ensure sorted by tgs
        return df, rates_all

    def cmp_y(self):
        ## units are mm
        df_py_y, rates_py   = self.y_stats('py')
        df_mat_y, rates_mat = self.y_stats('mat')

        ## unnecessary
        lbls  = ['Python3.8', 'Matlab2016a']

        ## create a property - property of the rates
        fig, axes = plt.subplots(figsize=(16,9))
        axes.scatter(df_mat_y.loc['avg'], df_py_y.loc['avg'], c='k')
        axes.plot([0,1],[0,1], 'k--', transform=axes.transAxes) # 1 - 1
        axes.xaxis.set_label_coords(0.275, -0.035)
        axes.set_xlabel(f'{lbls[1]} (mm/yr)')
        axes.set_ylabel(f'{lbls[0]} (mm/yr)')
        axes.set_title('RSL Comparison at each TG')
        fig.set_label('Prop-Prop'); # fig.patch.set_alpha(0)

        ## also compare the hists
        c     = ['darkblue', 'darkgreen'] # to ensure consistency with prev
        fig, axes = plt.subplots(figsize=(16,9), ncols=3, sharex=True, sharey=True)
        axe       = axes.ravel()
        for i, (rates, ax) in enumerate(zip([rates_py, rates_mat], axe)):
            ser_rates     = pd.Series(rates)
            avg, std, sem = ser_rates.mean(), ser_rates.std(), ser_rates.sem()
            x, y, f, t = 0.01, 0.97, 11, ax.transAxes
            ax.text(x, y, f'$\mu$ = {avg:.4f} $\pm$ {sem:.4f}', transform=t, fontsize=f, color='k')
            ax.text(x, y-0.05, f'$\sigma$ = {std:.4f}', transform=t, fontsize=f, color='k')

            ax.hist(rates, color=c[i])
            ax.set_title(lbls[i])
            ax.set_xlabel('Rate (cm/yr)')

        ## add both rates to one histogram
        axe[-1].hist(rates_mat, color=c[1])
        axe[-1].hist(rates_py, color=c[0], alpha=0.8)
        axe[-1].set_title('Both')
        axe[-1].set_xlabel('Rate (mm/yr)')
        axe[0].set_ylabel('RSL Counts')
        fig.set_label('Rate_Hists'); #fig.patch.set_alpha(0)
        # bbPlot.savefigs(self.path_res, True, True)#, **{'dpi':300, 'transparent':False})
        return

    def cmp_parms(self):
        """ Compare the solution parameters, also see Table S2 """
        med_pys, med_mats = [], []
        parms   = 'MU NU PHI PI_2 DELTA_2 SIGMA_2 TAU_2 R'.split()
        # table   = [3.03e-3, 6.95, 0.71e-3, 0.58e-6, 6.91e-6, 28.19e-6, 53.62e-6, 0.71]
        table   = [3.03, 6.95, 0.71, 0.58, 6.91, 28.19, 53.62, 0.71]
        ## blow up values for easier comparison / with Table S2
        for i, parm in enumerate(parms):
            meds  = np.array([np.median(self.dct_arrs[parm]), np.median(self.dct_mat[parm])])
            if i in [0, 2]:
                meds *= 1e3
            elif i in list(range(3,7)):
                meds  = np.sqrt(meds * 1e6)
            med_py, med_mat = meds.tolist()
            med_pys.append(med_py); med_mats.append(med_mat)
        df = pd.DataFrame([table, med_mats, med_pys], columns=parms, index=['S3', 'MAT', 'PY']).T

        fig, axes = plt.subplots(figsize=(16,9))
        axes.scatter(med_mats, med_pys, c='k')
        axes.plot([0,1],[0,1], 'k--', transform=axes.transAxes) # 1 - 1
        axes.set_xlabel('Matlab2016a')
        axes.set_ylabel('Python3.8')
        axes.set_title(f'Comparison for Parameters: {", ".join(parms)}')
        BZ.print_all (df)

        return

if __name__ == '__main__':
    cmpPyMat()
    plt.show()
