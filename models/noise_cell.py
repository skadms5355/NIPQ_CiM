import torch
import torch.nn as nn
from scipy.interpolate import interp1d
from scipy.stats import rv_continuous
import math
import os
import numpy as np
import pandas as pd

# Define a discrete random variable using the PDF function (Hynix ReRAM Noise)
class InterpolatedPDF(rv_continuous):
    def __init__(self, pdf_func, kind='linear', samples=1000, a=None, b=None, **kwargs):
        super().__init__(a=a, b=b, **kwargs)
        self.pdf_func = pdf_func
        self.samples = samples
        self.x = np.linspace(self.a, self.b, self.samples)
        self.y = self.pdf_func(self.x)
        self.cdf = np.cumsum(self.y) / np.sum(self.y)
        self.interp = interp1d(self.cdf, self.x, kind=kind, bounds_error=False, fill_value=(self.a, self.b))

    def _pdf(self, x):
        return self.pdf_func(x)
    
    def _cdf(self, x):
        return np.trapz([self._pdf(t) for t in np.linspace(self.a, x, 1000)], np.linspace(self.a, x, 1000))

    def _ppf(self, q):
        return self.interp(q)

class KIST_MRAM():
    def __init__(self, clevel):
        super(KIST_MRAM, self).__init__()
        self.TMR_std = 0.05
        self.Rp_100 = torch.tensor([1054, 753, 538, 384, 274, 196, 140, 100], dtype=torch.float32)
        self.Rp_std = torch.tensor([6.83, 6.45, 6.06, 5.71, 5.51, 5.41, 5.26, 5.21])/100 * self.Rp_100
        self.Rap = self.Rp_100*3
        self.Rap_std = torch.tensor([7.55, 7.26, 6.87, 6.61, 6.47, 6.40, 6.19, 6.15])/100 * self.Rap 
        # self.Rap = torch.normal(mean, self.TMR_std*mean)

        self.clevel = clevel
        self.step = 1
        # self.step = int(self.Rp_100.shape[0]/clevel)
        self.delta_R = 7.91 * self.step # ideal computed delta R

    def make_res(self, state, num, device):
        #Random AP 
        idx = self.Rp_100.shape[0] - self.step * state
        expand_Rp = self.Rp_100.unsqueeze(dim=0).expand(num, -1)
        expand_Rap = self.Rap.unsqueeze(dim=0).expand(num, -1)
        Rp_n = torch.normal(expand_Rp, self.Rp_std).to(device=device)
        Rap_n = torch.normal(expand_Rap, self.Rap_std).to(device=device)
        rRp_sum = torch.sum((1/Rp_n).transpose(1,0)[:idx], dim=0)
        rRap_sum = torch.sum((1/Rap_n).transpose(1,0)[idx:], dim=0)
        samples = 1/(rRp_sum+rRap_sum)

        return samples

    def forward(self, x):
        # Resistor is linearity (not Conductance), so using V=IR (voltage detection)
        # states has resistor values
        rseed = torch.randint(0, 32765, (1,))
        np.random.seed(rseed)
        # import matplotlib.pyplot as plt
        # import seaborn as sns
        # fig, ax = plt.subplots(nrows=2, figsize=(20, 12))
        
        for c in range(self.clevel):
            if torch.where(x==c)[0].shape[0] != 0:
                index = torch.where(x==c)
                samples = self.make_res(c, index[0].shape[0], x.device)
                x[index] = samples

            if torch.any(x<0):
                import pdb; pdb.set_trace()
            
        #     sns.histplot(samples.cpu().numpy(), ax=ax[0], bins=200, alpha=0.2, element='step', fill=True, stat='density')
        #     sns.histplot(samples.cpu().numpy(), ax=ax[1], bins=200, alpha=0.2, element='step', fill=True, stat='count')
        
        # # setting y-axis figure ax[0]
        # ax[0].set_ylabel(ax[0].get_ylabel(), fontsize=16)
        # ax[0].legend(('state 0', 'state 1', 'state 2', 'state 3', 'state 4', 'state 5', 'state 6', 'state 7'), loc='upper center', ncol=9, fontsize=12, frameon=False)
        # ax_ylabels = np.round(np.linspace(ax[0].get_yticks()[0], ax[0].get_yticks()[-2], num=5), 2)
        # ax[0].set_yticks(ax_ylabels)
        # ax[0].set_yticklabels(ax[0].get_yticks(), fontsize=14)

        # # setting y-axis figure ax[0]
        # ax[1].set_ylabel(ax[1].get_ylabel(), fontsize=16)
        # ax[1].legend(('state 0', 'state 1', 'state 2', 'state 3', 'state 4', 'state 5', 'state 6', 'state 7'), loc='upper center', ncol=9, fontsize=12, frameon=False)
        # ax_ylabels = np.linspace(ax[1].get_yticks()[0], ax[1].get_yticks()[-2], num=5, dtype=int)
        # ax[1].set_yticks(ax_ylabels)
        # ax[1].set_yticklabels(ax[1].get_yticks(), fontsize=14)
        # ax[1].set_yticks(ax_ylabels)

        # # setting x-axis figure ax
        # ax[0].set_title(f'MRAM Noise', loc='right', fontsize=16)
        # ax[0].set_xlabel('Resistance [kohm]')
        # ax[0].set_xlabel(ax[0].get_xlabel(), fontsize=16)
        # xlabels = ax[0].get_xticks()
        # ax[0].set_xlim(20, np.ceil(xlabels.max()))
        # ax[0].set_xticks(ax[0].get_xticks())
        # ax[0].set_xticklabels(ax[0].get_xticks(), fontsize=14)

        # # setting x-axis figure ax[1]
        # ax[1].set_xlabel('Resistance [kohm]')
        # # xlabels = ax[1].get_xticks()
        # ax[1].set_xlabel(ax[1].get_xlabel(), fontsize=16)
        # ax[1].set_xlim(20, np.ceil(xlabels.max()))
        # ax[1].set_xticks(ax[1].get_xticks())
        # ax[1].set_xticklabels(ax[1].get_xticks(), fontsize=14)
        # plt.savefig(os.getcwd() +"/graph/MRAM/sample.png")
        # import pdb; pdb.set_trace()

        return x / self.delta_R


class Noise_cell(nn.Module):
    def __init__(self, wbits, cbits, mapping_mode, wsym=False, co_noise=0.01, noise_type='static', res_val='rel', w_format='state', Gmin=1/3e5, ratio=100, max_epoch=-1):
        super(Noise_cell, self).__init__()
        """
            This module performs cell variation
            types of noise : static, grad (gradual), prop (proportional to weight)
            resistance value: rel (relative) => relative std to relative ratio according to weight value
                              abs (absolute) => convert weight value to resistance value
            weight format: state (0 ~ clevel)
                           weight (-2**(wbits-1) ~ 2**(wbits-1)-1)
        """
        self.mapping_mode = mapping_mode
        self.wbits = wbits
        self.cbits = cbits
        self.noise_type = noise_type
        self.Gmin = Gmin
        self.res_val = res_val
        self.co_noise = co_noise
        self.w_format = w_format
        self.ratio = ratio
        self.max_epoch = max_epoch
        
        self.wsymmetric = wsym

        self.init_state()

    def init_state(self):
        self.effective_clevel()
        if self.noise_type == 'meas':
            self.G = torch.tensor([43.9e-6, 96.16e-6])
            self.G_std = torch.tensor([5.02e-6, 9.52e-6]) #SAMSUNG MRAM data

        if self.res_val == 'rel':
            self.std_offset = torch.Tensor(1).fill_((self.clevel - 1) / (self.ratio - 1))
            self.init_std(self.max_epoch)
        elif self.res_val == 'abs': 
            self.Gmax = self.ratio * self.Gmin
            self.delta_G = (self.Gmax-self.Gmin) / (self.clevel-1)
            self.G = torch.tensor([self.Gmin + i*self.delta_G for i in range(self.clevel)])
            if self.noise_type == 'static':
                idx = int(torch.floor(torch.div(self.clevel, 2)).detach().cpu())
                self.G_std = torch.Tensor(1).fill_(self.co_noise * self.G[idx])
            elif self.noise_type == 'prop':
                self.G_std = torch.tensor([self.co_noise * self.G[i] for i in range(self.clevel)])
            elif self.noise_type == 'meas':
                self.delta_G = self.G[-1] - self.G[0]
            elif self.noise_type == 'interp':
                self.interp_init()
            elif self.noise_type == 'kist':
                self.kist = KIST_MRAM(self.clevel)
            else:
                assert False, 'In the {} mode, the {} noise type is not supported, you have to choose static or prop'.format(self.res_val, self.noise_type)
        else:
            assert False, 'You must choose the one of two options [rel, abs], but got {}'.format(self.res_val)

    def effective_clevel(self):
        if self.mapping_mode == 'two_com':
            if self.cbits >= (self.wbits-1):
                self.clevel = 2**(self.wbits - 1) # cell can represent self.wbits-1 bits
            elif self.cbits == 1:
                self.clevel = 2
            else:
                assert False, 'This file does not support that cbits are lower than wbits-1'
        elif (self.mapping_mode == '2T2R') or (self.mapping_mode == 'PN'):
            if self.wbits == 1:
                self.clevel = 2
            elif self.cbits >= (self.wbits-1):
                if self.wsymmetric:
                    self.clevel = 2**(self.wbits - 1) # cell can represent self.wbits-1 levels (4bits: 8 levels)
                else:
                    self.clevel = 2**(self.wbits - 1) + 1 # cell can represent self.wbits-1 + 1 levels (4bits: 9 levels)
            else:
                assert False, 'This file does not support that cbits are lower than wbits-1'
        elif (self.mapping_mode == 'ref_d') or (self.mapping_mode == 'ref_a'):
            if self.cbits >= self.wbits:
                self.clevel = 2**self.wbits # cell can represent self.wbits-1 bits
            else:
                assert False, 'This file does not support that cbits are lower than wbits'
    
    def interp_init(self):

        ## hynix reram data to convert continuous data
        if (self.co_noise == 1) or (self.co_noise == 3):
            df = pd.read_csv('/mnt/nfs/nameunkang/Project/NIPQ_CiM/data/ReRAM/Hynix_data_case1.csv')
        elif (self.co_noise == 2) or (self.co_noise == 4):
            df = pd.read_csv('/mnt/nfs/nameunkang/Project/NIPQ_CiM/data/ReRAM/Hynix_data_case2.csv')
        else:
            assert False, "Check co_noise parameter, You only select two option (1, 2) in {}".format(self.noise_type)
        
        self.max_state = df.iloc[:, 0::2].max().to_numpy()
        self.min_state = df.iloc[:, 0::2].min().to_numpy()
        self.Gmin = 10
        if self.co_noise <= 2: # step size is wide
            if 'ref' in self.mapping_mode:
                state = [df.iloc[:, 2*i:2*i+2].dropna(axis=0).to_numpy().transpose() for i in range(self.clevel)]
                self.delta_G = 10 # G_min = 10us, delta_G = 10uS
            elif self.mapping_mode == '2T2R':
                self.delta_G = 19 # G_min = 10us, delta_G = 10uS (not evenly uniform)
                index = [0, 4, 8, 12, 16, 20, 24, 28, 30] # location of state
                state = [df.iloc[:, i:i+2].dropna(axis=0).to_numpy().transpose() for i in index]
                self.max_state = df.iloc[:, index].max().to_numpy()
                self.min_state = df.iloc[:, index].min().to_numpy()
        else: # step size is constant to 10uS
            self.delta_G = 10
            state = [df.iloc[:, 2*i:2*i+2].dropna(axis=0).to_numpy().transpose() for i in range(self.clevel)]
            self.max_state = self.max_state[:self.clevel]
            self.min_state = self.min_state[:self.clevel]
            # import pdb; pdb.set_trace()

        self.pdf = [interp1d(state[c][0], state[c][1], kind='quadratic', bounds_error=False, fill_value=0) for c in range(self.clevel)]
        self.rv = [InterpolatedPDF(self.pdf[c], kind='quadratic', samples=1000, a=state[c][0].min(), b=state[c][0].max(), name='interpolated') for c in range(self.clevel)]

    def interp_sample(self, x):
        # import matplotlib.pyplot as plt
        # import seaborn as sns
        # fig, ax = plt.subplots(nrows=2, figsize=(20, 12))
        # ax1 = ax[0].twinx()
        rseed = torch.randint(0, 32765, (1,))
        np.random.seed(rseed)
        
        for c in range(self.clevel):
            if torch.where(x==c)[0].shape[0] != 0:
                index = torch.where(x==c)
                samples = torch.tensor(self.rv[c].rvs(size=x[index].numel()), dtype=x.dtype, device=x.device)

                while torch.any(np.floor(self.min_state[c]) > samples) or torch.any(np.ceil(self.max_state[c]) < samples):
                    if torch.any(np.floor(self.min_state[c]) > samples):
                        min_index = torch.where(self.min_state[c] > samples)
                        samples[min_index] = torch.tensor(self.rv[c].rvs(size=min_index[0].shape[0]), dtype=x.dtype, device=x.device)

                    if torch.any(np.ceil(self.max_state[c]) < samples):
                        max_index = torch.where(self.max_state[c] < samples)
                        samples[max_index] = torch.tensor(self.rv[c].rvs(size=max_index[0].shape[0]), dtype=x.dtype, device=x.device)

                x[index] = samples
            
            # # for graph
            # else:
            #     samples = torch.tensor([], dtype=x.dtype, device=x.device)

            if torch.any(x<0):
                import pdb; pdb.set_trace()
            
            # else:
            #     index = torch.where(x==c)
            #     samples = torch.tensor(self.rv[c].rvs(size=x[index].numel()), dtype=x.dtype, device=x.device)

        #     sns.histplot(samples.cpu().numpy(), ax=ax[0], bins=200, alpha=0.2, element='step', fill=True, stat='density')
        #     plt_x = np.linspace(self.min_state[c], self.max_state[c], num=1000)
        #     sns.lineplot(x=plt_x, y=self.pdf[c](plt_x), ax=ax1)
        #     sns.histplot(samples.cpu().numpy(), ax=ax[1], bins=200, alpha=0.2, element='step', fill=True, stat='count')

        # # setting y-axis figure ax[0]
        # ax[0].set_ylabel(ax[0].get_ylabel(), fontsize=16)
        # ax[0].legend(('state 0', 'state 1', 'state 2', 'state 3', 'state 4', 'state 5', 'state 6', 'state 7', 'state 8'), loc='upper center', ncol=9, fontsize=12, frameon=False)
        # ax_ylabels = np.round(np.linspace(ax[0].get_yticks()[0], ax[0].get_yticks()[-2], num=5), 2)
        # ax[0].set_yticks(ax_ylabels)
        # ax[0].set_yticklabels(ax[0].get_yticks(), fontsize=14)

        # # setting y-axis figure ax1 (ax[0] twinx)
        # ylabels = ax1.get_yticks()
        # ax1.set_ylabel('Probability [%]')
        # ax1.set_ylabel(ax1.get_ylabel(), fontsize=16)
        # ax1.set_ylim(0, ylabels[-1])
        # ax1.set_yticklabels(ax1.get_yticks(), fontsize=14)

        # # setting y-axis figure ax[0]
        # ax[1].set_ylabel(ax[1].get_ylabel(), fontsize=16)
        # ax[1].legend(('state 0', 'state 1', 'state 2', 'state 3', 'state 4', 'state 5', 'state 6', 'state 7', 'state 8'), loc='upper center', ncol=9, fontsize=12, frameon=False)
        # ax_ylabels = np.linspace(ax[1].get_yticks()[0], ax[1].get_yticks()[-2], num=5, dtype=int)
        # ax[1].set_yticks(ax_ylabels)
        # ax[1].set_yticklabels(ax[1].get_yticks(), fontsize=14)
        # ax[1].set_yticks(ax_ylabels)

        # # setting x-axis figure ax
        # ax[0].set_title(f'ReRAM Noise Sampling (Case 2, Step=10uS)', loc='right', fontsize=16)
        # ax[0].set_xlabel('Conductance [uS]')
        # ax[0].set_xlabel(ax[0].get_xlabel(), fontsize=16)
        # xlabels = ax[0].get_xticks()
        # ax[0].set_xlim(0, np.ceil(xlabels.max()))
        # ax[0].set_xticks(ax[0].get_xticks())
        # ax[0].set_xticklabels(ax[0].get_xticks(), fontsize=14)

        # # setting x-axis figure ax[1]
        # ax[1].set_xlabel('Conductance [uS]')
        # # xlabels = ax[1].get_xticks()
        # ax[1].set_xlabel(ax[1].get_xlabel(), fontsize=16)
        # ax[1].set_xlim(0, np.ceil(xlabels.max()))
        # ax[1].set_xticks(ax[1].get_xticks())
        # ax[1].set_xticklabels(ax[1].get_xticks(), fontsize=14)
        # plt.savefig(os.getcwd() +"/graph/ReRAM/pdf_sample.png")
        # import pdb; pdb.set_trace()

        return x 

    def grad_epoch_list(self, max_epoch):
        self.clvl_list = [int(math.floor(max_epoch/self.clevel.item())) for _ in range(int(self.clevel.item()))]
        for idx in range(int(max_epoch - self.clvl_list[0]*self.clevel.item())):
            self.clvl_list[idx] +=1
    
    def compute_std(self, state):
        std_offset = self.std_offset.to(state.device)
        if self.w_format == 'state':
            return self.co_noise * (state+std_offset)
        else:
            if (self.mapping_mode == '2T2R') or (self.mapping_mode == 'PN'):
                return self.co_noise * torch.sqrt(torch.pow(state+std_offset, 2) + torch.pow(std_offset, 2))
            elif self.mapping_mode == 'ref_a':
                w_ref = 2**(self.wbits-1).to(std_offset.device)
                return self.co_noise * torch.sqrt(torch.pow(state+std_offset, 2) + torch.pow(w_ref+std_offset, 2))
            elif self.mapping_mode == 'two_com':
                return self.co_noise * (state+std_offset)
            else:
                assert False, 'This {} mapping mode is not supported in relative mode'.format(self.mapping_mode)

    def init_std(self, max_epoch=-1):
        # rel std initialization
        if self.noise_type == 'static':
            std = self.compute_std(torch.floor(torch.div(self.clevel, 2)))
            self.G_std = torch.Tensor(1).fill_(std.item())
        elif self.noise_type == 'grad':
            if max_epoch != -1:
                self.grad_epoch_list(max_epoch)
                self.cell_idx = 0
                self.sum = 0     
            else:
                assert False, "Please enter max_epoch in relative gradual mode"
            G = torch.arange(0, self.clevel)
            self.std_list = self.compute_std(G)
            self.G_std = torch.Tensor(1).fill_(self.std_list[self.cell_idx])
        elif self.noise_type == 'prop':
            G = torch.arange(0, self.clevel)
            self.G_std = self.compute_std(G)
        elif self.noise_type == 'meas':
            #MRAM data
            self.delta_G = self.G[-1] - self.G[0] 
            self.std_offset = self.G[0] / self.delta_G 
            self.G_std /= self.delta_G 
        elif self.noise_type == 'interp':
            ## Approximating the interpolation graph to a gaussian distribution 
            self.interp_init()
            state_mean = [((self.rv[c].mean() - self.Gmin) / self.delta_G) for c in range(self.clevel)]
            state_std = [self.rv[c].std() / self.delta_G for c in range(self.clevel)]
            if (self.mapping_mode == '2T2R') or (self.mapping_mode == 'PN'):
                self.G_std = torch.tensor([np.sqrt(np.power(state_std[c], 2) + np.power(state_std[0], 2)) for c in range(self.clevel)])
                self.G = torch.tensor([state_mean[c] - state_mean[0] for c in range(self.clevel)])
            elif 'ref' in self.mapping_mode:   
                w_ref = int(self.clevel/2)
                self.G_std = torch.tensor([np.sqrt(np.power(state_std[c], 2) + np.power(state_std[w_ref], 2)) for c in range(self.clevel)])
                self.G = torch.tensor([state_mean[c] - state_mean[w_ref] for c in range(self.clevel)])
            else:
                assert False, "Only support 2T2R mapping mode"

            # array = [c + torch.normal(self.G[c], state_std[c], size=(10000, )).numpy() for c in range(self.clevel)]
            # import matplotlib.pyplot as plt
            # import seaborn as sns
            # import pandas as pd 

            # df = pd.DataFrame(array).transpose()
            # fig, ax = plt.subplots(figsize=(20, 6))
            # sns.histplot(array, ax=ax, alpha = 0.2, element='step', fill=True, bins=200)
            # ax.set_ylabel(ax.get_ylabel(), fontsize=16)
            # ax.legend(('state 8', 'state 7', 'state 6', 'state 5', 'state 4', 'state 3', 'state 2', 'state 1', 'state 0'), reverse=True, loc='upper center', ncol=9, fontsize=12, frameon=False)
            # ax_ylabels = np.linspace(ax.get_yticks()[0], ax.get_yticks()[-2], num=5, dtype=int)
            # ax.set_yticks(ax_ylabels)
            # ax.set_yticklabels(ax.get_yticks(), fontsize=14)
            # ax.set_title(f'ReRAM Noise Gaussian Modeling (Case 1)', loc='right', fontsize=16)
            # ax.set_xlabel('Conductance [uS]')
            # ax.set_xlim(-1, 10)
            # ax.set_xlabel(ax.get_xlabel(), fontsize=16)
            # ax.set_xticks(ax.get_xticks())
            # ax.set_xticklabels(ax.get_xticks(), fontsize=14)
            # plt.savefig("/mnt/nfs/nameunkang/Project/NIPQ_CiM/graph/ReRAM/gaussian_modeling.png")
            # import pdb; pdb.set_trace()
        else:
            assert False, 'Check noise type, you write {} noise type'.format(self.noise_type)

    def update_std(self, epoch=-1):
        if self.noise_type == 'grad':
            if (epoch - self.sum) == self.clvl_list[self.cell_idx]:
                self.sum += self.clvl_list[self.cell_idx]
                self.cell_idx += 1
                self.G_std.data[0] = self.std_list[self.cell_idx]
        else:
            assert False, 'Only gradually mode is updated'

    def get_deltaG(self, G_min=False):
        if G_min:
            return self.delta_G, self.G[0]
        return self.delta_G

    def get_offset(self):
        return self.std_offset
    
    def forward(self, x, float_comp=False, w_split=True):
        noise_type = self.noise_type
        res_val = self.res_val
        
        if float_comp and (res_val == 'rel'):
            if self.w_format == 'state':
                if noise_type == 'meas':
                    output = x + (85 * self.G_std.max()**2 * torch.randn_like(x, device=x.device))
                else:
                    assert False, "the input of noise_cell have to be weight format during training, but got {}".format(self.w_format)
            if noise_type == 'prop':
                x_cell = x + 2**(self.wbits-1) if self.mapping_mode == 'ref_a' else abs(x)
                output = x + torch.normal(0, self.compute_std(x_cell))
            else:
                output = x + (self.G_std[0]**2 * torch.randn_like(x, device=x.device))
        else:
            if res_val == 'abs':
                x_idx = x.detach().cpu().numpy()
                if noise_type == 'static':
                    output = torch.normal(self.G[x_idx], self.G_std[0]).to(x.device)
                    output = torch.where(output<0, output[output>0].min(), output).to(x.device)
                elif noise_type == 'prop' or noise_type == 'meas':
                    output = torch.normal(self.G[x_idx], self.G_std[x_idx]).to(x.device)
                    # Need G_std setting! 
                elif noise_type == 'interp':
                    output = self.interp_sample(x)
                elif noise_type == 'kist':
                    output = self.kist.forward(x) * self.delta_G
                assert torch.all(output > 0), "Do not set negative cell value"
            
                return output / self.delta_G
            elif res_val == 'rel':
                if noise_type == 'prop':
                    if self.w_format == 'state':
                        x_cell = x
                    else:
                        x_cell = x+2**(self.wbits-1) if self.mapping_mode == 'ref_a' else abs(x)
                    output = x + torch.normal(0, self.G_std[x_cell.detach().cpu().numpy()]).to(x.device)
                elif noise_type == 'meas':
                    if self.w_format == 'state':
                        x_cell = x
                        if w_split:
                            output = x + torch.normal(self.std_offset, self.G_std[x_cell.detach().cpu().numpy()]).to(x.device)
                        else:
                            output = x + (85 * self.G_std.max()**2 * torch.randn_like(x, device=x.device))
                    else:
                        x_cell = x+2**(self.wbits-1) if self.mapping_mode == 'ref_a' else abs(x)
                        output = x + torch.normal(0, self.G_std[x_cell.detach().cpu().numpy()]).to(x.device)
                elif noise_type == 'interp':
                    if self.w_format == 'state':
                        x_cell = x.to(torch.long)
                    else:
                        x_cell = x+2**(self.wbits-1) if self.mapping_mode == 'ref_a' else abs(x)
                        x_cell = x_cell.to(torch.long)

                    output = torch.normal(self.G[x_cell], self.G_std[x_cell]).to(x.device).type(x.dtype)

                    if self.mapping_mode == '2T2R':
                        output = torch.where(x<0, -1 * output, output)
                else:
                    output = x + (self.G_std[0]**2 * torch.randn_like(x, device=x.device))
        return output


