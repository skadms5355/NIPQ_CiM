import torch
import torch.nn as nn
import math

class Noise_cell(nn.Module):
    def __init__(self, wbits, cbits, mapping_mode, co_noise=0.01, noise_type='static', res_val='rel', w_format='state', Gmin=1/3e5, ratio=100, max_epoch=-1):
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

        self.init_state()

    def init_state(self):
        self.effective_clevel()
        if self.noise_type == 'meas':
            self.G = torch.tensor([43.9e-6, 96.16e-6])
            self.G_std = torch.tensor([5.02e-6, 9.52e-6]) #MRAM data

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
            if self.cbits >= (self.wbits-1):
                self.clevel = 2**(self.wbits - 1) + 1 # cell can represent self.wbits-1 + 1 levels (4bits: 9 levels)
            else:
                assert False, 'This file does not support that cbits are lower than wbits-1'
        elif (self.mapping_mode == 'ref_d') or (self.mapping_mode == 'ref_a'):
            if self.cbits >= self.wbits:
                self.clevel = 2**self.wbits # cell can represent self.wbits-1 bits
            else:
                assert False, 'This file does not support that cbits are lower than wbits'
    
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
                elif noise_type == 'prop' or 'meas':
                    output = torch.normal(self.G[x_idx], self.G_std[x_idx]).to(x.device)
                assert torch.all(output > 0), "Do not set negative cell value"
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
                else:
                    output = x + (self.G_std[0]**2 * torch.randn_like(x, device=x.device))
                
        return output
