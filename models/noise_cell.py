import torch
import torch.nn as nn

class Noise_Cell(nn.Module):
    def __init__(self, wbits, cbits, mapping_mode, noise_type='static', Gmin=1/3e5, Gmax=1/3e3):
        super(Noise_Cell, self).__init__()
        """
            This module performs cell variation (included IR drop effect (wire resistance = 1 ohm))
        """
        self.mapping_mode = mapping_mode
        self.wbits = wbits
        self.cbits = cbits
        self.clevel = None
        self.noise_type = noise_type
        self.Gmin = Gmin
        self.Gmax = Gmax
        ## nosie parameter
        # static: 0~0.3(30%) (3sigma point) based on reference delta_G
        # dynamic: 0~0.3 of each G mean levels
        self.effective_clevel()
        self.delta_G = (self.Gmax-self.Gmin) / (self.clevel-1)
        self.G = torch.zeros(self.clevel, dtype=torch.float)
        if self.noise_type == 'static':
            self.std_G = (self.Gmax-self.Gmin) / (2**(self.wbits)-1)
            self.G_std = None
        else:
            self.G_std = torch.zeros(self.clevel, dtype=torch.float)
    
    def effective_clevel(self):
        if self.mapping_mode == 'two_com':
            if self.cbits >= (self.wbits-1):
                self.clevel = 2**(self.wbits - 1) # cell can represent self.wbits-1 bits
            else:
                assert False, 'This file does not support when cbits are lower than wbits-1'
        elif (self.mapping_mode == '2T2R') or (self.mapping_mode == 'PN'):
            if self.cbits >= (self.wbits-1):
                self.clevel = 2**(self.wbits - 1) + 1 # cell can represent self.wbits-1 + 1 levels (9 levels)
            else:
                assert False, 'This file does not support when cbits are lower than wbits-1'
        elif (self.mapping_mode == 'ref_d') or (self.mapping_mode == 'ref_a'):
            if self.cbits >= self.wbits:
                self.clevel = 2**self.wbits # cell can represent self.wbits-1 bits
            else:
                assert False, 'This file does not support when cbits are lower than wbits-1'

    def get_deltaG(self):
        return self.delta_G

    def update_setting(self, noise_param, ratio=100):
        self.noise_param = noise_param
        if ratio != 100:
            self.Gmax = ratio * self.Gmin
            self.delta_G = (self.Gmax-self.Gmin) / (self.clevel-1) 
            if self.noise_type == 'static':
                self.std_G = (self.Gmax-self.Gmin) / (2**(self.wbits)-1)
        
        self.generate_G()
    
    def generate_G(self):
        # if self.noise_type == 'static':
        #     self.G_std =(self.noise_param+1)/6*self.std_G # sigma = (noise_param + 1) / 6 * std_G (reference delta G)
        
        for i in range(self.clevel):
            self.G[i] = self.Gmin + i*self.delta_G
            # if self.noise_type == 'dynamic':
            #     self.G_std[i] = self.noise_param * self.G[i]
            # else:
            #     assert False, "Check your noise_type argument"            
        
    def noise_generate(self, state, size, device="cuda"):

        if self.noise_type == 'static':
            noise_output = torch.normal(self.G[state], self.G_std, size=size).to(device)
        elif self.noise_type == 'dynamic':
            noise_output = torch.normal(self.G[state], self.G_std[state], size=size).to(device)

        noise_output[noise_output < 0] = self.G[state].to(device)
        assert torch.all(noise_output > 0), "Do not set negative cell value"
        return noise_output

    def forward(self, input):
        
        # each state noise 
        # output = input
        # for state in range(self.clevel):
        #     cell_noise = self.noise_generate(state, input.size(), device=input.device).to(input.device)
        #     output = torch.where(input==state, cell_noise, output)

        input_idx = input.detach().cpu().numpy()
        output = torch.normal(self.G[input_idx], self.noise_param*self.G[input_idx]).to(input.device)
        assert torch.all(output > 0), "Do not set negative cell value"
        # import pdb; pdb.set_trace()
        return output
