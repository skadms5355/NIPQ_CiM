import os
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker 
import seaborn as sns
import numpy as np
# from .models.psum_modules import *
# from .models.quantized_modules import *
# from .noise_cell import Noise_Cell
from models.noise_cell import Noise_cell

# pytest
def test_noise_cell(wbits, cbits, mapping_mode, noise_param, state, size):
    noise_cell = Noise_Cell(wbits, cbits, mapping_mode, noise_type='dynamic')
    noise_cell.update_setting(noise_param)
    G = state*torch.ones(size=size, device='cuda', dtype=torch.float32)
    G_out = noise_cell(G)
    # data_list = [G_out.mean().cpu().detach().numpy(), G_out.std().cpu().detach().numpy()]
    # import pdb; pdb.set_trace()
    # plt.plot(G_out, data_list[0], data_list[1])
    return G_out

def trans_cell(weight, wbits, cbits, mapping_mode, noise_param, ratio):
    noise_cell = Noise_Cell(wbits, cbits, mapping_mode, noise_param)
    noise_cell.update_setting(noise_param, ratio)
    
    G_out = noise_cell(weight)
    delta_G = noise_cell.get_deltaG()
    # data_list = [G_out.mean().cpu().detach().numpy(), G_out.std().cpu().detach().numpy()]
    # import pdb; pdb.set_trace()
    # plt.plot(G_out, data_list[0], data_list[1])
    return G_out, delta_G

def mac_graph(output, output_G, mapping_mode, noise_param, idx):
    f = open(f'mac_noise_errorsize.txt', 'a')
    f.write('layer_idx  noise_param  mapping_mode    Max_Error   Error(w/ shift remove) \n')
    fig, ax = plt.subplots(figsize=(15, 10))
    df_total = pd.DataFrame()
    import pdb; pdb.set_trace()
    df_total['MAC'] = output.cpu().numpy().ravel()
    df_total['ADC_out'] = output_G.cpu().numpy().ravel()
    df_total['Error'] = df_total['ADC_out'] - df_total['MAC']
    sns.set(font_scale=1.3)
    sns.lineplot(data=df_total, x='MAC', y='Error', ax=ax, label=mapping_mode)
    # sns.histplot(data=df_total, bins=150, kde=True, linewidth=0, alpha=0.7, ax=ax[i], color=colors)
    # ax.set_xticks(range(0, df_total.max().max(), df_total.max().max()+1))
    # plt.setp(ax[i].get_legend().get_texts(), fontsize='18')
    # plt.xticks(fontsize=18)
    # plt.yticks(fontsize=18)
    ax.grid(True, axis='y', alpha=0.5, linestyle='--')
    print(mapping_mode, df_total['Error'].max(), df_total['Error'].min())
    f.write('{} {} {} {}   {} \n'.format(idx, noise_param, mapping_mode, max(df_total['Error'].max(), abs(df_total['Error'].min())), abs(df_total['Error'].max() - df_total['Error'].min())))
    ax.set_yticks(ax.get_yticks())
    ax.set_xticks(ax.get_xticks())
    ax.set_ylabel(ax.get_ylabel(), fontsize=16)
    ax.set_xlabel(ax.get_xlabel(), fontsize=16)
    ax.set_yticklabels(ax.get_yticks(), fontsize=14)
    ax.set_xticklabels(ax.get_xticks(), fontsize=14)
    ax.legend(fontsize=18)
    sns.move_legend(ax, "upper center",  title=None, frameon=False)
    ax.set_title(f'noise_param: {noise_param}', loc='right', fontsize=14)
    print(f"{noise_param} graph is drawn")
    plt.savefig(f'layer_{idx} {mapping_mode}_mac_noise_{noise_param}.png')
    f.close()
    exit()

if __name__ == '__main__':
    mapping_mode_list =['2T2R', 'two_com', 'ref_a',]
    # mapping_mode_list =['2T2R', 'ref_a', 'two_com']
    wbits=4
    pbits=8
    noise_list = [0.01, 0.02, 0.03, 0.04, 0.05]
    measure='cell_hist'
    # max noise graph 
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(25, 15))
    ax = axes.flatten()
    graph_path = os.path.join("graph", "eval", "noise")
    if not os.path.exists(graph_path):
        os.makedirs(graph_path)
    if measure == 'mac':
        size = [64, 10000]
        input = torch.ones(size=size, device='cuda', dtype=torch.float32)
        weight = torch.randint(-8, 8, size=size, device='cuda', dtype=torch.float32)
        f = open(f'{graph_path}/mac_noise_errorsize.txt', 'w')
        f.write('noise_param  mapping_mode    Max_Error   Error(w/ shift remove) \n')
        for i, noise_param in enumerate(noise_list):
            for mapping_mode in mapping_mode_list:
                Linear = PsumQLinear(1, 1, wbits, wbit_serial=True, mapping_mode=mapping_mode, cbits=4)
                weight_split, num = Linear._weight_bitserial(weight=weight, w_scale=1, cbits=4)
                weight_split_com = torch.chunk(weight_split, num, dim=1)
                weight_G, delta_G = trans_cell(weight_split, wbits, cbits=4, mapping_mode=mapping_mode, noise_param=noise_param)
                weight_split_G = torch.chunk(weight_G, num, dim=1)
                mac = weight.sum(axis=0) 
                mac_fp =fp(weight.sum(axis=0), pbits=pbits, maxVal=mac.max(), minVal=mac.min())
                if mapping_mode == 'ref_d':
                    maxVal = (weight_split.sum(axis=0)).round().max()
                    minVal = (weight_split.sum(axis=0)).round().min()
                    split_mac = fp(weight_split_com[0].sum(axis=0), pbits=pbits, maxVal=maxVal, minVal=minVal) - fp(weight_split_com[1].sum(axis=0), pbits=pbits, maxVal=maxVal, minVal=minVal)
                    maxVal = (weight_G.sum(axis=0)/delta_G).round().max()
                    minVal = (weight_G.sum(axis=0)/delta_G).round().min()
                    split_G_mac = fp(weight_split_G[0].sum(axis=0)/delta_G, pbits=pbits, maxVal=maxVal, minVal=minVal) - fp(weight_split_G[1].sum(axis=0)/delta_G, pbits=pbits, maxVal=maxVal, minVal=minVal)
                elif mapping_mode == 'ref_a':
                    maxVal = ((weight_split_com[0] - weight_split_com[1]).sum(axis=0)).round().max()
                    minVal = ((weight_split_com[0] - weight_split_com[1]).sum(axis=0)).round().min()
                    split_mac = fp((weight_split_com[0] - weight_split_com[1]).sum(axis=0), pbits=pbits, maxVal=maxVal, minVal=minVal)
                    maxVal = ((weight_split_G[0] - weight_split_G[1]).sum(axis=0)/delta_G).round().max()
                    minVal = ((weight_split_G[0] - weight_split_G[1]).sum(axis=0)/delta_G).round().min()
                    split_G_mac = fp((weight_split_G[0] - weight_split_G[1]).sum(axis=0)/delta_G, pbits=pbits, maxVal=maxVal, minVal=minVal)
                elif mapping_mode == 'two_com':
                    maxVal = (weight_split.sum(axis=0)).round().max()
                    minVal = (weight_split.sum(axis=0)).round().min()
                    # print(maxVal, minVal)
                    split_mac = fp(weight_split_com[0].sum(axis=0), pbits=pbits, maxVal=maxVal, minVal=minVal) - 2**(wbits-1)/(2**(wbits-1)-1) * fp(weight_split_com[1].sum(axis=0), pbits=pbits, maxVal=maxVal, minVal=minVal)
                    maxVal = (weight_G.sum(axis=0)/delta_G).round().max()
                    minVal = (weight_G.sum(axis=0)/delta_G).round().min()
                    split_G_mac = fp(weight_split_G[0].sum(axis=0)/delta_G, pbits=pbits, maxVal=maxVal, minVal=minVal) - 2**(wbits-1)/(2**(wbits-1)-1) * fp(weight_split_G[1].sum(axis=0)/delta_G, pbits=pbits, maxVal=maxVal, minVal=minVal)
                    # print(maxVal, minVal)
                else:
                    maxVal = ((weight_split_com[0] - weight_split_com[1]).sum(axis=0)).round().max()
                    minVal = ((weight_split_com[0] - weight_split_com[1]).sum(axis=0)).round().min()
                    split_mac = fp((weight_split_com[0] - weight_split_com[1]).sum(axis=0), pbits=pbits, maxVal=maxVal, minVal=minVal)
                    maxVal = ((weight_split_G[0] - weight_split_G[1]).sum(axis=0)/delta_G).round().max()
                    minVal = ((weight_split_G[0] - weight_split_G[1]).sum(axis=0)/delta_G).round().min()
                    split_G_mac = fp((weight_split_G[0] - weight_split_G[1]).sum(axis=0)/delta_G, pbits=pbits, maxVal=maxVal, minVal=minVal)
                
                df_total = pd.DataFrame()
                df_total['MAC'] = mac_fp.cpu().numpy()
                df_total['ADC_out'] = split_G_mac.cpu().numpy()
                df_total['Error'] = df_total['ADC_out'] - df_total['MAC']
                # mac_list = sorted(list(set(mac.cpu().numpy().ravel())))
                # for val in mac_list:
                #     df=pd.DataFrame(split_G_mac[mac==val].cpu(), columns=[val])
                #     df_total = pd.concat([df_total, df], axis=1)
                # print(df_total)
                # colors=sns.set_palette("husl")
                sns.set(font_scale=1.3)
                sns.lineplot(data=df_total, x='MAC', y='Error', ax=ax[i], label=mapping_mode)
                # sns.histplot(data=df_total, bins=150, kde=True, linewidth=0, alpha=0.7, ax=ax[i], color=colors)
                # ax.set_xticks(range(0, df_total.max().max(), df_total.max().max()+1))
                # plt.setp(ax[i].get_legend().get_texts(), fontsize='18')
                # plt.xticks(fontsize=18)
                # plt.yticks(fontsize=18)
                ax[i].grid(True, axis='y', alpha=0.5, linestyle='--')
                print({mapping_mode}, df_total['Error'].max(), df_total['Error'].min())
                f.write('{} {} {}   {} \n'.format(noise_param, mapping_mode, max(df_total['Error'].max(), abs(df_total['Error'].min())), abs(df_total['Error'].max() - df_total['Error'].min())))
            ax[i].set_yticks(ax[i].get_yticks())
            ax[i].set_xticks(ax[i].get_xticks())
            ax[i].set_ylabel(ax[i].get_ylabel(), fontsize=16)
            ax[i].set_xlabel(ax[i].get_xlabel(), fontsize=16)
            ax[i].set_yticklabels(ax[i].get_yticks(), fontsize=14)
            ax[i].set_xticklabels(ax[i].get_xticks(), fontsize=14)
            ax[i].legend(fontsize=18)
            sns.move_legend(ax[i], "upper center",  ncol=len(mapping_mode_list), title=None, frameon=False)
            ax[i].set_title(f'noise_param: {noise_param}', loc='right', fontsize=14)
            print(f"{noise_param} graph is drawn")
            plt.savefig(f'{graph_path}/mac_noise.png')
        f.close()
                
    elif measure=='cell_hist':
        # cell state noise graph 
        size = [10000]
        title = ["(a) 2T2R", "(b) TC", "(c) RC"]
        fig, ax = plt.subplots(nrows=len(mapping_mode_list), ncols=1, figsize=(18, 3.5*len(mapping_mode_list)))
        sns.set(font_scale=1.3)
        for i, mapping_mode in enumerate(mapping_mode_list):
            if 'ref' in mapping_mode:
                state=2**(wbits)
                level_state=int(state/2)
            elif '2T2R' in mapping_mode:
                state=2**(wbits-1)+1
                level_state = state
            else:
                state=2**(wbits-1)
                level_state = state
            # for i, noise_param in enumerate(noise_list):
            df_total = pd.DataFrame()
            noise_param = 0.03
            for level in range(state):
                df = test_noise_cell(wbits=4, cbits=4, mapping_mode=mapping_mode, noise_param=noise_param, state=level, size=size)
                df_total[f'state {level}'] = pd.DataFrame(df.cpu())
                # df_total.loc[len(df_total)] = data_list
            # plt.hist(df_total, alpha=0.8, bins=10, range=[0, df_total.max().max()])
            df_total = df_total*1e6
            # print(df_total)
            colors=sns.set_palette("husl", state)
            sns.histplot(data=df_total, bins=150, kde=True, linewidth=0, alpha=0.7, ax=ax[i], color=colors, stat="density")
            # ax.set_xticks(range(0, df_total.max().max(), df_total.max().max()+1))
            ax[i].set_xlabel('Conductance [uS]', fontsize=22, fontdict=dict(weight='bold'))
            ax[i].set_ylabel(ax[i].get_ylabel(), fontsize=20, fontdict=dict(weight='bold'))
            # ax[i].set_ylim(0, 5000)
            ax[i].set_xlim(0, 400)
            xlabels = [int(x) for x in ax[i].get_xticks()]
            ylabels = [np.round(x, 2) for x in ax[i].get_yticks()]
            ax[i].set_xticks(xlabels)
            ax[i].set_yticks(ylabels)
            ax[i].set_xticklabels(ax[i].get_xticks(), fontsize=18, fontdict=dict(weight='bold'))
            ax[i].set_yticklabels(ax[i].get_yticks(), fontsize=18, fontdict=dict(weight='bold'))
            for axis in ['top', 'bottom', 'left', 'right']:
                ax[i].spines[axis].set_linewidth(2.5) 
            ax[i].tick_params(width=2.5)
            ax[i].set_title(title[i], y=1.1, fontsize=26, fontdict=dict(weight='bold'))
            sns.move_legend(ax[i], "upper center", ncol=level_state, title=None, frameon=False)
            # ax[i].set_title(f'{mapping_mode}, noise_param: {noise_param}', loc='right', fontsize=13)
            fig.tight_layout(pad=1.0, h_pad=2)

            print(f"{mapping_mode} graph is drawn")
        # plt.rc('legend', fontsize=18, fontdict=dict(weight='bold'))
        plt.savefig(f'{graph_path}/noise_{noise_param}_noise.pdf')

    elif measure=='cdf':
        # cell state noise graph 
        size = [10000]
        for mapping_mode in mapping_mode_list:
            fig, axes = plt.subplots(nrows=1, ncols=len(noise_list), figsize=(8*len(noise_list), 5), constrained_layout=True)
            # sns.set(font_scale=1.2)
            ax = axes.flatten()
            if 'ref' in mapping_mode:
                state=2**(wbits)
                level_state=int(state/2)
            elif '2T2R' in mapping_mode:
                state=2**(wbits-1)+1
                level_state = state
            else:
                state=2**(wbits-1)
                level_state = state
            for i, noise_param in enumerate(noise_list):
                df_total = pd.DataFrame()
                for level in range(state):
                    df = test_noise_cell(wbits=4, cbits=4, mapping_mode=mapping_mode, noise_param=noise_param, state=level, size=size)
                    df_total[f'level {level}'] = pd.DataFrame(df.cpu())
                    # df_total.loc[len(df_total)] = data_list
                # plt.hist(df_total, alpha=0.8, bins=10, range=[0, df_total.max().max()])
                df_total = df_total*1e6
                # print(df_total)
                colors=sns.set_palette("husl", state)
                sns.ecdfplot(data=df_total, ax=ax[i], palette=colors, stat="proportion", lw=3)
                # sns.histplot(data=df_total, bins=150, kde=True, linewidth=0, alpha=0.7, ax=ax[i], color=colors, stat="density")
                # ax.set_xticks(range(0, df_total.max().max(), df_total.max().max()+1))
                ax[i].set_xlabel('Conductance [uS]', fontsize=15)
                ax[i].set_ylabel(ax[i].get_ylabel(), fontsize=15)
                # ax[i].set_ylim(0, 5000)
                ax[i].set_xlim(0, 450)
                xlabels = [int(x) for x in ax[i].get_xticks()]
                ylabels = [np.round(x, 1) for x in ax[i].get_yticks()]
                ax[i].set_xticks(xlabels)
                ax[i].set_yticks(ylabels)
                ax[i].set_xticklabels(xlabels, fontsize=13)
                ax[i].set_yticklabels(ylabels, fontsize=13)
                # ax[i].set_title(f'{mapping_mode}, noise_param: {noise_param}', loc='right', fontsize=13)
                sns.move_legend(ax[i], "lower right", title=None, frameon=False)
            # fig.subplots_adjust(wspace=0.3)

            plt.savefig(f'{graph_path}/{mapping_mode}_noise_cdf.png')
            print(f"{mapping_mode} graph is drawn")