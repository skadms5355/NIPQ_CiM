import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def cell_graph(weight_chunk, wsplit_num, graph_path, layer_idx, mapping_mode, wbits, cbits):
    if not os.path.exists(graph_path):
        os.makedirs(graph_path)

    fig, ax = plt.subplots(wsplit_num, 1, figsize=(10, 1.8 * wsplit_num), constrained_layout=True)
    sns.set(font_scale=1.25)
    sns.set_style("white")
    sns.despine()
    if mapping_mode == 'two_com':
        if cbits >= (wbits-1):
            state = 2**(wbits - 1) # cell can represent wbits-1 bits
            title = f"TC method: layer {layer_idx}"
            legend = ['w_unsign', 'w_sign']
            # legend = [f'Layer {layer_idx}', f'Layer {layer_idx}']
            color = ["goldenrod"]
        else:
            assert False, 'This file does not support when cbits are lower than wbits-1'
    elif (mapping_mode == '2T2R') or (mapping_mode == 'PN'):
        if cbits >= (wbits-1):
            state = 2**(wbits - 1) + 1 # cell can represent wbits-1 + 1 levels (9 levels)
            title = f"PN method: layer {layer_idx}"
            if wsplit_num > 1: 
                legend = ['w_p', 'w_n']
            else:
                legend = ['combined cells']
            # legend = [f'Layer {layer_idx}', f'Layer {layer_idx}']
            color = ["firebrick"]
        else:
            assert False, 'This file does not support when cbits are lower than wbits-1'
    elif (mapping_mode == 'ref_d') or (mapping_mode == 'ref_a'):
        if cbits >= wbits:
            state = 2**wbits # cell can represent wbits-1 bits
            title = f"RC method: layer {layer_idx}"
            legend = ['w_c', 'w_ref']
            # legend = [f'Layer {layer_idx}', f'Layer {layer_idx}']
            color = ["olivedrab"]
        else:
            assert False, 'This file does not support when cbits are lower than wbits-1'

    for i in range(wsplit_num):
        df_data = pd.DataFrame(weight_chunk[i].cpu().numpy().ravel(), columns=[legend[i]])
        sns.histplot(data=df_data, alpha=0.8, ax=ax[i], palette=color, linewidth=0, discrete=True, shrink=0.7, stat='count')
        ax[i].set_xticks(range(0, state, 1))
        ax[i].set_yticks(ax[i].get_yticks())
        ax[i].set_ylabel(ax[i].get_ylabel(), fontsize=12)

        ax[i].set_yticklabels(ax[i].get_yticks(), fontsize=11)
        # import pdb; pdb.set_trace()
        ax[i].set_xticklabels(ax[i].get_xticks(), fontsize=13)
        ax[i].tick_params(length=0)
        sns.move_legend(ax[i], 'best', title=None, frameon=False)

    # ax[0].set_title(title, loc='right', fontsize=14)
    ax[wsplit_num-1].set_xlabel('Cell levels', fontsize=14)
    # plt.autoscale(axis='y', tight=False)
    # fig.subplots_adjust(hspace=0.3, bottom=1)
    plt.savefig(f'{graph_path}/layer{layer_idx}_cell_dist.png')
    plt.close(fig)