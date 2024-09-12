import matplotlib.pyplot as plt
import numpy as np
import torch

def stats_plot(stats, filename):
    # Convert tensors to numpy arrays if they are tensors
    for key in stats:
        if isinstance(stats[key], torch.Tensor):
            stats[key] = stats[key].detach().cpu().numpy()

    fig = plt.figure(figsize=(14, 12))

    plt.subplot(4, 4, 1)
    plt.plot(stats['kl_div_s'] + stats['mse_o'], c='k', label='F')
    plt.plot(stats['F'], 'k--', label='F (weighted)')
    plt.yscale("log")
    plt.ylabel('F')
    plt.legend()
    plt.grid(True)

    plt.subplot(4, 4, 2)
    plt.plot(stats['F_top'], 'k--', label='F top')
    plt.yscale("log")
    plt.ylabel('F top')
    plt.grid(True)

    plt.subplot(4, 4, 3)
    plt.plot(stats['F_mid'], 'k--', label='F mid')
    plt.yscale("log")
    plt.ylabel('F mid')
    plt.legend()
    plt.grid(True)

    plt.subplot(4, 4, 4)
    plt.plot(stats['F_down'], 'k--', label='F down')
    plt.yscale("log")
    plt.ylabel('F down')
    plt.legend()
    plt.grid(True)

    plt.subplot(4, 4, 5)
    plt.plot(stats['kl_div_s'], 'r', label='kl_s')
    plt.yscale("log")
    plt.ylabel('KL(s)')
    plt.grid(True)

    plt.subplot(4, 4, 6)
    plt.ylabel('KL s dimensions')
    plt.xlabel('epochs')
    for ii in range(len(stats['kl_div_s_anal'][0])):
        if ii < 10:
            plt.plot(stats['kl_div_s_anal'][:, ii], label=str(ii))
        else:
            plt.plot(stats['kl_div_s_anal'][:, ii])
    plt.legend()

    plt.subplot(4, 4, 7)
    plt.ylabel('KL s (naive) dimensions')
    plt.xlabel('epochs')
    for ii in range(len(stats['kl_div_s_naive_anal'][0])):
        if ii < 10:
            plt.plot(stats['kl_div_s_naive_anal'][:, ii], label=str(ii))
        else:
            plt.plot(stats['kl_div_s_naive_anal'][:, ii])
    plt.legend()

    plt.subplot(4, 4, 8)
    plt.ylabel('Variables')
    for varname in ['a', 'b', 'c', 'beta_s', 'gamma']:
        plt.plot(stats['var_' + varname], label=varname)
    plt.xlabel('epochs')
    plt.yscale("log")
    plt.legend()

    plt.subplot(4, 4, 9)
    plt.plot(stats['kl_div_pi'], c='y', label='kl_pi')
    plt.yscale("log")
    plt.ylabel('KL(pi)')
    plt.grid(True)

    plt.subplot(4, 4, 10)
    plt.ylabel('KL pi dimensions')
    plt.xlabel('epochs')
    for ii in range(len(stats['kl_div_pi_anal'][0])):
        if ii < 10:
            plt.plot(stats['kl_div_pi_anal'][:, ii], label=str(ii))
        else:
            plt.plot(stats['kl_div_pi_anal'][:, ii])
    plt.legend()

    plt.subplot(4, 4, 11)
    plt.plot(stats['mse_o'], 'k', label='H(o,P(o))')
    plt.plot([0, len(stats['mse_o'])], [80.0, 80.0], 'r--', label='acceptable')
    plt.plot([0, len(stats['mse_o'])], [60.0, 60.0], 'g', label='perfect')
    plt.yscale("log")
    plt.ylabel('nats')
    plt.legend()
    plt.grid(True)

    plt.subplot(4, 4, 12)
    plt.plot(stats['mse_r'])
    plt.ylabel('MSE_r')
    plt.xlabel('iterations(x1000)')
    plt.yscale("log")

    plt.subplot(4, 4, 13)
    plt.ylabel('Total correlation')
    plt.xlabel('epochs')
    plt.plot(stats['TC'], c='k')
    plt.yscale("log")

    plt.subplot(4, 4, 14)
    plt.ylabel('Deep reconstructions')
    plt.xlabel('epochs')
    plt.plot(stats['deep_mse_o'], c='r', label='mse visual')
    plt.yscale("log")
    plt.legend()

    plt.subplot(4, 4, 15)
    plt.plot(stats['omega'], c='b', label='omega')
    plt.plot(stats['omega'] + stats['omega_std'], 'b--')
    plt.plot(stats['omega'] - stats['omega_std'], 'b--')
    plt.yscale("log")
    plt.ylabel('omega')
    plt.grid(True)

    fig.set_tight_layout(True)
    plt.savefig(filename + '.png')
    plt.savefig(filename + '.svg')
    plt.close()
