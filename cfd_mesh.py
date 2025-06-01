import pandas as pd
import numpy as np
from cfd_toolbox.plot import *


if __name__ == '__main__':
    data = [pd.read_excel(r".\plug_result_mesh.xlsx", sheet_name='50'),
            pd.read_excel(r".\plug_result_mesh.xlsx", sheet_name='100'),
            pd.read_excel(r".\plug_result_mesh.xlsx", sheet_name='150'),
            pd.read_excel(r".\plug_result_mesh.xlsx", sheet_name='200'),
            pd.read_excel(r".\plug_result_mesh.xlsx", sheet_name='300'),
            pd.read_excel(r".\plug_result_mesh.xlsx", sheet_name='400'),
            pd.read_excel(r".\plug_result_mesh.xlsx", sheet_name='600'),
            pd.read_excel(r".\plug_result_mesh.xlsx", sheet_name='800'),
            pd.read_excel(r".\plug_result_mesh.xlsx", sheet_name='1000'),
            pd.read_excel(r".\plug_result_mesh.xlsx", sheet_name='1200'),
            pd.read_excel(r".\plug_result_mesh.xlsx", sheet_name='1400'),
            pd.read_excel(r".\plug_result_mesh.xlsx", sheet_name='1800')]
    mesh_n = np.array([0.522, 1.95, 4.36, 7.47, 17.0, 29.4, 67.3, 117, 181, 262, 357, 470])
    column = ['index', 'report-def-massflow', 'report-def-thrust', 'Cf', 'SpecImpulse']

    valid_row = np.all([(np.abs(sheet['report-def-continuity']) < 1e-3).to_list() for sheet in data], axis=0)
    data_selected = np.array([sheet[valid_row][column] for sheet in data])
    fig, ax = plt.subplots()
    ax.plot(mesh_n, data_selected[:, 0, 3], 'k-', linewidth=1.5)
    ax.scatter(mesh_n, data_selected[:, 0, 3], s=50, linewidths=1.2,
               edgecolors='k', facecolors='none', marker='o')
    ax.set_xlabel('$n_{mesh}$ / w')
    ax.set_ylabel('$C_f$')
    fig.tight_layout()
    fig.show()

    valid_row = np.all([(np.abs(sheet['report-def-continuity']) < 1).to_list() for sheet in data], axis=0)
    data_selected = np.array([sheet[valid_row][column] for sheet in data])
    data_selected /= data_selected[-1]
    mesh_n = mesh_n[2: -2]
    data_selected = data_selected[2: -2]
    fig, axes = plt.subplots(2, 2, sharex='all')
    axes = axes.flatten()
    for i in range(data_selected.shape[1]):
        axes[0].plot(mesh_n, data_selected[:, i, 1], 'r-', linewidth=0.8)
        axes[0].scatter(mesh_n, data_selected[:, i, 1], s=80, linewidths=0.8,
                        edgecolors='r', facecolors='none', marker='^')
        axes[0].set_ylabel('$error$ $of$ $Q_m$')
        axes[1].plot(mesh_n, data_selected[:, i, 2], 'b-', linewidth=0.8)
        axes[1].scatter(mesh_n, data_selected[:, i, 2], s=80, linewidths=0.8,
                        edgecolors='b', facecolors='none', marker='o')
        axes[1].set_ylabel('$error$ $of$ $F$')
        axes[2].plot(mesh_n, data_selected[:, i, 3], 'c-', linewidth=0.8)
        axes[2].scatter(mesh_n, data_selected[:, i, 3], s=80, linewidths=0.8,
                        edgecolors='c', facecolors='none', marker='D')
        axes[2].set_ylabel('$error$ $of$ $C_f$')
        axes[2].set_xlabel('$n_{mesh}$ / w')
        axes[3].plot(mesh_n, data_selected[:, i, 4], 'g-', linewidth=0.8)
        axes[3].scatter(mesh_n, data_selected[:, i, 4], s=80, linewidths=0.8,
                        edgecolors='g', facecolors='none', marker='s')
        axes[3].set_ylabel('$error$ $of$ $I_s$')
        axes[3].set_xlabel('$n_{mesh}$ / w')
        for ax in axes:
            ax.set_yticks([])
    fig.tight_layout()
    fig.show()


