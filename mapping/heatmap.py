#%%

import pyvista as pv
import cv2
import numpy as np


def plot_heatmap(visualized_mesh, reference_mesh):
    pv.set_plot_theme("ParaView")

    shown_mesh = pv.read(visualized_mesh)
    reference_mesh = pv.read(reference_mesh)

    distance11 = shown_mesh.points - reference_mesh.points
    distance11 = np.sqrt(distance11[:,0]*distance11[:,0] + distance11[:,1]*distance11[:,1] +distance11[:,2]*distance11[:,2])
    print(np.sum(distance11))

    min_corr = np.min(distance11)
    max_corr = np.max(distance11)
    shown_mesh['colors'] = distance11

    plotter = pv.Plotter(off_screen=False)
    plotter.add_mesh(shown_mesh, opacity=1, scalars='colors', clim=[min_corr, max_corr] ,show_edges=False, rgb=False, cmap='GnBu')
    plotter.add_mesh(reference_mesh.points, color='red', render_points_as_spheres=True)
    #plotter.add_text('{} eigenvectors'.format(file.split('.')[0].split('_')[-1]))

    plotter.show()
    # cv2.imwrite('results.png', plotter.image)
    plotter.clear()


if __name__ == '__main__':
    clean_mesh = "C:/Users/Tareq/pythonProject/LBOmeshfilter/data/clean.ply"
    raw_mesh = "C:/Users/Tareq/pythonProject/LBOmeshfilter/data/raw.ply"

    plot_heatmap(visualized_mesh=clean_mesh, reference_mesh=raw_mesh)