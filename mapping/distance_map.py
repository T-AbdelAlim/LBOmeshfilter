#%%
import pyvista as pv
import numpy as np

def calc_dmap(mesh_vertices, ref_vertices):

    distance11 = mesh_vertices - ref_vertices
    distance11 = np.sqrt(distance11[:,0]*distance11[:,0] + distance11[:,1]*distance11[:,1] +distance11[:,2]*distance11[:,2])
    RSS = np.sum(distance11)
    return float(RSS)


def show_dmap(visualized_mesh, reference_mesh, show=True):
    shown_mesh = pv.read(visualized_mesh)
    reference_mesh = pv.read(reference_mesh)

    distance11 = shown_mesh.points - reference_mesh.points
    distance11 = np.sqrt(distance11[:,0]*distance11[:,0] + distance11[:,1]*distance11[:,1] +distance11[:,2]*distance11[:,2])

    pv.set_plot_theme("ParaView")
    min_corr = np.min(distance11)
    max_corr = np.max(distance11)
    shown_mesh['colors'] = distance11

    plotter = pv.Plotter(off_screen=False)
    plotter.add_mesh(shown_mesh, opacity=1, scalars='colors', clim=[min_corr, max_corr] ,show_edges=False, rgb=False, cmap='GnBu')
    plotter.add_mesh(reference_mesh.points, color='red', render_points_as_spheres=True)

    plotter.show()
    plotter.clear()


if __name__ == '__main__':
    clean_mesh = "C:/Users/Tareq/pythonProject/LBOmeshfilter/data/clean.ply"
    raw_mesh = "C:/Users/Tareq/pythonProject/LBOmeshfilter/data/raw.ply"

    c = pv.read(clean_mesh)
    r = pv.read(raw_mesh)
    calc_dmap(mesh_vertices=c.points, ref_vertices=r.points)

    show_dmap(visualized_mesh=raw_mesh, reference_mesh=clean_mesh)