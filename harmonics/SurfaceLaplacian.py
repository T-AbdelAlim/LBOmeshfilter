import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pyacvd
import pyvista as pv
import spharapy.spharabasis as sb
import spharapy.trimesh as tm

from pathlib import Path

from harmonics.ply_writer import write_ply_file, stl_obj_to_ply
from mapping.distance_map import calc_dmap, show_dmap

warnings.filterwarnings("ignore")
cwd = Path(os.getcwd())


class MeshHarmonics:
    """
    Mesh Harmonics Class

    Main methods
        LBO_reconstruction - reconstruct mesh based on n-eigen vectors
        plot_harmonics - visualize harmonic frequencies on a mesh
    """

    def __init__(self, path_to_mesh, n_vertices='default', examples=False):
        self.file_path = Path(path_to_mesh)

        self.file_name = self.file_path.name
        self.file_ext = self.file_path.suffix
        self.pvmesh = pv.read(self.file_path)

        if self.file_ext == '.ply':
            pass

        elif self.file_ext == '.stl' or self.file_ext == '.obj':
            ply_path = self.file_path.replace(self.file_ext, '.ply')
            stl_obj_to_ply(self.pvmesh, ply_path)
            self.file_path = ply_path
            self.file_ext = self.file_path.split('.')[-1]
            self.pvmesh = pv.read(self.file_path)

        else:
            'Not tested files with {} extension'.format(self.file_ext)

        if examples == True:
            self.result_path = cwd.joinpath('examples/results/file_' + self.file_name + '/')
        else:
            self.result_path = cwd.joinpath('results/file_' + self.file_name + '/')

        if type(n_vertices) == int and self.pvmesh.n_points != n_vertices:
            new_mesh = pyacvd.Clustering(self.pvmesh)
            new_mesh.subdivide(3)
            new_mesh.cluster(n_vertices)
            self.pvmesh = new_mesh.create_mesh()
            print('Mesh resampled for analysis (n_vertices = {})'.format(n_vertices))

        self.vertlist = np.array(self.pvmesh.points)
        self.trilist = np.array(self.pvmesh.faces)

        self.trilist = self.trilist.reshape((self.pvmesh.n_faces, 4))[::, 1:4]
        self.tm_mesh = tm.TriMesh(self.trilist, self.vertlist)

        self.LaplacianMatrix = np.array(self.tm_mesh.laplacianmatrix('half_cotangent'))
        self.basis_functions, self.natural_freqs = sb.SpharaBasis(self.tm_mesh, 'unit').basis()

    def LBO_reconstruction(self, basis_functions, EV_upper, EV_lower=0, write_ply=True):
        """
        Function that reconstructs a mesh from a desired number of eigenvectors

        :param basis_functions: Input the eigenvector matrix (MeshHarmonics.basisfunctions)
        :param EV_upper: n eigenvectors upper limit to reconstruct the mesh
        :param EV_lower: n eigenvectors lwer limit to reconstruct the mesh (default: 0)
        :param write_ply: Boolean true if user wants to write and save a ply file of the reconstructed mesh
        :return:
        reconstructed vertices (x,y,z) from the number of eigenvectors,
        if write_ply = True (default), a reconstructed mesh is saved as ply_filename_nEV.ply
        """

        # each colomn in EV matrix phi is an eigenvector
        phi = basis_functions  # phi = eigenvector matrix
        phi_rec = np.zeros([len(phi), len(phi)])
        phi_rec[:, EV_lower:EV_upper] = phi[:, EV_lower:EV_upper]
        phi_t = phi_rec.T

        # eigenvector multiplied with vertices x,y,z
        px_hat = np.matmul(phi_t, self.vertlist[::, 0])
        py_hat = np.matmul(phi_t, self.vertlist[::, 1])
        pz_hat = np.matmul(phi_t, self.vertlist[::, 2])

        # reconstructed vertices
        px_rec = np.matmul(phi, px_hat)
        py_rec = np.matmul(phi, py_hat)
        pz_rec = np.matmul(phi, pz_hat)
        p_rec = np.stack([px_rec, py_rec, pz_rec])

        self.reconstr_verts = p_rec.T

        if write_ply == True:
            try:
                os.makedirs(self.result_path.joinpath('meshes_' + str(len(self.vertlist)) + '_vertices'))
            except FileExistsError:
                pass

            reconstr_filepath = self.result_path.joinpath('meshes_' + str(
                len(self.vertlist)) + '_vertices/' + self.file_name + '_' + str(EV_upper) + '.ply')
            write_ply_file(self.reconstr_verts, self.trilist, reconstr_filepath)
            print('Mesh reconstructed (using eigenvectors {}-{}): {}'.format(EV_lower, EV_upper, reconstr_filepath))

        else:
            pass

    def plot_harmonics(self, EV_list: type = list):
        """
        Function to plot specific harmonic frequencies on the original mesh

        :param basis_functions: Input the eigenvector matrix (MeshHarmonics.basisfunctions)
        :param EV_list: list of eigenvectors to be plotted (e.g. [10,200,500])
        :return: Figure saved of the plotted harmonic frequencies corresponding to the eigenvectors in EV_list
        """

        try:
            os.makedirs(self.result_path.joinpath('figures_' + str(len(self.vertlist)) + '_vertices'))
        except FileExistsError:
            pass

        if len(EV_list) <= 3:
            n_cols = len(EV_list)
            n_rows = 1
        else:
            n_cols = int(np.ceil(np.sqrt(len(EV_list))))
            n_rows = int(np.round(np.sqrt(len(EV_list))))

        figsb1, axes1 = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(n_cols * 2, n_cols * 2),
                                     subplot_kw={'projection': '3d'})

        plt.suptitle('Surface Harmonics\n {}'.format(self.file_name))

        for i in range(len(EV_list)):

            colors = np.mean(self.basis_functions[self.trilist, EV_list[i]], axis=1)
            if len(EV_list) > 1:
                ax = axes1.flat[i]
            else:
                ax = axes1

            ax.view_init(elev=70., azim=15.)
            ax.set_aspect('auto')
            ax.set_axis_off()
            ax.set_title('e = ' + str(EV_list[i]))

            trisurfplot = ax.plot_trisurf(self.vertlist[:, 0], self.vertlist[:, 1],
                                          self.vertlist[:, 2], triangles=self.trilist,
                                          cmap=plt.cm.bwr,
                                          edgecolor='white', linewidth=0.)
            trisurfplot.set_array(colors)
            plt.tight_layout()

        # remove empty plots:
        if len(EV_list) > 1 and (n_rows * n_cols % 2 == 0 or n_rows == n_cols):
            ax_diff = (n_cols * n_rows) - len(EV_list)
            for i in range(ax_diff + 1):
                axes1[-1, -i].axis('off')

        plt.tight_layout()
        for f_ext in ['.png', '.svg']:
            plt.savefig(self.result_path.joinpath('figures_' + str(len(self.vertlist)) + '_vertices/' + self.file_name + '_' +
                        str(len(EV_list)) + '_harmonics' + f_ext),
                        facecolor='w', edgecolor='w', orientation='portrait', transparent=False, pad_inches=0.1)

    def calc_distance_error(self, mesh_vertices, reference_vertices):
        """
        Function that calculates the difference between two sets of vertices

        :param mesh_vertices: Vertices of source mesh
        :param reference_vertices: Vertices of reference mesh
        :return:
        """
        self.dist_error = calc_dmap(mesh_vertices, reference_vertices)

    def calc_normal_error(self, mesh_normals, reference_normals):
        """
        Function that calculates the difference between two sets of normal vectors

        :param mesh_vertices: Vertices of source mesh
        :param reference_vertices: Vertices of reference mesh
        :return:
        """
        self.norm_error = calc_dmap(mesh_normals, reference_normals)

    def calc_volume_error(self, pv_mesh, pv_reference_mesh):
        """
        Function that calculates the difference between two mesh volumes

        :param mesh_vertices: Vertices of source mesh
        :param reference_vertices: Vertices of reference mesh
        :return:
        """
        self.volume_error = (pv_mesh.volume - pv_reference_mesh.volume)


if __name__ == '__main__':
    print('LBO')
