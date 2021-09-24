# modules
import os

import matplotlib.pyplot as plt
import numpy as np
import pyacvd
import pyvista as pv
import spharapy.spharabasis as sb
import spharapy.trimesh as tm

from ply_writer import write_ply_file, stl_obj_to_ply


class MeshHarmonics:
    """
    Mesh Harmonics Class

    :param file_name: String representing the filename
    :param ...: ...
    """

    def __init__(self, path_to_mesh, n_vertices='default'):
        self.file_path = path_to_mesh
        self.file_name = self.file_path.split('/')[-1].split('.')[0]
        self.file_ext = '.' + self.file_path.split('/')[-1].split('.')[1]
        self.pvmesh = pv.read(self.file_path)

        if self.file_ext == '.ply':
            pass

        elif self.file_ext == '.stl' or self.file_ext == '.obj':
            print('{} file'.format(self.file_ext))
            ply_path = self.file_path.replace(self.file_ext, '.ply')
            stl_obj_to_ply(self.pvmesh, ply_path)
            self.file_path = ply_path
            self.file_ext = self.file_path.split('.')[-1]
            self.pvmesh = pv.read(self.file_path)

        else:
            'Not tested files with {} extension'.format(self.file_ext)

        self.result_path = '../results/file_' + self.file_name + '/'

        if type(n_vertices) == int and self.pvmesh.n_points != n_vertices:
            new_mesh = pyacvd.Clustering(self.pvmesh)
            new_mesh.subdivide(3)
            new_mesh.cluster(n_vertices)
            self.pvmesh = new_mesh.create_mesh()
            print('Mesh resampled for analysis (n_vertices = {}).'.format(n_vertices))

        self.vertlist = np.array(self.pvmesh.points)
        self.trilist = np.array(self.pvmesh.faces)

        self.trilist = self.trilist.reshape((self.pvmesh.n_faces, 4))[::, 1:4]
        self.tm_mesh = tm.TriMesh(self.trilist, self.vertlist)

        self.LaplacianMatrix = np.array(self.tm_mesh.laplacianmatrix('half_cotangent'))
        self.basis_functions, self.natural_freqs = sb.SpharaBasis(self.tm_mesh, 'unit').basis()

        subdirs = ['meshes', 'figures']
        for subdir in subdirs:
            try:
                os.makedirs(self.result_path + subdir)
            except FileExistsError:
                pass

    def LBO_reconstruction(self, basis_functions, EV_upper, EV_lower=0):
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

        reconstr_filepath = self.result_path + 'meshes/' + self.file_name + '_' + str(EV_upper) + '.ply'

        write_ply_file(self.reconstr_verts, self.trilist, reconstr_filepath)
        print('Mesh reconstructed (using {} eigen vectors): {}'.format(EV_upper, reconstr_filepath))

    def plot_harmonics(self, EV_list: type=list):

        if len(EV_list) <= 3:
            n_cols = len(EV_list)
            n_rows = 1
        else:
            n_cols = int(np.ceil(np.sqrt(len(EV_list))))
            n_rows = int(np.round(np.sqrt(len(EV_list))))

        figsb1, axes1 = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(n_cols * 2, n_cols * 2),
                                     subplot_kw={'projection': '3d'})

        plt.suptitle('Surface Harmonics\n {}'.format(self.file_name + self.file_ext))

        for i in range(len(EV_list)):

            colors = np.mean(self.basis_functions[self.trilist, EV_list[i]], axis=1)
            if len(EV_list)>1:
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
        if len(EV_list) > 1 and (n_rows*n_cols %2==0 or n_rows==n_cols):
            ax_diff = (n_cols*n_rows)-len(EV_list)
            for i in range(ax_diff+1):
                axes1[-1, -i].axis('off')

        plt.tight_layout()
        for f_ext in ['.png', '.svg']:
            plt.savefig(self.result_path + 'figures/' + self.file_name + '_' + str(len(EV_list)) + '_harmonics' + f_ext,
                        facecolor='w', edgecolor='w', orientation='portrait', transparent=False, pad_inches=0.1)


if __name__ == '__main__':
    M_ply = MeshHarmonics('../data/ply_test.ply', n_vertices=1000)
    # reconstruct meshes
    M_ply.LBO_reconstruction(basis_functions=M_ply.basis_functions, EV_upper=100)
    # plot harmonics
    M_ply.plot_harmonics(EV_list=[2,5,7,10,30,50,80])

    M_obj = MeshHarmonics('../data/obj_test.obj', n_vertices=1000)
    # reconstruct meshes
    M_obj.LBO_reconstruction(basis_functions=M_obj.basis_functions, EV_upper=100)
    # plot harmonics
    M_obj.plot_harmonics(EV_list=[2,5,7,10,30,50,80])

    M_stl = MeshHarmonics('../data/stl_test.stl', n_vertices=1000)
    # reconstruct meshes
    M_stl.LBO_reconstruction(basis_functions=M_stl.basis_functions, EV_upper=100)
    # plot harmonics
    M_stl.plot_harmonics(EV_list=[2,5,7,10,30,50,80])
