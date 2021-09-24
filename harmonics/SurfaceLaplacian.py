# modules
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import pyacvd
from ply_writer import write_ply_file

import spharapy.trimesh as tm
import spharapy.spharabasis as sb
import spharapy.datasets as sd

def LBO_reconstruction(basis_functions, EV_upper):
    # each colomn in EV matrix phi is an eigenvector
    phi = basis_functions # phi = eigenvector matrix
    phi_rec = np.zeros([len(phi),len(phi)])
    phi_rec[:,:EV_upper] = phi[:,:EV_upper]
    phi_t = phi_rec.T

    # eigenvector multiplied with vertices x,y,z
    px_hat = np.matmul(phi_t,vertlist[::, 0])
    py_hat = np.matmul(phi_t,vertlist[::, 1])
    pz_hat = np.matmul(phi_t,vertlist[::, 2])

    # reconstructed vertices
    px_rec = np.matmul(phi, px_hat)
    py_rec = np.matmul(phi, py_hat)
    pz_rec = np.matmul(phi, pz_hat)
    p_rec = np.stack([px_rec,py_rec,pz_rec])

    reconstruction_verts = p_rec.T
    return reconstruction_verts

def my_write_ply_file(points, faces, savepath):
    numVertices = len(points)
    numFaces = len(faces)
    faces = faces.reshape((numFaces, 3))
    with open(savepath, 'w+') as fileOut:
        # Writes ply header
        fileOut.write("ply\n")
        fileOut.write("format ascii 1.0\n")
        fileOut.write("comment VCGLIB generated\n")
        fileOut.write("element vertex " + str(numVertices) + "\n")
        fileOut.write("property float x\n")
        fileOut.write("property float y\n")
        fileOut.write("property float z\n")

        fileOut.write("element face " + str(numFaces) + "\n")
        fileOut.write("property list uchar int vertex_indices\n")
        fileOut.write("end_header\n")

        for i in range(numVertices):
            # writes "x y z" of current vertex
            fileOut.write(str(points[i,0]) + " " + str(points[i,1]) + " " + str(points[i,2]) + "255\n")


        # Writes faces
        for f in faces:
            #print(f)
            # WARNING: Subtracts 1 to vertex ID because PLY indices start at 0 and OBJ at 1
            fileOut.write("3 " + str(f[0]) + " " + str(f[1]) + " " + str(f[2]) + "\n")

# if __name__ == '__main__':
#     mesh_in = pv.read('test.ply')
#     new_mesh = pyacvd.Clustering(mesh_in)
#     new_mesh.subdivide(3)
#     new_mesh.cluster(5023)
#     mesh_in = new_mesh.create_mesh()
#
#     vertlist = np.array(mesh_in.points)
#     trilist = np.array(mesh_in.faces)
#     print('vertices = ', vertlist.shape)
#     print('triangles = ', trilist.shape)
#
#     numFaces = mesh_in.n_faces
#     trilist = trilist.reshape((numFaces, 4))[::, 1:4]
#
#     # create an instance of the TriMesh class
#     cranial_mesh = tm.TriMesh(trilist, vertlist)
#
#     # Laplacian matrix
#     L = cranial_mesh.laplacianmatrix('half_cotangent')
#     L_numpy = np.array(L)
#
#     # basis functions
#     sphara_basis = sb.SpharaBasis(cranial_mesh, 'unit')
#     basis_functions, natural_frequencies = sphara_basis.basis()
#
#     # sphinx_gallery_thumbnail_number = 2
#     figsb1, axes1 = plt.subplots(nrows=2, ncols=4, figsize=(9, 5),
#                                  subplot_kw={'projection': '3d'})
#
#     BF_list = [4, 10, 50, 100, 200, 500, 1000, 4000]
#
#     for i in range(np.size(axes1)):
#         colors = np.mean(basis_functions[trilist, BF_list[i]], axis=1)
#         ax = axes1.flat[i]
#         plt.grid()
#         ax.view_init(elev=70., azim=15.)
#         ax.set_aspect('auto')
#
#         trisurfplot = ax.plot_trisurf(vertlist[:, 0], vertlist[:, 1],
#                                       vertlist[:, 2], triangles=trilist,
#                                       cmap=plt.cm.bwr,
#                                       edgecolor='white', linewidth=0.)
#         trisurfplot.set_array(colors)
#         ax.set_xticklabels([])
#         ax.set_yticklabels([])
#         ax.set_zticklabels([])
#
#         plt.grid()
#
#     cbar = figsb1.colorbar(trisurfplot, ax=axes1.ravel().tolist(), shrink=0.75,
#                            orientation='vertical', fraction=0.05, pad=0.05,
#                            anchor=(0.5, -4.0))
#
#     plt.subplots_adjust(left=0.0, right=1.0, bottom=0.08, top=1.0)
#     plt.show()
#     # plt.savefig('spectral.svg', facecolor='w', edgecolor='w',
#     #             orientation='portrait', transparent=False, pad_inches=0.1)
#
#
#     fig_dir = ""
#     for i in range(len(BF_list)):
#         verts = LBO_reconstruction(basis_functions, BF_list[i])
#         filename = 'mesh_' + str(BF_list[i]) + '.ply'
#         my_write_ply_file(verts, trilist, fig_dir+filename)


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


    def LBO_reconstruction(self, basis_functions, EV_upper):
        # each colomn in EV matrix phi is an eigenvector
        phi = basis_functions  # phi = eigenvector matrix
        phi_rec = np.zeros([len(phi), len(phi)])
        phi_rec[:, :EV_upper] = phi[:, :EV_upper]
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
        reconstr_filepath = '../results/' + self.file_name +'_' + str(EV_upper) + '.ply'

        write_ply_file(self.reconstr_verts, self.trilist, reconstr_filepath)
        print('Mesh reconstructed (using {} eigen vectors): {}'.format(EV_upper, reconstr_filepath))



# class LossView:
#
#     @staticmethod
#     def plot_loss(train_list, val_list):
#         line1, = plt.plot(train_list, label="train loss", linestyle='--')
#         line2, = plt.plot(val_list, label="valid loss", linewidth=4)
#
#         # Create legend for both lines
#         plt.legend(handles=[line1, line2], loc='upper right')
#         plt.show()

if __name__ == '__main__':
    M = MeshHarmonics('../data/test.ply', n_vertices=1000)

    # reconstruct meshes
    M.LBO_reconstruction(basis_functions=M.basis_functions, EV_upper=1000)