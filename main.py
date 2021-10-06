import os
import matplotlib.pyplot as plt
import pyvista as pv
from pathlib import Path
from harmonics.SurfaceLaplacian import MeshHarmonics
from harmonics.ply_writer import write_ply_file

if __name__ == '__main__':
    print(f'\nMesh Processing Laplace-Beltrami Operator ')
    print(f'=============================================')
    data_path = Path("./data/")

    eigenvectors=[50, 250, 500, 750, 1000, 2250, 3500, 4000, 4999]

    ref_mesh = MeshHarmonics(Path(data_path, 'clean.ply'), n_vertices=5000)
    ref_mesh.plot_harmonics(EV_list=eigenvectors)
    plt.show()
    # calculate basis functions and natural frequencies:
    mesh = MeshHarmonics(Path(data_path, 'raw.ply'), n_vertices=5000)
    mesh.plot_harmonics(EV_list=eigenvectors)
    plt.show()

    eigen_list = []
    DE_list = [] # coordinate distance errors
    NE_list = [] # point normal errors
    VE_list = [] # volume errors

    for i in range(50,200,5):
        mesh.LBO_reconstruction(basis_functions=mesh.basis_functions, EV_upper=i, write_ply=True)
        write_ply_file(mesh.reconstr_verts, mesh.trilist, 'temp.ply') # create temp file

        temp_mesh = pv.read('temp.ply')
        mesh.calc_distance_error(temp_mesh.points, ref_mesh.pvmesh.points)
        mesh.calc_normal_error(temp_mesh.point_normals, ref_mesh.pvmesh.point_normals)
        mesh.calc_volume_error(temp_mesh, ref_mesh.pvmesh)

        eigen_list.append(i)
        DE_list.append(mesh.dist_error)
        NE_list.append(mesh.norm_error)
        VE_list.append(mesh.volume_error)

    os.remove('temp.ply')
    plt.scatter(eigen_list, DE_list)
    plt.ylabel('Dist')
    plt.show()

    plt.scatter(eigen_list, NE_list)
    plt.ylabel('Norm')
    plt.show()

    plt.scatter(eigen_list, VE_list)
    plt.ylabel('Vol')
    plt.show()


    # # reconstruct mesh using the first e eigenvectors
    # for e in range(len()):
    #     mesh.LBO_reconstruction(basis_functions=mesh.basis_functions+1, EV_upper=e)

    # mesh.plot_harmonics(EV_list=eigenvectors)