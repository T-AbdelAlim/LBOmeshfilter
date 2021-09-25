import os
import matplotlib.pyplot as plt
from harmonics.SurfaceLaplacian import MeshHarmonics


if __name__ == '__main__':
    print(f'\nMesh Processing Laplace-Beltrami Operator ')
    print(f'=============================================')
    data_path = os.getcwd().replace('\\', '/') + '/data/'

    eigenvectors=[50, 100, 250, 500, 750, 950]

    ref_mesh = MeshHarmonics(data_path + '/clean.ply', n_vertices=1000)
    ref_mesh.plot_harmonics(EV_list=eigenvectors)
    # calculate basis functions and natural frequencies:
    mesh = MeshHarmonics(data_path + '/raw.ply', n_vertices=1000)
    mesh.plot_harmonics(EV_list=eigenvectors)

    eigen_list = []
    error_list = []
    for i in range(10):
        print(i)
        mesh.LBO_reconstruction(basis_functions=mesh.basis_functions, EV_upper=i, write_ply=True)
        mesh.distance_error(mesh.reconstr_verts, ref_mesh.pvmesh.points)

        eigen_list.append(i)
        error_list.append(mesh.RSS)

    plt.scatter(eigen_list, error_list)
    plt.show()




    # # reconstruct mesh using the first e eigenvectors
    # for e in range(len()):
    #     mesh.LBO_reconstruction(basis_functions=mesh.basis_functions+1, EV_upper=e)

    # mesh.plot_harmonics(EV_list=eigenvectors)