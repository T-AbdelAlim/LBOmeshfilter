import os
from harmonics.SurfaceLaplacian import MeshHarmonics


if __name__ == '__main__':
    print(f'\nMesh Processing Laplacian-Beltrami Operator ')
    print(f'=============================================')
    cwd = os.getcwd().replace('\\', '/')

    eigenvectors=[10, 50, 75, 100, 150, 300]

    M_ply = MeshHarmonics(cwd+'/data/ply_test.ply', n_vertices=1000)
    # reconstruct meshes
    M_ply.LBO_reconstruction(basis_functions=M_ply.basis_functions, EV_upper=25)
    # plot harmonics
    M_ply.plot_harmonics(EV_list=eigenvectors)

    M_obj = MeshHarmonics(cwd+'/data/obj_test.obj', n_vertices=1000)
    # reconstruct meshes
    M_obj.LBO_reconstruction(basis_functions=M_obj.basis_functions, EV_upper=25)
    # plot harmonics
    M_obj.plot_harmonics(EV_list=eigenvectors)

    M_stl = MeshHarmonics(cwd+'/data/stl_test.ply', n_vertices=1000)
    # reconstruct meshes
    M_stl.LBO_reconstruction(basis_functions=M_stl.basis_functions, EV_upper=25)
    # plot harmonics
    M_stl.plot_harmonics(EV_list=eigenvectors)
