import os
from harmonics.SurfaceLaplacian import MeshHarmonics


if __name__ == '__main__':
    print(f'\nMesh Processing Laplace-Beltrami Operator ')
    print(f'=============================================')
    examples_path = os.getcwd().replace('\\', '/') + '/data/'

    eigenvectors=[10, 50, 75, 100, 150, 300]

    # calculate basis functions and natural frequencies:
    M_ply = MeshHarmonics(examples_path+'/ply_test.ply', n_vertices=1000)

    # plot harmonics
    M_ply.plot_harmonics(EV_list=eigenvectors)

    # reconstruct mesh using the first e eigenvectors
    for e in eigenvectors:
        # reconstruct meshes
        M_ply.LBO_reconstruction(basis_functions=M_ply.basis_functions, EV_upper=e)


