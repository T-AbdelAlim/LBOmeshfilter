import os
from harmonics.SurfaceLaplacian import MeshHarmonics


if __name__ == '__main__':
    print(f'\nMesh Processing Laplacian-Beltrami Operator ')
    print(f'=============================================')
    cwd = os.getcwd().replace('\\', '/')

    eigenvectors=[10, 50, 75, 100, 150, 300]

    M_ply = MeshHarmonics(cwd+'/examples/data/ply_bunny.ply', n_vertices=1000, examples=True)
    # reconstruct meshes
    M_ply.LBO_reconstruction(basis_functions=M_ply.basis_functions, EV_upper=20)
    # plot harmonics
    M_ply.plot_harmonics(EV_list=eigenvectors)
