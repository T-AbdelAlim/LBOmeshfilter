import os
from harmonics.SurfaceLaplacian import MeshHarmonics
from pathlib import Path


if __name__ == '__main__':
    print(f'\nMesh Processing Laplace-Beltrami Operator ')
    print(f'=============================================')

    examples_path = Path("./examples/")
    try:
        os.mkdir(Path(examples_path,'results'))
    except FileExistsError:
        pass

    eigenvectors=[10, 300]
    # calculate basis functions and natural frequencies:
    M_ply = MeshHarmonics(Path(examples_path, 'data/ply_bunny.ply'), n_vertices=1000, examples=False)

    # plot harmonics
    M_ply.plot_harmonics(EV_list=eigenvectors)

    # reconstruct mesh using the first e eigenvectors
    for e in eigenvectors:
        # reconstruct meshes
        M_ply.LBO_reconstruction(basis_functions=M_ply.basis_functions, EV_upper=e)



