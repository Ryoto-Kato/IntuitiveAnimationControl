
# Deformation Learning with structured 3D Gaussian for Intuitive Animation of Photo-realistic Head Avatars

**> code comming soon...**

<p align="center">
Ryoto Kato
</p>

<p align="center">
Techcnical University of Munich
</p>

<p align="center">
    <img src = https://github.com/Ryoto-Kato/IntuitiveAnimationControl/assets/127607089/31ee2b5b-484a-47c0-bc7f-6077981eeef3 alt="Results"/>
</p>

We presented deformation learning of a photorealistic head avatar using structured 3D Gaussians for intuitive control and real-time realistic animation rendering. Our deformation learning is able to produce global deformation components that support rough fitting as a foundation for plausible facial expression, as well as local deformation components that allow for more comprehensive modelling of facial expression. Our local deformation components, in particular, have the potential to be used for the automatic facial expression fitting task by finding a plausible deformation at an arbitrary point on the face using a probabilistic approach and capturing latent deformation space with a Gaussian distribution mixture in our deformation learning.

[paper (comming soon)]()
<!-- \[[paper](./doc_sources/3DSSL_finalpaper.pdf)\] -->

# Pipeline overview
![Pipeline Overview](./doc_sources/Pipeline-Overview.png)

# Requirements
- Linux x86-64 (Debian is preferrable)
- python (conda)
- GPU 10GB VRAM (at least 8GB)
    - must fulfill requirements for 3D Gaussian splatting pipelines

# Setup
## set up conda env
```sh
conda create --name 3dsrf --file requirements.txt
conda activate 3dsrf
```
## Multi-face dataset setup
### Download tracked meshes (66 expresssions) and multi-views (38 cameras) of the target person (e.g., 6795937)
- Follow the guide to download them from [official-repo](https://github.com/facebookresearch/multiface).
### Convert data structure into following by using `./src/tools/dataset_composer.py`
1. Make `dataset` at the same level with `3DSSL-WS23_IntuitiveAnimation` and create new folders (`multi_views`, `multiface`)
2. Clone the `tracked_mesh`, `meta_data` folder from downloaded multi-face data into folder `multiface`
3. [TODO] set `path_to_imageFolder` to the path to image folder which is downloade from multi-face and set `ID` appropriately in `./src/tools/dataset_composer.py`,
    - e.g.,
        ```py
        ID = '6795937'
        path_to_imageFolder = "/path_to_multi-face/path_to_download_folder/m--20180227--0000--6795937--GHS/images"
        ```
4. Run `./src/tools/dataset_composer.py` 

### Final data structure
- from now, `path_to_dataset` = ./3DSSL-WS23-IntuitiveAnimation/dataset
```sh
├── 3DSSL-WS23_IntuitiveAnimation
    ├── src
    ├── dataset
        ├── multi_views
            ├──6795937
                ├── E001_Neutral_Eyes_Open
                    ├── 000102
                        ├── 000102.obj
                        ├── 000102.ply
                        ├── 000102_subd2.ply
                        ├── 000102_subd.ply
                        ├── 000102_transform.txt
                        ├── 400002.png
                        ├── 400004.png
                        ...
                    ├── 000108
                    ├── 000114
                    ...
        
        ├── multiface
            ├── tracked_mesh
                ├── E001_Neutral_Eyes_Open
                    ├── 000102.obj
                    ├── 00102_transform.txt
                    ...
                ├── E002...
            ├── meta_data
                ├── KRT
                ...

```
## Facemask generation
- We provide facemasks (.obj for visualization and .pkl for subsequent computation) in `./samples` which are requied for further computation
    - 3DGS `./samples/3dgs`
        - 87652 (after twice subdivision of multi-face trackedd mesh): `FaceMask_sample_subd2_face_trimesh.obj`
        - 5509 (multi-face tracked mesh): `FaceMask_sample_face_trimesh.obj`
- You can generate a facemask by using `./samples` and `GetMask_mesh.ipynb`
- Visualization of a face mesh and its facemask can be done with `./src/tools/FaceMask_and_Mesh_visualizer.py`

## Deformation Learning with structured 3D Gaussians
### Setting up 3D Gaussian splatting pipeline by referring
1. Clone `gaussian-splatting` from the [official repo](https://github.com/graphdeco-inria/gaussian-splatting).
    - set up environment for original `gaussian-splatting`
        - additionally, `pip install pytranform3d`
    - [option] Clone `DeformationLearning_3DGS` from [repo](git@github.com:Ryoto-Kato/DeformationLearning_3DGS.git) or recursive clone of `3DSSL-WS23_IntuitiveAnimation`
    - Copy all source codes in `./DeformationLearning_3DGS` and paste them into the `gaussian_splatting` to use our customized gaussian splatting codes

2. [TODO] Set constant appropriately as your environment
    ```py 
    # multiface_dataset_readers.py and original_train.py
    path_to_dataset = ...
    # original_train.py
    path_to_3WI = ...
    ID = "6795973"
    path_to_dataset = ...
    path_to_output = ...
    ```

### optimization of structured 3D Gaussians
```sh
# activate the conda env
conda activate gaussian_splatting

python original_train.py #check options in the source code
tensorboad --logdir=path_to_output
```

## Deformation components analysis (global/local) 
### How to obtain the global/local deformation components given trained 3D Gaussians (.pkl)
1. Serialized trained 3D Gaussian properties in .pkl (use ./src/utils/pickel_io.py)
2. Convert the .pkl to Hierachcal data format (.hdf5)
    - Set the constant parameters
        ```py
        path_to_output =...
        session = ... # training session id
        ```
    - Use DeformatioLearning/3DGS_DC.ipynb
        - Input: `GaussianProp` data structure in .pkl
    
            ```py
            @dataclass
            class GaussianProp:
                xyz: np.ndarray 
                normals: np.ndarray
                f_dc: np.ndarray #this is SH_coeffs, needs to be converted to RGB by SH2RGB
                f_rest: np.ndarray
                opacities: np.ndarray
                scale: np.ndarray
                rotation: np.ndarray
                covariance: np.ndarray
            ```

        - Output: 
            
            ```sh
            # original trained 3D Gaussians
            /src/samples/deformation_components/trained_3dgs/<session_name>_87652.hdf5
            # after 1x downsampling (de-subdivision)
            /src/samples/deformation_components/trained_3dgs/<session_name>_21954.hdf5
            # after 2x downsampling (de-subdivision)
            /src/samples/deformation_components/trained_3dgs/<session_name>_5509.hdf5
            ```
       Contents 
        ```py
            # where the number of vertex = 87652
            xyz shape: (330, 262956) # 3D coordinate of center of Gaussians 
            normals shape: (330, 262956) # Normal vectors at the center of Gaussians
            rgbs shape: (330, 262956) # RGB obtained by converting the f_dc
            f_dc shape: (330, 262956) # 0-deg spherical harmonics (SH) lighting coefficients
            f_rest shape: (330, 3944340) # rest of the SH lighting coefficients
            opacities shape: (330, 87652) # opacities of Gaussians
            scales shape: (330, 262956) # scales of 3D Gaussians
            rotation shape: (330, 350608) # rotation of 3D Gaussians
        ```
3. Obtain deformation components for global effects (with PCA) and local effects (with SLDC) by using `src/tools/PCA_MBSPCA_3DGS.py`
    ### PCA on selected attributes `[xyz, f_dc, scale, rotation]`
    Apply PCA/MBSPCA on selected attributes of 3D Gaussians
    - input: `<session_name>_87652.hdf5`
    - output: `3dgs_87652_ALL_5perExp_trimesh_dcs.hdf5`
    
    
        - If we apply downsampling to trained 3D Gaussians in advance, you need to add `--upsampling` to store the standard deviation and average of the undownsampled 3D Gaussians for later upsampling
       
    ```sh 
    # where you are applying PCA and MBSPCA on the trained 3D Gaussians
    python PCA_MBSPCA_3DGS.py --path2folder="../samples/deformation_components/trained_3dgs" --hdf5_fname="<session_name>_87652.hdf5"
    # --selectedAttribs=['xyz', 'f_dc', 'scales', 'rotation'] (Default)
    ```

    ```sh
    # where you are applying PCA and MBSPCA on the downsampled 3D Gaussians
    python PCA_MBSPCA_3DGS.py --path2folder="../samples/deformation_components/trained_3dgs" --hdf5_fname="<session_name>_5509.hdf5" --upsampling
    ```
    - [optional] you can get deformation components from MiniBatch sparse PCA using `scikit-learn`. but this is not appropriate method for local effects due to the inappropriate constraints in the matrix factorization. Look at the details [here](https://scikit-learn.org/stable/modules/decomposition.html#sparsepca) 
    
    ### PCA on each attribute `[xyz, f_dc, scale, rotation]`
    **This step is required for the subsequent SLDC step**
    - input:`<session_name>_87652.hdf5`
    - output:`3dgs_87652_[xyz]_PCAMBSPCA_5perExp_trimesh_dcs.hdf5`
        - We will apply PCA on the data matrix which concatenates the 3D Gaussians representation of each face with the center of Gaussian `xyz` 

    ```sh
    python PCA_MBSPCA_3DGS_separateAttribs.py --hdf5_fname="f336a291-bnotALLcam_datamat_87652.hdf5"
    ```

    ### SLDC on single attribute `[xyz]`
    - input: `3dgs_87652_xyz_PCAMBSPCA_5perExp_trimesh_dcs.hdf5` (from **PCA on single attribute**)
    - output:
        - `gauss_3dgs_87652_xyz_SLDC_5perExp_trimesh_dcs.hdf5` (with `--gauss`) 
            - otherwise `3dgs_87652_xyz_SLDC_5perExp_trimesh_dcs.hdf5`
        - To make sure that you apply PCA/SLDC on the same data, we used the output .hdf5 from previous step and apply SLDC on the data matrix from `3dgs_87652_PCAMBSPCA_5perExp_trimesh_dcs.hdf5`
    ```sh 
    python SLDC_trained3dgs.py --path2folder="../samples/deformation_components/trained_3dgs" --hdf5_fname="3dgs_87652_xyz_PCAMBSPCA_5perExp_trimesh_dcs.hdf5" --gauss # without --gauss runs original SLDC
    ```

[optional] upsampling process
- input: `3dgs_5509_ALL_5perExp_trimesh_dcs.hdf5` or `3dgs_21954_ALL_5perExp_trimesh_dcs.hdf5`
- output: `upsampled_3dgs_5509_ALL_5perExp_trimesh_dcs.hdf5` or `upsampled_3dgs_21954_ALL_5perExp_trimesh_dcs.hdf5`
    - upsampling the number of Gaussians and their attributes acordingly


```sh
python upsampling_DCs.py --numGauss=21954
```

## Animation
```sh
# activate the conda env
conda activate gaussian_splatting

# Global deformation components
python original_render.py --path_to_hdf5="./output/f336a291-bnotALLcam/3dgs_87652_ALL_5perExp_trimesh_dcs.hdf5" --path_to_saveIMG=./output/f336a291-bnotALLcam/blendshape_result --dc_type=pca

# Local deformation components 
python original_render.py --path_to_hdf5="./output/f336a291-bnotALLcam/gauss_3dgs_87652_xyz_SLDC_5perExp_trimesh_dcs" --path_to_saveIMG=./output/f336a291-bnotALLcam/blendshape_result --dc_type=sldc
```

## Evaluation
- Comparison between 4 mehtods
    - **Ours (global)**: 3dgs_87652_ALL_5perExp_trimesh_dcs.hdf5
    - **COG-PCA**: 3dgs_87652_xyz_PCAMBSPCA_5perExp_trimesh_dcs.hdf5
    - **Ours (local)**: gauss_3dgs_87652_xyz_SLDC_5perExp_trimesh_dcs.hdf5
    - **T-SLDC**: 3dgs_87652_xyz_SLDC_5perExp_trimesh_dcs.hdf5
```sh
# activate the conda env
conda activate 3dsrf
# Run evaluation and visualization of deformation region 
python evalutation_DCs.py
```
