# Repository for Land Cover Recognition Model Training Using Satellite Imagery

Welcome to the dedicated repository for advancing land cover recognition through the application of state-of-the-art models on satellite imagery. This repository serves as a comprehensive resource for researchers and practitioners in the field, providing access to research code, detailed setup instructions, and guidelines for conducting experiments with satellite image timeseries data.

## Featured Research Publications

This repository highlights contributions to the field through the following research publications:

- [ViTs for SITS: Vision Transformers for Satellite Image Time Series](https://openaccess.thecvf.com/content/CVPR2023/html/Tarasiou_ViTs_for_SITS_Vision_Transformers_for_Satellite_Image_Time_Series_CVPR_2023_paper.html) - Featured at CVPR 2023, this paper explores the application of Vision Transformers to Satellite Image Time Series analysis. For further details, please consult the [README_TSVIT.md](https://github.com/michaeltrs/DeepSatModels/blob/main/README_TSVIT.md) document.
- [Context-self contrastive pretraining for crop type semantic segmentation](https://ieeexplore.ieee.org/abstract/document/9854891) - 
Published in IEEE Transactions on Geoscience and Remote Sensing, this work introduces a novel supervised pretraining method for semantic segmentation 
of crop types exhibiti performance gains along object boundaries. Additional information is available in the [README_CSCL.md](https://github.com/michaeltrs/DeepSatModels/blob/main/README_CSCL.md) document.

## Environment Setup

### Installation of Miniconda
For the initial setup, please follow the instructions for downloading and installing Miniconda available at the [official Conda documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html).

### Environment Configuration
1. **Creating the Environment**: Navigate to the code directory in your terminal and create the environment using the provided `.yml` file by executing:

        conda env create -f deepsatmodels_env.yml

2. **Activating the Environment**: Activate the newly created environment with:

        source activate deepsatmodels

3. **PyTorch Installation**: Install the required version of PyTorch along with torchvision and torchaudio by running:

        conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia

4. **Extra Packages**: Install additional packages that require PyTorch:

        conda install timm kornia -c conda-forge

5. **Environment variables**: Fix CUDNN env vars.

        conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
        python3 -m pip install nvidia-cudnn-cu11==8.7.0.84

        mkdir -p $CONDA_PREFIX/etc/conda/activate.d

        echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

        source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh


## Experiment Setup

- **Configuration**: Specify the base directory and paths for training and evaluation datasets within the `data/datasets.yaml` file.
- **Experiment Configuration**: Use a distinct `.yaml` file for each experiment, located in the `configs` folder. These configuration files encapsulate default parameters aligned with those used in the featured research. Modify these `.yaml` files as necessary to accommodate custom datasets.
- **Guidance on Experiments**: For detailed instructions on setting up and conducting experiments, refer to the specific README.MD files associated with each paper or dataset.

## License and Copyright

This project is made available under the Apache License 2.0. Please see the [LICENSE](https://github.com/michaeltrs/DeepSatModels/blob/main/LICENSE.txt) file for detailed licensing information.

Copyright © 2023 by Michail Tarasiou
