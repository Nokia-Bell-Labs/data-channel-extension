## DEX: Data Channel Extension for Efficient CNN Inference on Tiny AI Accelerators (NeurIPS '24)

This is the official PyTorch Implementation of "DEX: Data Channel Extension for Efficient CNN Inference on Tiny AI Accelerators" (NeurIPS '24) 
by 
[Taesik Gong](https://taesikgong.com/), 
[Fahim Kawsar](https://www.fahim-kawsar.net/), and
[Chulhong Min](http://chulhongmin.com/).

The codebase is forked from the [AI8X Training repository](https://github.com/analogdevicesinc/ai8x-training), the official repository for MAX78000 and MAX78002 AI accelerators, and additional implementations are added for the DEX project.

### Tested Environment

- Ubuntu 22.04
- NVIDIA A40 GPUs

### Step-by-Step Guide


1. Move to the root directory:
   ```bash
   cd data-channel-extension
   ```

2. Build the Docker image:
   ```bash
   ./docker/build_image.sh
   ```

3. Run the Docker container:
   ```bash
   ./docker/run_container.sh
   ```

4. Execute the container:
   ```bash
   ./docker/exec_container.sh
   ```

   This starts the SSH service in the Docker container.
 
   **Password**: `root`

5. Run the training script:
   ```bash
   ./scripts/dex/train_all.sh
   ```

6. Download and extract datasets:

   - **Caltech101**:
     - Download: [Caltech101](https://drive.google.com/file/d/137RyRjvTBkBiIfeYBNZBtViDHQ6_Ewsp)
     - Extract under `data/Caltech101/caltech101/101_ObjectCategories`
   
   - **Caltech256**:
     - Download: [Caltech256](https://drive.google.com/file/d/1r6o0pSROcV1_VwT4oSjA2FBUSCWGuxLK)
     - Extract under `data/Caltech256/caltech256/256_ObjectCategories/`
   
   - **ImageNette**:
     - Download: [ImageNette](https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz)
     - Extract under `data/Imagenette/`
   
   - **Food101**:
     - The required data will be automatically downloaded.

7. Modify the `train_all.sh` file to run specific experiments as needed. By default, it launches training for ImageNette with SimpleNet.


### Licensing

This repository is multi licensed. All the files copyrighted by Nokia are licenced with BSD-3-Clause-Clear.
All the remainint files are licensed with Apache License 2.0 and copyrghted to Maxim Integrated Products and Analog Devices according to the copyright notices [here](https://github.com/analogdevicesinc/ai8x-training/).

For more detailed liceensing information please fo to the [LICENSE](LICENSE) file. 

### Citation

```
@inproceedings{gong2024dex,
    title={{DEX}: Data Channel Extension for Efficient CNN Inference on Tiny AI Accelerators},
    author={Gong, Taesik and Kawsar, Fahim and Min, Chulhong},
    booktitle={Thirty-eighth Conference on Neural Information Processing Systems},
    year={2024}
}
```
