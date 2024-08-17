# Sentinel-2 Active Fire Segmentation: Analyzing Convolutional and Transformer Architectures Knowledge Transfer, Fine-Tuning and Seam-Lines

This work analyze three different architectures (U-net, Deeplab v3+ and SegFormer) in regard of transfer learning for segment active fire using remote sensing images. In this study we pre-trained the networks using Landsat-8 images and then perform a transfer learning to Sentinel-2 using manually annotated images. To pre-train the base networks we use the dataset built by [Pereira et al. (2021)](https://www.sciencedirect.com/science/article/abs/pii/S092427162100160X), available on [GitHub/Google Drive](https://github.com/pereira-gha/activefire/). To fine-tune we use the dataset built by [Fusioka et al. (2024)](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10620606) available on [GitHub/Google Drive](https://github.com/Minoro/transfer-learning-landsat8-sentinel2). We evaluated the performance of the networks for active fire segmentation when performing the transfer learning and also the performance when facing images with seam-lines that causes many false positives detections. Experiments show that the proposed method achieves F1-scores of up to 88.4% for Sentinel-2 images, outperforming three threshold-based algorithms by at least 19%.

## Authors

[Andre Minoro Fusioka](https://github.com/Minoro)

[Gabriel Henrique de Almeida Pereira](https://github.com/pereira-gha)

[Bogdan Tomoyuki Nassu](https://github.com/btnassu)

[Rodrigo Minetto](https://github.com/rminetto)

# Dataset

To pre-train the networks we use the Landsat-8 of the dataset built by [Pereira et al. (2021)](https://www.sciencedirect.com/science/article/abs/pii/S092427162100160X). The dataset consists of images from around the world with active fire and their respective masks produced by three threshold algorithms and their combination by intersection and majority voting. In this work we use only the majority **voting masks**.

With the networks trained on Landsat-8 images, we fine-tune then using Sentinel-2 images manually annotated, we use the dataset built by [Fusioka et al. (2024)](https://ieeexplore.ieee.org/document/10620606). This dataset consist of 26 Sentinel-2 images manually annotated, which has 22 images with at least one fire pixel and 4 images without any fire pixel. While the dataset offer the images in patches (images cropped in 256x256 pixels), we choose to discard the patches with only no-data (black images) or partially with no-data. 

In addition to these dataset we analyzed a set of Sentinel-2 images without fire, but with seam-lines (artifacts caused by the composition of multiple captures, a known issue that results in misaligned bands visible in regions with clouds) that cause false positives in the active fire segmentation task.


## Adjusting Sentinel-2 images

As mentioned before, we have made a small change in the dataset provided by [Fusioka et al. (2024)](https://ieeexplore.ieee.org/document/10620606), we chose to not use the patches with "no-data". We provide two distinct scripts to adjust the dataset, the first script is the `src/utils/crop_images.py` this script can be used to crop the Sentinel-2 scenes available in the dataset. We also provide the script `src/utils/remove_no_data_patches.py` that will remove any patch with "no-data" from the samples.



# Training with Landsat-8 images

We trained a U-net, Deeplab v3+ and a SegFormer using the Landsat-8 images. To train the base networks we provide and script `src/train_landsat.py` that can be used to train the base networks. This script use the configuration defined in the `src/landsat_config.py` to train the network defined in the constant `MODEL`. You can override this configuration using the argument `--model` when invoking the script. You can also change the batch size (`--batch-size`), learning rate (`--lr`), number of epochs (`--epochs`) and early stopping patience (`--early-stopping-patience`). The final models will be saved on the folder defined in the constant `LANDSAT_OUTPUT_DIR` in the configuration script.

A example of invoking the training script to train the U-net:

```shell
python train_landsat.py --model unet
```

# Fine-tuning with Sentinel-2 images

After training the base models you can fine-tune them using the Sentinel-2 images. First, you need to define the CSV files that contains the folds that will be used to train, validate and test the models. Use the script `src/generate_folds.py` to create the this files. You can change the configuration of this script in the file `src/config.py`. The constant `MANUAL_ANNOTATIONS_IMAGE_PATH` must point to the folder holding the Sentinel-2 patches, the constant `MANUAL_ANNOTATIONS_MASK_PATH` must point to the manual annotations and the `SEAMLINE_IMAGES_PATH` must point to the folder with the seam-line patches. The code will check your images, counting  the fire pixels in each image, a summary will be saved in the file defined in the `CSV_NUM_FIRE_PIXELS_PER_PATCH_PATH` constant. The number of folds is defined by the constant `NUM_FOLDS` the default value is 5. If you want to use stratified folds, based in the categories `fire`, `seam-line` and `no-fire` you can set the constant `STRATIFIED_FOLDS` to `True`, the default is `False`. After setting the configurations in the `config.py` you can run the script:

```
python generate_folds.py
```

This code will set one split to test and one split to validation, all others splits will be used to train. If you don't want to define a validation fold you can set `GENERATE_VALIDATION_FOLD` to `False`.

With the folds defined you can fine-tune the network using the Sentinel-2 images. The script `src/transfer_learning.py` can be used to this task. The configurations used in this script is also defined in the `src/config.py` script. Alternatively, you can change the default configuration using the arguments available in the fine-tuning script. You can use the `--model` argument to define the base model (unet, deeplabv3+ or SegFormerB0) to fine-tune. The `--csv-folds-dir` argument can be used to point to the folder with the csv files with the folds definition, if you want to use specify the folds to be used you can set them in the `--fold` argument. You can also change the number of epochs to fine-tune the model with the argument `--epochs`, if you want to use the networks without any fine-tuning you can set the number of epochs to zero, this will only evaluate the model. Alternatively you can pass the argument `--no-tuning`, this will disable the fine-tuning step and execute only the evaluation.

When running this script it will fine-tuning and evaluate the model using the defined folds. After the fine-tuning step it will be saved the history in a json file inside the folder defined in the `OUTPUT_RESULTS_TRANSFER_LEARNING_PATH` constant, the weights will be saved in the folder defined in `OUTPUT_WEIGHTS_TRANSFER_LEARNING_PATH`, the results of the evaluation over the test fold will be saved in the folder `OUTPUT_RESULTS_TRANSFER_LEARNING_PATH` as a json file. 


# Citation

Full article is available in [IEEEXplore](https://ieeexplore.ieee.org/document/10636193). Bibtex citation:

```bibtex
@ARTICLE{Fusioka2024,
  author={Fusioka, Andre M. and Pereira, Gabriel H. D. A. and Nassu, Bogdan T. and Minetto, Rodrigo},
  journal={IEEE Geoscience and Remote Sensing Letters}, 
  title={Sentinel-2 Active Fire Segmentation: Analyzing Convolutional and Transformer Architectures, Knowledge Transfer, Fine-Tuning and Seam-Lines}, 
  year={2024},
  volume={},
  number={},
  pages={1-1},
  keywords={Remote sensing;Earth;Artificial satellites;Image segmentation;Satellites;Transfer learning;Training;active fire segmentation;transfer learning;fine-tuning;Sentinel-2 imagery;seam-lines},
  doi={10.1109/LGRS.2024.3443775}
}
```