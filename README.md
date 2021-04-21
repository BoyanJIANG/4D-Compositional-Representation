# 4D Compositional Representation

This is an implementation of the CVPR'2021 paper "Learning Compositional Representation for 4D Captures with Neural ODE".

In this paper, we introduce a compositional representation for 4D captures, i.e. a deforming 3D object over a temporal span, that disentangles shape, initial state, and motion respectively.
Our model takes a point cloud sequence performing non-rigid deformation as input and outputs corresponding 3D mesh sequence.

Please check our [paper](https://arxiv.org/abs/2103.08271) and the [project webpage](https://boyanjiang.github.io/4D-CR/) for more details.

If you have any question, please contact Boyan Jiang  <byjiang18@fudan.edu.cn>.

#### Citation

If you use this code for any purpose, please consider citing:

```
@inProceedings{jiang2021learning,
  title={Learning Compositional Representation for 4D Captures with Neural ODE},
  author={Jiang, Boyan and Zhang, Yinda and Wei, Xingkui and Xue, Xiangyang and Fu, Yanwei},
  booktitle={CVPR},
  year={2021}
}
```

## Installation

First, install PyTorch and other dependencies using
```
pip install -r requirements.txt
```

Next, compile the extension modules via
```
python setup.py build_ext --inplace
```

Our code has been tested with Python 3.6, PyTorch 1.3.1, CUDA 10.1 on Ubuntu 16.04.


## Data and Model
### Pre-trained Model
We provide pre-trained model on the D-FAUST dataset. 
Please download from [this links](https://drive.google.com/file/d/1zStOVZn-UzfgznyJywFtrDrdM-5Q2ycz/view?usp=sharing), 
and unzip to the `out/4d_cr_pretrained` folder.
```
mkdir -p out/4d_cr_pretrained
unzip pretrained_model.zip -d out/4d_cr_pretrained
```

### D-FAUST Dataset

To perform identity exchange during training, we generate the training data based on the [Dynamic FAUST (D-FAUST)](https://dfaust.is.tue.mpg.de/) dataset. And for validation/evaluation, we use the original validation/test set in the D-FAUST.

Because the full training set is very large, you need to build it yourself through the following steps:
1. Download our retrieved SMPL shape and pose parameters of D-FAUST meshes, and the validation/test data from [here](https://drive.google.com/file/d/1dRWeYJr4jlJmI9aJolZLe-K-qiVbmk44/view?usp=sharing).
2. unzip it to the `data/` folder, the train/validate/test split can be found in `data/human_dataset/test/D-FAUST`
    ```
    unzip 4dcr_data.zip -d data/
    ```
3. Generate SMPL mesh vertices using:
    ```
    python scripts/gen_mesh_verts.py
    ```
    After that, you can find the mesh vertex data stored in `.npy` form in `data/human_dataset/all_train_mesh`.
4. Sample point clouds and query points for training:
    ```
    bash scripts/build_dataset.sh
    ```
5. After completing the above steps, the training data is organized as following structure:
    ```
    train/D-FAUST/
    |50002/
        |50002_chicken_wings/
            |pcl_seq
                |00000000.npz
                |00000001.npz
                |...
                |00000215.npz
            |points_seq
                |00000000.npz
                |00000001.npz
                |...
                |00000215.npz
        |50002_hips/
            |pcl_seq
                |...
            |points_seq
                |...
        |...
        |50027_shake_shoulders/
            |pcl_seq
                |...
            |points_seq
                |...
    |50004/
        |...
    |50027/
        |...
    ```


### Warping Cars Dataset
We build a Warping Cars dataset using the approach introduced in [Occupancy Flow](https://github.com/autonomousvision/occupancy_flow).
As we mainly focused on D-FAUST, we do not provide this dataset.
Please contact the authors of Occupancy Flow to request the code for generating warping cars.


## Quick Demo

You can run the demo code on the data samples in the `data/demo` folder to obtain mesh sequences.
```
python generate.py configs/demo.yaml
```
The output is stored in the `out/4d_cr_pretrained/generation` folder.

## Detailed Instruction
### Training
You can train the network from scratch by executing the scrip:
```
python train.py configs/4d_cr.yaml
```
The training log file and the saved models will be placed in the `out_dir` you set in `4d_cr.yaml`.

### Generation
After training, you can generate and visualize the predicted mesh sequences for testing data. Simply run:
```
python generate.py configs/4d_cr.yaml
```
This script should automatically create a `generation/` folder in the output directory of the trained model.

### Evaluation
For evaluation the trained model, you should run the script below after completing the generation process:
```
python eval.py configs/4d_cr.yaml
```
By default, we use the original testing set in the D-FAUST to evaluate our model.
Because the mesh outputted by our model is not aligned with the test data,
it will takes an extra step to align the point cloud with the output for calculating metrics when fetching the data.
You can check this step in `lib/data/field.py`.

### Motion Transfer
To performing 3D motion transfer, you can run:
```
python motion_transfer.py configs/4d_cr.yaml
```
You can change the identity and motion sequences in `motion_transfer.py`.
This script will generate novel mesh sequence, which transfers the motion from motion sequence to the human from identity sequence.

### 4D Completion & Future Prediction
You can conduct 4D completion and future motion prediction with the trained model by running:
```
python 4d_completion.py configs/4d_cr.yaml --experiment [temporal or spatial or future]
```
This script will disable the encoder and freeze the parameters of decoder in our model,
and optimize latent codes with back-propagation.


## Further Information
### Occupancy Flow
This project is build upon [Occupancy Flow](https://github.com/autonomousvision/occupancy_flow).
If you are interested in 4D representation, please check their project which is the pioneer work in this area.
We also provide the code for Occupancy Flow to perform motion transfer, 4D completion and future prediction, you can download their pretrained model from the link above and run with `configs/oflow_pretrained.yaml`.

### Neural Pose Transfer
While our method realizes motion transfer, it also transfers the initial pose to a new identity. If you are interested in pose transfer, please take a look at [Neural Pose Transfer](https://github.com/jiashunwang/Neural-Pose-Transfer), which is the state-of-the-art method.

## License
Apache License Version 2.0
