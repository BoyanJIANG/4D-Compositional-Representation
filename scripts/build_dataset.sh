NPROC=0
BUILD_PATH='data/human_dataset/train'  # output folder
INPUT_PATH='data/human_dataset/all_train_mesh'  # mesh folder

mkdir -p $BUILD_PATH

echo " Building Human Dataset for 4DCR Project."
echo " Input Path: $INPUT_PATH"
echo " Build Path: $BUILD_PATH"

echo "Sample mesh..."
python scripts/sample_mesh.py $INPUT_PATH \
    --out_folder $BUILD_PATH \
    --n_proc $NPROC --resize \
    --points_folder points_seq \
    --pointcloud_folder pcl_seq \
    --points_uniform_ratio 0.5 \
    --overwrite --float16 --packbits
echo "done!"



