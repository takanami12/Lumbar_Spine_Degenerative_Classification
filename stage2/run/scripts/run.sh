#!/bin/bash
set -e

echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
echo "@        Download and Preprocess the Data          @"
echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
python3 src/utils/prepare_data.py
python3 src/utils/generate_kfold.py

echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
echo "@        Train Keypoint Detection Models           @"
echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
### Moyashii
# Create sagittal dataset v2 and train the sagittal keypoint models
python3 src/stage1/moyashii/tools/create_dataset_v2.py moyashii 2024
python3 src/stage1/moyashii/sagittal/train.py src/stage1/moyashii/sagittal/config/v2_0008_config.yaml 0008 --dst_root stage1/moyashii/v2/0008

# Create axial dataset v5 and train the axial keypoint models
python3 src/stage1/moyashii/tools/create_dataset_v5.py moyashii 2024
python3 src/stage1/moyashii/axial/train.py src/stage1/moyashii/axial/config/v5_0003_config.yaml 0003 --dst_root stage1/moyashii/v5/0003

echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
echo "@          Train Classification Models             @"
echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"

### Moyashii
# Create the dataset using keypoint detection models
python3 src/stage2/moyashii/tools/create_dastaset_v9.py moyashii
# Train the center classification models (using dataset v9)
python3 src/stage2/moyashii/train.py src/stage2/moyashii/config/0053.yaml 0053 --dst_root stage2/moyashii/center/v9/0053_42 --options folds=[0,1,2,3,4] seed=42
python3 src/stage2/moyashii/train.py src/stage2/moyashii/config/0053.yaml 0053 --dst_root stage2/moyashii/center/v9/0053_2024 --options folds=[0,1,2,3,4] seed=2024
# Train the side classification models (using dataset v9)
python3 src/stage2/moyashii/train.py src/stage2/moyashii/config/0054.yaml 0054 --dst_root stage2/moyashii/side/v9/0054_42 --options folds=[0,1,2,3,4] seed=42
python3 src/stage2/moyashii/train.py src/stage2/moyashii/config/0054.yaml 0054 --dst_root stage2/moyashii/side/v9/0054_2024 --options folds=[0,1,2,3,4] seed=2024

### suguuuuu

echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
echo "@                     Predict                      @"
echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
python3 src/predict.py