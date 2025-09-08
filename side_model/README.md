1. This still need the dataset path in order to run (waiting for you guys to provide 😊)
2. Replace the path for dataset folder in the main.py there are an example in there:

```
run_training(
    config_path="config/3.yaml",
    experiment="3",
    dst_root="stage2/tkmn/side/v9/0057",
    options=["folds=[0,1,2,3,4]"]
)
```

3. The code for data augmentation or dataphraze still assume the same as the orignal code, so if there is changes in term of output or path change in ***train.py***