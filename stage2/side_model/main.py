from train import run_training

run_training(
    config_path="config/1.yaml",
    experiment="1",
    dst_root="",
    options=["folds=[0,1,2,3,4]"]
)
run_training(
    config_path="config/2.yaml",
    experiment="2",
    dst_root="",
    options=[""]
)
# run_training(
#     config_path="config/3.yaml",
#     experiment="3",
#     dst_root="",
#     options=[""]
# )
# run_training(
#     config_path="config/3.yaml",
#     experiment="3",
#     dst_root="stage2/tkmn/side/v9/0057",
#     options=["folds=[0,1,2,3,4]"]
# )
