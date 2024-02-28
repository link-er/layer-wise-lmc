Structure of the code:
1) LLMC:
- data folder will contain all the downloaded datasets
- traces folder will have stored checkpoints for further evaluations
- federated_train.py and single_train_exp.py are for training models and saving checkpoints
- layerwise_nonconvex_compute.py and layerwise_nonconvex_heatmap.py are for computing and plotting layer-wise barriers
- layer_cumulative_compute.py and layer_cumulative_heatmap.py are for computing and plotting cumulative barriers
- layerwise_robustness_eval.py and layerwise_robustness_plot.py are for computing and plotting robustness in averaging and random directions
- eval_interpolation_multiple_layers.py and eval_interpolation_all.sh are scripts for robustness evaluation on ViTis from https://github.com/tml-epfl/sharpness-vs-generalization
2) subspaceExperiments contains code for subspace noise robustness experiments


Note: for non-iid data separation we used scripts from https://github.com/TsingZ0/PFL-Non-IID

Note: for training LLMs locally we used https://github.com/epfml/llm-baselines

Note: for the robustness tests on ViTs (eval_interpolation_multiple_layers.py and eval_interpolation_all.sh) the missing files (like data.py, models.py, utils.py) can be found in the original repo https://github.com/tml-epfl/sharpness-vs-generalization