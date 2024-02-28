# no augm, no mixup, smallest LR vs. largest LR, augm, mixup=0.5 (i.e., the farthest apart models)
python eval_interpolation_multiple_layers.py --dataset=cifar10 --model=vit_exp --model_width=512 --n_eval=1024 --eval_layer_str=weight --alpha_step=0.1 --exp_name='farthest_apart_models' --model_path1='models/2023-01-14 23:40:28.481 dataset=cifar10 model=vit_exp epochs=200 lr_max=0.005156 model_width=512 l2_reg=0.0 sam_rho=0.0 batch_size=128 frac_train=1 p_label_noise=0.0 lr_schedule=cyclic augm=False randaug=False seed=0 epoch=200.pth' --model_path2='models/2023-01-14 23:42:12.896 dataset=cifar10 model=vit_exp epochs=200 lr_max=0.056401 model_width=512 l2_reg=0.0 sam_rho=0.0 batch_size=128 frac_train=1 p_label_noise=0.0 lr_schedule=cyclic augm=True randaug=True seed=0 epoch=200.pth'
python eval_interpolation_multiple_layers.py --dataset=cifar10 --model=vit_exp --model_width=512 --n_eval=1024 --eval_layer_str=weight --alpha_step=0.1 --exp_name='farthest_apart_models_rev' --model_path1='models/2023-01-14 23:42:12.896 dataset=cifar10 model=vit_exp epochs=200 lr_max=0.056401 model_width=512 l2_reg=0.0 sam_rho=0.0 batch_size=128 frac_train=1 p_label_noise=0.0 lr_schedule=cyclic augm=True randaug=True seed=0 epoch=200.pth' --model_path2='models/2023-01-14 23:40:28.481 dataset=cifar10 model=vit_exp epochs=200 lr_max=0.005156 model_width=512 l2_reg=0.0 sam_rho=0.0 batch_size=128 frac_train=1 p_label_noise=0.0 lr_schedule=cyclic augm=False randaug=False seed=0 epoch=200.pth'
python eval_interpolation_multiple_layers.py --dataset=cifar10 --model=vit_exp --model_width=512 --n_eval=1024 --eval_layer_str=weight --alpha_step=0.1 --exp_name='farthest_apart_models_random' --interpolate_random_direction --model_path1='models/2023-01-14 23:40:28.481 dataset=cifar10 model=vit_exp epochs=200 lr_max=0.005156 model_width=512 l2_reg=0.0 sam_rho=0.0 batch_size=128 frac_train=1 p_label_noise=0.0 lr_schedule=cyclic augm=False randaug=False seed=0 epoch=200.pth' --model_path2='models/2023-01-14 23:42:12.896 dataset=cifar10 model=vit_exp epochs=200 lr_max=0.056401 model_width=512 l2_reg=0.0 sam_rho=0.0 batch_size=128 frac_train=1 p_label_noise=0.0 lr_schedule=cyclic augm=True randaug=True seed=0 epoch=200.pth'
python eval_interpolation_multiple_layers.py --dataset=cifar10 --model=vit_exp --model_width=512 --n_eval=1024 --eval_layer_str=weight --alpha_step=0.1 --exp_name='farthest_apart_models_rev_random' --interpolate_random_direction --model_path1='models/2023-01-14 23:42:12.896 dataset=cifar10 model=vit_exp epochs=200 lr_max=0.056401 model_width=512 l2_reg=0.0 sam_rho=0.0 batch_size=128 frac_train=1 p_label_noise=0.0 lr_schedule=cyclic augm=True randaug=True seed=0 epoch=200.pth' --model_path2='models/2023-01-14 23:40:28.481 dataset=cifar10 model=vit_exp epochs=200 lr_max=0.005156 model_width=512 l2_reg=0.0 sam_rho=0.0 batch_size=128 frac_train=1 p_label_noise=0.0 lr_schedule=cyclic augm=False randaug=False seed=0 epoch=200.pth'


# no augmentations: smallest LR vs. largest LR
python eval_interpolation_multiple_layers.py --dataset=cifar10 --model=vit_exp --model_width=512 --n_eval=1024 --eval_layer_str=weight --alpha_step=0.1 --exp_name='no_augm_small_vs_large_lr' --model_path1='models/2023-01-14 23:40:28.481 dataset=cifar10 model=vit_exp epochs=200 lr_max=0.005156 model_width=512 l2_reg=0.0 sam_rho=0.0 batch_size=128 frac_train=1 p_label_noise=0.0 lr_schedule=cyclic augm=False randaug=False seed=0 epoch=200.pth' --model_path2='models/2023-01-14 23:40:10.063 dataset=cifar10 model=vit_exp epochs=200 lr_max=0.155923 model_width=512 l2_reg=0.0 sam_rho=0.0 batch_size=128 frac_train=1 p_label_noise=0.0 lr_schedule=cyclic augm=False randaug=False seed=0 epoch=200.pth'
python eval_interpolation_multiple_layers.py --dataset=cifar10 --model=vit_exp --model_width=512 --n_eval=1024 --eval_layer_str=weight --alpha_step=0.1 --exp_name='no_augm_small_vs_large_lr_rev' --model_path1='models/2023-01-14 23:40:10.063 dataset=cifar10 model=vit_exp epochs=200 lr_max=0.155923 model_width=512 l2_reg=0.0 sam_rho=0.0 batch_size=128 frac_train=1 p_label_noise=0.0 lr_schedule=cyclic augm=False randaug=False seed=0 epoch=200.pth' --model_path2='models/2023-01-14 23:40:28.481 dataset=cifar10 model=vit_exp epochs=200 lr_max=0.005156 model_width=512 l2_reg=0.0 sam_rho=0.0 batch_size=128 frac_train=1 p_label_noise=0.0 lr_schedule=cyclic augm=False randaug=False seed=0 epoch=200.pth' 
python eval_interpolation_multiple_layers.py --dataset=cifar10 --model=vit_exp --model_width=512 --n_eval=1024 --eval_layer_str=weight --alpha_step=0.1 --exp_name='no_augm_small_vs_large_lr_random' --interpolate_random_direction --model_path1='models/2023-01-14 23:40:28.481 dataset=cifar10 model=vit_exp epochs=200 lr_max=0.005156 model_width=512 l2_reg=0.0 sam_rho=0.0 batch_size=128 frac_train=1 p_label_noise=0.0 lr_schedule=cyclic augm=False randaug=False seed=0 epoch=200.pth' --model_path2='models/2023-01-14 23:40:10.063 dataset=cifar10 model=vit_exp epochs=200 lr_max=0.155923 model_width=512 l2_reg=0.0 sam_rho=0.0 batch_size=128 frac_train=1 p_label_noise=0.0 lr_schedule=cyclic augm=False randaug=False seed=0 epoch=200.pth'
python eval_interpolation_multiple_layers.py --dataset=cifar10 --model=vit_exp --model_width=512 --n_eval=1024 --eval_layer_str=weight --alpha_step=0.1 --exp_name='no_augm_small_vs_large_lr_rev_random' --interpolate_random_direction --model_path1='models/2023-01-14 23:40:10.063 dataset=cifar10 model=vit_exp epochs=200 lr_max=0.155923 model_width=512 l2_reg=0.0 sam_rho=0.0 batch_size=128 frac_train=1 p_label_noise=0.0 lr_schedule=cyclic augm=False randaug=False seed=0 epoch=200.pth' --model_path2='models/2023-01-14 23:40:28.481 dataset=cifar10 model=vit_exp epochs=200 lr_max=0.005156 model_width=512 l2_reg=0.0 sam_rho=0.0 batch_size=128 frac_train=1 p_label_noise=0.0 lr_schedule=cyclic augm=False randaug=False seed=0 epoch=200.pth' 


# no augmentations, small LR: rho=0 vs. rho=0.1
python eval_interpolation_multiple_layers.py --dataset=cifar10 --model=vit_exp --model_width=512 --n_eval=1024 --eval_layer_str=weight --alpha_step=0.1 --exp_name='no_augm_rho0_vs_rho0.1' --model_path1='models/2023-01-14 23:40:28.481 dataset=cifar10 model=vit_exp epochs=200 lr_max=0.005156 model_width=512 l2_reg=0.0 sam_rho=0.0 batch_size=128 frac_train=1 p_label_noise=0.0 lr_schedule=cyclic augm=False randaug=False seed=0 epoch=200.pth' --model_path2='models/2023-01-14 23:50:36.906 dataset=cifar10 model=vit_exp epochs=200 lr_max=0.006417 model_width=512 l2_reg=0.0 sam_rho=0.1 batch_size=128 frac_train=1 p_label_noise=0.0 lr_schedule=cyclic augm=False randaug=False seed=0 epoch=200.pth'
python eval_interpolation_multiple_layers.py --dataset=cifar10 --model=vit_exp --model_width=512 --n_eval=1024 --eval_layer_str=weight --alpha_step=0.1 --exp_name='no_augm_rho0_vs_rho0.1_rev' --model_path1='models/2023-01-14 23:50:36.906 dataset=cifar10 model=vit_exp epochs=200 lr_max=0.006417 model_width=512 l2_reg=0.0 sam_rho=0.1 batch_size=128 frac_train=1 p_label_noise=0.0 lr_schedule=cyclic augm=False randaug=False seed=0 epoch=200.pth' --model_path2='models/2023-01-14 23:40:28.481 dataset=cifar10 model=vit_exp epochs=200 lr_max=0.005156 model_width=512 l2_reg=0.0 sam_rho=0.0 batch_size=128 frac_train=1 p_label_noise=0.0 lr_schedule=cyclic augm=False randaug=False seed=0 epoch=200.pth'
python eval_interpolation_multiple_layers.py --dataset=cifar10 --model=vit_exp --model_width=512 --n_eval=1024 --eval_layer_str=weight --alpha_step=0.1 --exp_name='no_augm_rho0_vs_rho0.1_random' --interpolate_random_direction --model_path1='models/2023-01-14 23:40:28.481 dataset=cifar10 model=vit_exp epochs=200 lr_max=0.005156 model_width=512 l2_reg=0.0 sam_rho=0.0 batch_size=128 frac_train=1 p_label_noise=0.0 lr_schedule=cyclic augm=False randaug=False seed=0 epoch=200.pth' --model_path2='models/2023-01-14 23:50:36.906 dataset=cifar10 model=vit_exp epochs=200 lr_max=0.006417 model_width=512 l2_reg=0.0 sam_rho=0.1 batch_size=128 frac_train=1 p_label_noise=0.0 lr_schedule=cyclic augm=False randaug=False seed=0 epoch=200.pth'
python eval_interpolation_multiple_layers.py --dataset=cifar10 --model=vit_exp --model_width=512 --n_eval=1024 --eval_layer_str=weight --alpha_step=0.1 --exp_name='no_augm_rho0_vs_rho0.1_rev_random' --interpolate_random_direction --model_path1='models/2023-01-14 23:50:36.906 dataset=cifar10 model=vit_exp epochs=200 lr_max=0.006417 model_width=512 l2_reg=0.0 sam_rho=0.1 batch_size=128 frac_train=1 p_label_noise=0.0 lr_schedule=cyclic augm=False randaug=False seed=0 epoch=200.pth' --model_path2='models/2023-01-14 23:40:28.481 dataset=cifar10 model=vit_exp epochs=200 lr_max=0.005156 model_width=512 l2_reg=0.0 sam_rho=0.0 batch_size=128 frac_train=1 p_label_noise=0.0 lr_schedule=cyclic augm=False randaug=False seed=0 epoch=200.pth'

