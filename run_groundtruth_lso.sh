python LSO_groundtruth_eval.py --reload_model ./weights/default-periodic-8.pth --dump_path ./dump/eval/snip-512-strogatz --eval_lso_on_pmlb True --pmlb_data_type gt --target_noise 0.0 --max_input_points 200 --lso_optimizer gwo --lso_pop_size 50 --lso_max_iteration 20 --lso_stop_r2 0.99 --beam_size 2

#python LSO_groundtruth_eval.py --reload_model ./weights/snip-e2e-sr.pth --dump_path ./dump/eval/snip-e2e-strogatz --eval_lso_on_pmlb True --pmlb_data_type gt --target_noise 0.0 --max_input_points 200 --lso_optimizer gwo --lso_pop_size 50 --lso_max_iteration 20 --lso_stop_r2 0.99 --beam_size 2
