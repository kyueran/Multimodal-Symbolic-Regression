import os
import torch
import numpy as np
import datetime
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from itertools import cycle
from tqdm import tqdm
import argparse
from parsers import get_parser
from pathlib import Path
import symbolicregression
from symbolicregression.envs import build_env
from symbolicregression.model import check_model_params, build_modules
from symbolicregression.utils import bool_flag, initialize_exp
from symbolicregression.slurm import init_signal_handler, init_distributed_mode
from symbolicregression.trainer import Trainer
from const_opt import *
import copy 
from alg_update import lso_update_pop
from collections import OrderedDict, defaultdict
import pandas as pd 
import warnings
import random
import nevergrad as ng
from scipy.optimize import differential_evolution
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  

import traceback

warnings.filterwarnings("ignore")  # Ignore all warnings


def reload_model(modules, path, requires_grad=False):
    """
    Reload a checkpoint if we find one.
    """
    if path is None:
        path = "checkpoint.pth"
    assert os.path.isfile(path)

    data = torch.load(path)

    # reload model parameters
    for k, v in modules.items():
        try:
            weights = data[k]
            v.load_state_dict(weights)
            print("load model successful")
        except RuntimeError:  # remove the 'module.'
            weights = {name.partition(".")[2]: v for name, v in data[k].items()}
            v.load_state_dict(weights)

        v.requires_grad = requires_grad

    return modules


def gen2eq(env, params, encoded_y, generations, sample_to_learn, stored_skeletons):
    dimension = sample_to_learn['x_to_fit'][0].shape[1]
    x_gt = sample_to_learn['x_to_fit'][0].reshape(-1,dimension) 
    y_gt = sample_to_learn['y_to_fit'][0].reshape(-1,1) 
    generations = generations.unsqueeze(-1)
    generations = generations.transpose(1, 2).cpu().tolist()
    generations_tree = [
        list(filter(lambda x: x is not None,
                [env.idx_to_infix(hyp[1:-1], is_float=False, str_array=False)
                    for hyp in generations[i]
                ],))for i in range(len(encoded_y))]
    ### 
    non_skeleton_tree = copy.deepcopy(generations_tree)
    skeleton_candidate, _ = env.generator.function_to_skeleton(non_skeleton_tree[0][0], constants_with_idx=False)
    if skeleton_candidate.infix() in stored_skeletons:
        return False, skeleton_candidate, None, None
    else:
        refined = refine(env, x_gt, y_gt, non_skeleton_tree[0], verbose=True)
        best_tree = list(filter(lambda gen: gen["refinement_type"]=='BFGS', refined))
        tree = best_tree[0]['predicted_tree'] #non_skeleton_tree[i][0] #

        numexpr_fn = env.simplifier.tree_to_numexpr_fn(tree) 

        y = numexpr_fn(x_gt)[:,0].reshape(-1,1)
        r2_zero = max(r2_score(y_gt, y), 0)
       
        eq_outputs = (True, skeleton_candidate, tree, r2_zero)
        return eq_outputs


def create_population(env,params,model, sample_to_learn, encoded_y):
    encoded_pop = encoded_y
    for i in range(15): # 15
        aug_sample = copy.deepcopy(sample_to_learn)
        seq_len = len(sample_to_learn['x_to_fit'][0])
        all_indices = list(range(seq_len))
        if seq_len > 10*params.max_input_points:
            random_indices = random.sample(all_indices, params.max_input_points) ### Sample a bag of 200 points from the whole training set
        else:
            random_indices = random.sample(all_indices, seq_len//2) 

        aug_sample['X_scaled_to_fit'][0] = np.array([sample_to_learn['X_scaled_to_fit'][0][i] for i in random_indices])
        aug_sample['Y_scaled_to_fit'][0] = np.array([sample_to_learn['Y_scaled_to_fit'][0][i] for i in random_indices])
        outputs = model(aug_sample,max_len = params.max_target_len)
        encoded_aug, generations, _ = outputs
        encoded_pop = torch.cat((encoded_pop, encoded_aug), dim=0)

    for i in range(10): 
        sigma = 0.01*(i/5+1)  
        aug_sample = copy.deepcopy(sample_to_learn)
        aug_sample['Y_scaled_to_fit'][0]=sample_to_learn['Y_scaled_to_fit'][0] + np.random.normal(0,sigma*np.std(sample_to_learn['Y_scaled_to_fit'][0]), sample_to_learn['Y_scaled_to_fit'][0].shape)
        seq_len = len(sample_to_learn['x_to_fit'][0])
        all_indices = list(range(seq_len))
        if seq_len > 10*params.max_input_points:
            random_indices = random.sample(all_indices, params.max_input_points) ### Sample a bag of 200 points from the whole training set
        else:
            random_indices = random.sample(all_indices, seq_len//2) 

        aug_sample['X_scaled_to_fit'][0] = np.array([sample_to_learn['X_scaled_to_fit'][0][i] for i in random_indices])
        aug_sample['Y_scaled_to_fit'][0] = np.array([sample_to_learn['Y_scaled_to_fit'][0][i] for i in random_indices])
        outputs = model(aug_sample,max_len = params.max_target_len)
        encoded_aug, generations, _ = outputs
        encoded_pop = torch.cat((encoded_pop, encoded_aug), dim=0)

    for i in range(24): 
        r = (2*torch.rand(1)).to(params.device) ### Original : 1.5 , # 2 for black-box
        encoded_aug= encoded_y + r*torch.randn(encoded_y.shape).to(params.device)
        encoded_pop = torch.cat((encoded_pop, encoded_aug), dim=0)

    return encoded_pop


def lso_fit(sample_to_learn, env, params, model,batch_results,bag_number):
    sub_sample = copy.deepcopy(sample_to_learn)

    seq_len = len(sample_to_learn['x_to_fit'][0])
    all_indices = list(range(seq_len))
    if seq_len >= params.max_input_points:
        random_indices = random.sample(all_indices, params.max_input_points) ### Sample a bag of 200 points from the whole training set
        sub_sample['X_scaled_to_fit'][0] = np.array([sample_to_learn['X_scaled_to_fit'][0][i] for i in random_indices])
        sub_sample['Y_scaled_to_fit'][0] = np.array([sample_to_learn['Y_scaled_to_fit'][0][i] for i in random_indices])
        sub_sample['x_to_fit'][0] = np.array([sample_to_learn['x_to_fit'][0][i] for i in random_indices])
        sub_sample['y_to_fit'][0] = np.array([sample_to_learn['y_to_fit'][0][i] for i in random_indices])

        seq_len = len(sample_to_learn['x_to_predict'][0])
        all_indices = list(range(seq_len))
        if seq_len >= params.max_input_points:
            random_indices = random.sample(all_indices, params.max_input_points) 
            sub_sample['x_to_predict'][0] = np.array([sample_to_learn['x_to_predict'][0][i] for i in random_indices])
            sub_sample['y_to_predict'][0] = np.array([sample_to_learn['y_to_predict'][0][i] for i in random_indices])
        else:
            sub_sample['x_to_predict'][0] = sample_to_learn['x_to_predict'][0]
            sub_sample['y_to_predict'][0] = sample_to_learn['y_to_predict'][0]
    else:
        sub_sample = sample_to_learn
    
    outputs = model(sub_sample,max_len = params.max_target_len)
    encoded_y, generations, gen_len = outputs
    #logger.info("encoded_y: %s", encoded_y)
    #logger.info("generations: %s", generations)

    max_r2 = 0
    best_eq = None

    start_time = time.time()
    encoded_pop = create_population(env, params, model, sample_to_learn, encoded_y)
    pop = encoded_pop

    pop_list = []
    r2_pop_list = []
    max_iteration = params.lso_max_iteration

    ### LSO elites
    Alpha_pos, Beta_pos, Delta_pos = np.zeros(encoded_pop.shape[1]), np.zeros(encoded_pop.shape[1]), np.zeros(encoded_pop.shape[1])
    Alpha_score, Beta_score, Delta_score = np.inf, np.inf, np.inf

    count_repeated = 0

    stored_skeletons = []
    for t in range(max_iteration):
        r2_pop = np.zeros(len(encoded_pop))
        start_time_fwd = time.time()
        generations_pop = model.generate_from_latent_sampling(pop)
        print("forward pass with b=2, pop=50 : ", time.time() - start_time_fwd )
        for i in range(len(pop)):
            highest_r2_in_beam = 0
            best_eq_in_beam = 'NaN'
            for b in range(params.beam_size):
                try:
                    eq_outputs = gen2eq(env, params, pop[i].reshape(1,-1), generations_pop[params.beam_size*i+b].reshape(1,-1), sample_to_learn , stored_skeletons)
                    success, skeleton_candidate, tree, r2_zero = eq_outputs
                    if success == True:
                        stored_skeletons.append(skeleton_candidate.infix())
                        if r2_zero >= highest_r2_in_beam:
                            highest_r2_in_beam = r2_zero
                            best_eq_in_beam = tree
                    if success == False:
                        count_repeated += 1
                        # print(f"Repeated Skeleton {count_repeated}")

                except:
                    pass
            
            r2_pop[i] = highest_r2_in_beam

            if highest_r2_in_beam > max_r2:
                max_r2 = highest_r2_in_beam
                best_eq = best_eq_in_beam
            if highest_r2_in_beam > params.lso_stop_r2:
                print(f"Finish at population: {i}")
                break

        pop_list.append(pop.cpu().numpy())
        r2_pop_list.append(r2_pop)

        ### update Pop 
        ## mse is actually Normalized MSE
        objective = -r2_pop #+ (1e-6)*mse_pop
        pop , elites = lso_update_pop(
            pop, objective, t, max_iteration, 
            Alpha_score, Beta_score, Delta_score ,
            Alpha_pos, Beta_pos, Delta_pos,
            lb= torch.min(pop).item(), ub= torch.max(pop).item()
            )

        pop = torch.tensor(pop,  dtype=encoded_pop.dtype).cuda()
        Alpha_score, Beta_score, Delta_score , Alpha_pos, Beta_pos, Delta_pos = elites

        print(f"Max R2 of sample at iteration {t} is {max_r2}")

        if highest_r2_in_beam > params.lso_stop_r2:
            print(f"Finish at iteration: {t}")
            break

    
    optimization_duration = time.time() - start_time

    ###
    dimension = sample_to_learn['x_to_fit'][0].shape[1]
    x_gt_pred = sample_to_learn['x_to_predict'][0].reshape(-1,dimension) 
    y_gt_pred = sample_to_learn['y_to_predict'][0].reshape(-1,1) 

    try:
        numexpr_fn = env.simplifier.tree_to_numexpr_fn(best_eq) 
        y_pred = numexpr_fn(x_gt_pred)[:,0].reshape(-1,1)
        complexity = len(best_eq.prefix().split(','))

        batch_results["mse_test"].extend([mean_squared_error(y_gt_pred, y_pred)])
        batch_results["mae_test"].extend([mean_absolute_error(y_gt_pred, y_pred)])
        batch_results["r2_test"].extend([r2_score(y_gt_pred, y_pred)])
        batch_results["r2_zero_test"].extend([max(batch_results["r2_test"][0], 0)])

        batch_results["symbolic_model"].extend([best_eq])
        batch_results["model_size"].extend([complexity])
        batch_results["training time (s)"].extend([optimization_duration])
        batch_results["training time (hr)"].extend([optimization_duration / 3600])

    except Exception as e:
        # Log the exception message and type
        print(f"An error occurred: {e}")

    return batch_results
    


class LSOFitNeverGrad():
    def __init__(self, env, params, model, sample_to_learn, batch_results, bag_number, logger):
        self.env = env
        self.params = params
        self.model = model
        self.sample_to_learn = sample_to_learn
        self.batch_results = batch_results
        self.bag_number = bag_number
        self.logger = logger

        self.max_r2 = 0
        self.min_mse = 'NaN'
        self.best_eq = 'NaN'
        self.best_y = 'NaN'
        self.best_y_pred = 'NaN'

    def fit_func(self):
        ### GET the first encoded y from a max input seq len to the SNIP Encoder
        sub_sample = copy.deepcopy(self.sample_to_learn)
        seq_len = len(self.sample_to_learn['x_to_fit'][0])
        all_indices = list(range(seq_len))
        if seq_len >= self.params.max_input_points:
            random_indices = random.sample(all_indices, self.params.max_input_points) ### Sample a bag of 200 points from the whole training set
            sub_sample['X_scaled_to_fit'][0] = np.array([self.sample_to_learn['X_scaled_to_fit'][0][i] for i in random_indices])
            sub_sample['Y_scaled_to_fit'][0] = np.array([self.sample_to_learn['Y_scaled_to_fit'][0][i] for i in random_indices])
            sub_sample['x_to_fit'][0] = np.array([self.sample_to_learn['x_to_fit'][0][i] for i in random_indices])
            sub_sample['y_to_fit'][0] = np.array([self.sample_to_learn['y_to_fit'][0][i] for i in random_indices])

            seq_len = len(self.sample_to_learn['x_to_predict'][0])
            all_indices = list(range(seq_len))
            if seq_len >= self.params.max_input_points:
                random_indices = random.sample(all_indices, self.params.max_input_points) 
                sub_sample['x_to_predict'][0] = np.array([self.sample_to_learn['x_to_predict'][0][i] for i in random_indices])
                sub_sample['y_to_predict'][0] = np.array([self.sample_to_learn['y_to_predict'][0][i] for i in random_indices])
            else:
                sub_sample['x_to_predict'][0] = self.sample_to_learn['x_to_predict'][0]
                sub_sample['y_to_predict'][0] = self.sample_to_learn['y_to_predict'][0]
        else:
            sub_sample = self.sample_to_learn
        
        outputs = self.model(sub_sample,max_len = self.params.max_target_len)
        encoded_y, generations, gen_len = outputs
        try: 
            eq_outputs = self.gen2eq(encoded_y, generations, self.logger)
            skeleton_candidate, tree, complexity, y, y_pred ,mse_fit, mse_pred, results_fit, results_predict = eq_outputs
            max_r2 = results_fit['r2_zero'][0]
            self.batch_results["direct_predicted_tree"].extend([tree])
            self.batch_results["complexity_gt_tree"].extend([complexity])
            self.batch_results["direct_fit_mse"].extend([mse_fit])
            self.batch_results["direct_pred_mse"].extend([mse_pred])
            for k, v in results_fit.items():
                self.batch_results[k + "_direct_fit"].extend(v)
            del results_fit
            for k, v in results_predict.items():
                self.batch_results[k + "_direct_predict"].extend(v)
            del results_predict            
        except:
            tree = 'NaN'
            complexity = 'NaN'
            mse_fit = float('inf')
            mse_pred = 'NaN'
            max_r2 = 0
            y= 'NaN'
            y_pred = 'NaN'

        self.min_mse = mse_fit
        self.max_r2 = max_r2
        self.best_eq = tree
        self.best_y = y
        self.best_y_pred = y_pred

        start_time = time.time()

        encoded_pop = self.create_population(encoded_y)
        pop = encoded_pop.cpu().numpy()

        # Get bounds from initial population
        lb = np.min(pop)  
        ub = np.max(pop)

        # instrum = ng.p.Instrumentation(ng.p.Array(shape=(pop.shape[1],)).set_bounds(lower=lb, upper=ub))
        instrum = ng.p.Instrumentation(ng.p.Array(shape=(pop.shape[1],)))
        # Create  bounds
        if self.params.lso_optimizer == "cma-es":
            # optimizer = ng.optimizers.CMA(parametrization=ng.p.Array(init=pop[0]), lower_bound=lb, upper_bound=ub)
            optimizer = ng.optimizers.CMA(parametrization=instrum, budget=100, num_workers=1)

        ### NGOpt
        if self.params.lso_optimizer == "ngopt":
            optimizer = ng.optimizers.NGOpt(parametrization=instrum, budget=100, num_workers=1)
            # optimizer = ng.optimizers.NGOpt(parametrization=instrum, budget=self.params.lso_max_iteration, num_workers=1)
            

        ### Two Points DE
        if self.params.lso_optimizer == "twopoints-de":
            optimizer = ng.optimizers.TwoPointsDE(parametrization=instrum, budget=100, num_workers=1)

        ### RandomSearch
        if self.params.lso_optimizer == "randomsearch":
            optimizer = ng.optimizers.RandomSearch(parametrization=instrum, budget=100, num_workers=1)

        ### initial evaluation
        r2_pop = self.evaluate_pop(pop)
        fitness = r2_pop

        for i in range(len(pop)):
            # candidate = optimizer.parametrization.spawn_child(pop[i])
            optimizer.suggest(pop[i])
            candidate = optimizer.ask()
            optimizer.tell(candidate, fitness[i])

        print("Max Fitness initially: ", - np.min(fitness))

        recommendation = optimizer.minimize(self.evaluate_agent) 
        r2_final = self.evaluate_agent(recommendation[0].value[0])
        print("Recommended point fitness: ", -r2_final)

        optimization_duration = time.time()- start_time
        self.batch_results["final_predicted_tree"].extend([self.best_eq])
        self.batch_results["final_fit_mse"].extend([self.min_mse])

        dimension = self.sample_to_learn['x_to_fit'][0].shape[1]
        x_gt = self.sample_to_learn['x_to_fit'][0].reshape(-1,dimension) 
        y_gt = self.sample_to_learn['y_to_fit'][0].reshape(-1,1) 
        x_gt_pred = self.sample_to_learn['x_to_predict'][0].reshape(-1,dimension) 
        y_gt_pred = self.sample_to_learn['y_to_predict'][0].reshape(-1,1) 

        ## modified
        try:
            numexpr_fn = self.env.simplifier.tree_to_numexpr_fn(self.best_eq) 

            y = numexpr_fn(x_gt)[:,0].reshape(-1,1)
            y_pred = numexpr_fn(x_gt_pred)[:,0].reshape(-1,1)
            if np.isnan(y).any():
                mse_fit = float('inf')
            else:
                mse_fit = np.mean((y-y_gt)**2) / ( np.mean(y_gt**2) + 1e-10)
                mse_pred = np.mean((y_pred - y_gt_pred)**2) / ( np.mean(y_gt**2) + 1e-10)

            complexity = len(self.best_eq.prefix().split(','))

            results_fit = compute_metrics(
                {
                    "true": [y_gt],
                    "predicted": [y],
                    # "tree": [tree_gt],
                    "predicted_tree": [self.best_eq],
                },
                metrics=self.params.validation_metrics,
            )

            results_predict = compute_metrics(
                {
                    "true": [y_gt_pred],
                    "predicted": [y_pred],
                    # "tree": [tree_gt],
                    "predicted_tree": [self.best_eq],
                },
                metrics=self.params.validation_metrics,
            )
            for k, v in results_fit.items():
                self.batch_results[k + "_final_fit"].extend(v)
            del results_fit

            for k, v in results_predict.items():
                self.batch_results[k + "_final_predict"].extend(v)
            del results_predict

            self.batch_results["time"].extend([optimization_duration])
        except:
            pass

        return self.batch_results

    def evaluate_pop(self, pop):
        pop = torch.tensor(np.array(pop), dtype=torch.float32).cuda()
        generations_pop = self.model.generate_from_latent_sampling(pop)
        mse_pop = np.zeros(len(pop))
        r2_pop = np.zeros(len(pop))
        cand_r2 = np.zeros(len(pop))

        for i in range(len(pop)):
            highest_r2_in_beam = 0
            mse_in_beam = float('inf') 
            best_eq_in_beam = 'NaN'
            best_y_in_beam = 'NaN'
            best_y_pred_in_beam = 'NaN'
            for b in range(self.params.beam_size):
                try:
                    eq_outputs = self.gen2eq(pop[i].reshape(1,-1), generations_pop[self.params.beam_size*i+b].reshape(1,-1), self.logger)
                    skeleton_candidate, tree, complexity, y, y_pred, mse_fit, mse_pred, results_fit, results_predict = eq_outputs
                    cand_r2[i] = results_fit['r2_zero'][0]

                    if results_fit['r2_zero'][0] >= highest_r2_in_beam:
                        highest_r2_in_beam = results_fit['r2_zero'][0] 
                        mse_in_beam = mse_fit
                        best_eq_in_beam = tree
                        best_y_in_beam = y 
                        best_y_pred_in_beam = y_pred 
                except:
                    #traceback.print_exc()
                    pass

            r2_pop[i] = highest_r2_in_beam
            mse_pop[i] = mse_in_beam
        
            if highest_r2_in_beam > self.max_r2:
                self.max_r2 = highest_r2_in_beam
                self.min_mse = mse_in_beam
                self.best_eq = best_eq_in_beam
                self.best_y = best_y_in_beam
                self.best_y_pred = best_y_pred_in_beam

        self.logger.info(f"cand_r2: {cand_r2}, min: {np.min(cand_r2)}, max: {np.max(cand_r2)}")
        return -r2_pop

    def evaluate_agent(self, agent):
        pop = agent.reshape(1,-1)
        pop = torch.tensor(np.array(pop), dtype=torch.float32).cuda()
        generations_pop = self.model.generate_from_latent_sampling(pop)
        mse_pop = np.zeros(len(pop))
        r2_pop = np.zeros(len(pop))
        for i in range(len(pop)):
            highest_r2_in_beam = 0
            mse_in_beam = float('inf') 
            best_eq_in_beam = 'NaN'
            best_y_in_beam = 'NaN'
            best_y_pred_in_beam = 'NaN'
            for b in range(self.params.beam_size):
                try:
                    eq_outputs = self.gen2eq(pop[i].reshape(1,-1), generations_pop[self.params.beam_size*i+b].reshape(1,-1), self.logger)
                    skeleton_candidate, tree, complexity, y, y_pred, mse_fit, mse_pred, results_fit, results_predict = eq_outputs
                    if results_fit['r2_zero'][0] >= highest_r2_in_beam:
                        highest_r2_in_beam = results_fit['r2_zero'][0] 
                        mse_in_beam = mse_fit
                        best_eq_in_beam = tree
                        best_y_in_beam = y 
                        best_y_pred_in_beam = y_pred 
                except:
                    pass

            r2_pop[i] = highest_r2_in_beam
            mse_pop[i] = mse_in_beam
        
            if highest_r2_in_beam > self.max_r2:
                self.max_r2 = highest_r2_in_beam
                self.min_mse = mse_in_beam
                self.best_eq = best_eq_in_beam
                self.best_y = best_y_in_beam
                self.best_y_pred = best_y_pred_in_beam

        return -r2_pop

    def gen2eq(self, encoded_y, generations, logger):
        dimension = self.sample_to_learn['x_to_fit'][0].shape[1]
        x_gt = self.sample_to_learn['x_to_fit'][0].reshape(-1,dimension) 
        y_gt = self.sample_to_learn['y_to_fit'][0].reshape(-1,1) 
        x_gt_pred = self.sample_to_learn['x_to_predict'][0].reshape(-1,dimension) 
        y_gt_pred = self.sample_to_learn['y_to_predict'][0].reshape(-1,1) 
        generations = generations.unsqueeze(-1)
        generations = generations.transpose(1, 2).cpu().tolist()
        generations_tree = [
            list(filter(lambda x: x is not None,
                    [self.env.idx_to_infix(hyp[1:-1], is_float=False, str_array=False)
                        for hyp in generations[i]
                    ],))for i in range(len(encoded_y))]
        ### 
        non_skeleton_tree = copy.deepcopy(generations_tree)
        print(non_skeleton_tree[0][0])

        skeleton_candidate, _ = self.env.generator.function_to_skeleton(non_skeleton_tree[0][0], constants_with_idx=False)
        refined = refine(self.env, x_gt, y_gt, non_skeleton_tree[0], logger=logger, verbose=True)
        best_tree = list(filter(lambda gen: gen["refinement_type"]=='BFGS', refined))
        tree = best_tree[0]['predicted_tree'] #non_skeleton_tree[i][0] #

        
        numexpr_fn = self.env.simplifier.tree_to_numexpr_fn(tree) 
        y = numexpr_fn(x_gt)[:,0].reshape(-1,1)
        y_pred = numexpr_fn(x_gt_pred)[:,0].reshape(-1,1)
        if np.isnan(y).any():
            mse_fit = float('inf')
        else:
            mse_fit = np.mean((y-y_gt)**2) / ( np.mean(y_gt**2) + 1e-10)
            mse_pred = np.mean((y_pred - y_gt_pred)**2) / ( np.mean(y_gt**2) + 1e-10)

        complexity = len(tree.prefix().split(','))

        results_fit = compute_metrics(
            {
                "true": [y_gt],
                "predicted": [y],
                # "tree": [tree_gt],
                "predicted_tree": [tree],
            },
            metrics=self.params.validation_metrics,
        )

        results_predict = compute_metrics(
            {
                "true": [y_gt_pred],
                "predicted": [y_pred],
                # "tree": [tree_gt],
                "predicted_tree": [tree],
            },
            metrics=self.params.validation_metrics,
        )

        eq_outputs = (skeleton_candidate, tree, complexity, y, y_pred , mse_fit, mse_pred, results_fit, results_predict)
        return eq_outputs

    def create_population(self, encoded_y):
        encoded_pop = encoded_y
        ### 1) Create Part of population by sampling subsections of y with size of params max seq len: 
        for i in range(15): # 15
            aug_sample = copy.deepcopy(self.sample_to_learn)
            seq_len = len(self.sample_to_learn['x_to_fit'][0])
            all_indices = list(range(seq_len))
            if seq_len > 10*self.params.max_input_points:
                random_indices = random.sample(all_indices, self.params.max_input_points) ### Sample a bag of 200 points from the whole training set
            else:
                random_indices = random.sample(all_indices, seq_len//2) 

            aug_sample['X_scaled_to_fit'][0] = np.array([self.sample_to_learn['X_scaled_to_fit'][0][i] for i in random_indices])
            aug_sample['Y_scaled_to_fit'][0] = np.array([self.sample_to_learn['Y_scaled_to_fit'][0][i] for i in random_indices])
            outputs = self.model(aug_sample,max_len = self.params.max_target_len)
            encoded_aug, generations, _ = outputs
            encoded_pop = torch.cat((encoded_pop, encoded_aug), dim=0)

        ### 3) Create Part of population by adding noise to the input y: 
        for i in range(10): # 10
            sigma = 0.01*(i/10+1)  # sigma = 0.01*(i/10+1) , /5 for blackbox
            aug_sample = copy.deepcopy(self.sample_to_learn)
            aug_sample['Y_scaled_to_fit'][0]=self.sample_to_learn['Y_scaled_to_fit'][0] + np.random.normal(0,sigma*np.std(self.sample_to_learn['Y_scaled_to_fit'][0]), self.sample_to_learn['Y_scaled_to_fit'][0].shape)
            ## sample max seq len points to get encoded y
            seq_len = len(self.sample_to_learn['x_to_fit'][0])
            all_indices = list(range(seq_len))
            if seq_len > 10*self.params.max_input_points:
                random_indices = random.sample(all_indices, self.params.max_input_points) ### Sample a bag of 200 points from the whole training set
            else:
                random_indices = random.sample(all_indices, seq_len//2) 

            aug_sample['X_scaled_to_fit'][0] = np.array([self.sample_to_learn['X_scaled_to_fit'][0][i] for i in random_indices])
            aug_sample['Y_scaled_to_fit'][0] = np.array([self.sample_to_learn['Y_scaled_to_fit'][0][i] for i in random_indices])
            outputs = self.model(aug_sample,max_len = self.params.max_target_len)
            encoded_aug, generations, _ = outputs
            encoded_pop = torch.cat((encoded_pop, encoded_aug), dim=0)

        ### 4) Create Part of population by adding noise in latent space (z_y):
        for i in range(24): # 24
            r = (0.1*torch.rand(1)).to(self.params.device) ### Original : 1.5 , # 2 for black-box
            encoded_aug= encoded_y + r*torch.randn(encoded_y.shape).to(self.params.device)
            encoded_pop = torch.cat((encoded_pop, encoded_aug), dim=0)

        return encoded_pop

import logging

# Suppress INFO logs from ax.service.ax_client
logging.getLogger("ax").setLevel(logging.WARNING)

from typing import Optional

from botorch.models.gpytorch import GPyTorchModel
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.models import ExactGP
from torch import Tensor

from ax.models.torch.botorch_modular.model import BoTorchModel
from ax.models.torch.botorch_modular.surrogate import Surrogate

from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models

from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties

import botorch
botorch.settings.debug(True)

class LSOFitBoTorch(LSOFitNeverGrad):
    def fit_func(self):
        class SimpleCustomGP(ExactGP, GPyTorchModel):
            _num_outputs = 1  # to inform GPyTorchModel API

            def __init__(self, train_X, train_Y, train_Yvar: Optional[Tensor] = None):
                # NOTE: This ignores train_Yvar and uses inferred noise instead.
                # squeeze output dim before passing train_Y to ExactGP
                super().__init__(train_X, train_Y.squeeze(-1), GaussianLikelihood())
                self.mean_module = ConstantMean()
                self.covar_module = ScaleKernel(
                    base_kernel=RBFKernel(ard_num_dims=train_X.shape[-1]),
                )

                self.to(train_X)  # make sure we're on the right device/dtype

            def forward(self, x):
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return MultivariateNormal(mean_x, covar_x)
        
        random.seed(9)
        sub_sample = copy.deepcopy(self.sample_to_learn)
        seq_len = len(self.sample_to_learn['x_to_fit'][0])
        all_indices = list(range(seq_len))
        if seq_len >= self.params.max_input_points:
            random_indices = random.sample(all_indices, self.params.max_input_points)
            sub_sample['X_scaled_to_fit'][0] = np.array([self.sample_to_learn['X_scaled_to_fit'][0][i] for i in random_indices])
            sub_sample['Y_scaled_to_fit'][0] = np.array([self.sample_to_learn['Y_scaled_to_fit'][0][i] for i in random_indices])
            sub_sample['x_to_fit'][0] = np.array([self.sample_to_learn['x_to_fit'][0][i] for i in random_indices])
            sub_sample['y_to_fit'][0] = np.array([self.sample_to_learn['y_to_fit'][0][i] for i in random_indices])

            seq_len = len(self.sample_to_learn['x_to_predict'][0])
            all_indices = list(range(seq_len))
            if seq_len >= self.params.max_input_points:
                random_indices = random.sample(all_indices, self.params.max_input_points)
                sub_sample['x_to_predict'][0] = np.array([self.sample_to_learn['x_to_predict'][0][i] for i in random_indices])
                sub_sample['y_to_predict'][0] = np.array([self.sample_to_learn['y_to_predict'][0][i] for i in random_indices])
            else:
                sub_sample['x_to_predict'][0] = self.sample_to_learn['x_to_predict'][0]
                sub_sample['y_to_predict'][0] = self.sample_to_learn['y_to_predict'][0]
        else:
            sub_sample = self.sample_to_learn

        outputs = self.model(sub_sample, max_len=self.params.max_target_len)
        encoded_y, generations, gen_len = outputs
        #self.logger.info("encoded_y: %s", encoded_y)
        #self.logger.info("generations: %s", generations)
        try:
            eq_outputs = self.gen2eq(encoded_y, generations, self.logger)
            skeleton_candidate, tree, complexity, y, y_pred, mse_fit, mse_pred, results_fit, results_predict = eq_outputs
            #self.logger.info("BO Direct Tree: %s", tree)
            max_r2 = results_fit['r2_zero'][0]
            self.batch_results["direct_predicted_tree"].extend([tree])
            self.batch_results["complexity_gt_tree"].extend([complexity])
            self.batch_results["direct_fit_mse"].extend([mse_fit])
            self.batch_results["direct_pred_mse"].extend([mse_pred])
            for k, v in results_fit.items():
                self.batch_results[k + "_direct_fit"].extend(v)
            del results_fit
            for k, v in results_predict.items():
                self.batch_results[k + "_direct_predict"].extend(v)
            del results_predict
        except:
            tree = 'NaN'
            complexity = 'NaN'
            mse_fit = float('inf')
            mse_pred = 'NaN'
            max_r2 = 0
            y = 'NaN'
            y_pred = 'NaN'

        self.min_mse = mse_fit
        self.max_r2 = max_r2
        self.best_eq = tree
        self.best_y = y
        self.best_y_pred = y_pred

        start_time = time.time()

        encoded_pop = self.create_population(encoded_y)
        pop = encoded_pop.detach().cpu().numpy()

        r2_pop = self.evaluate_pop(pop)
        fitness = r2_pop

        ax_model = BoTorchModel(
            surrogate=Surrogate(
                # The model class to use
                botorch_model_class=SimpleCustomGP,
                # Optional, MLL class with which to optimize model parameters
                # mll_class=ExactMarginalLogLikelihood,
                # Optional, dictionary of keyword arguments to model constructor
                # model_options={}
            ),
            # Optional, acquisition function class to use - see custom acquisition tutorial
            # botorch_acqf_class=qExpectedImprovement,
        )

        gs = GenerationStrategy(
            steps=[
                # Quasi-random initialization step
                GenerationStep(
                    model=Models.SOBOL,
                    num_trials=100,  # How many trials should be produced from this generation step
                ),
                # Bayesian optimization step using the custom acquisition function
                GenerationStep(
                    model=Models.BOTORCH_MODULAR,
                    num_trials=-1,  # No limitation on how many trials should be produced from this step
                    # For `BOTORCH_MODULAR`, we pass in kwargs to specify what surrogate or acquisition function to use.
                    model_kwargs={
                        "surrogate": Surrogate(SimpleCustomGP),
                    },
                ),
            ]
        )

        ax_client = AxClient(generation_strategy=gs)
        # Setup the experiment
        ax_client.create_experiment(
            name="optimal_equation_experiment",
            parameters = [
                {
                    "name": f"param_{i}",
                    "type": "range",
                    #"bounds": [float(-1.5), float(1.5)],
                    "bounds": [float(np.min(pop[:, i])), float(np.max(pop[:, i]))],
                }
                for i in range(pop.shape[1])  # Loop over each column in the pop matrix
            ],
            objectives={
                "neg_r2": ObjectiveProperties(minimize=True),
            },
        )
        # Setup a function to evaluate the trials
        neg_r2 = self.evaluate_agent


        def evaluate(parameters):
            # Extract all 512 parameters
            agent = torch.tensor([[parameters.get(f"param_{i}") for i in range(pop.shape[1])]], dtype=torch.float32)

            negative_r2_score = self.evaluate_agent(agent)
            
            # Return the result as a dictionary with the objective name
            return {"neg_r2": (negative_r2_score.item(), float("nan"))}
        
        with torch.enable_grad():
            NUM_EVALS = 2000
            for i in range(NUM_EVALS):
                try:
                    parameters, trial_index = ax_client.get_next_trial()
                    raw_data = evaluate(parameters)
                    if raw_data["neg_r2"][0] < -self.params.lso_stop_r2:
                        self.logger.info(f"Stopping early as neg_r2 is less than -0.99: {raw_data['neg_r2'][0]}")
                        ax_client.complete_trial(trial_index=trial_index, raw_data=raw_data)
                        break
                    ax_client.complete_trial(trial_index=trial_index, raw_data=raw_data)
                except Exception as e:
                    self.logger.error(f"An error occurred during trial {i}: {str(e)}")
                    break
        
            parameters, values = ax_client.get_best_parameters()
        #print(f"Best parameters: {parameters}")

        optimization_duration = time.time() - start_time
        self.batch_results["final_predicted_tree"].extend([self.best_eq])
        self.batch_results["final_fit_mse"].extend([self.min_mse])

        dimension = self.sample_to_learn['x_to_fit'][0].shape[1]
        x_gt = self.sample_to_learn['x_to_fit'][0].reshape(-1,dimension) 
        y_gt = self.sample_to_learn['y_to_fit'][0].reshape(-1,1) 
        x_gt_pred = self.sample_to_learn['x_to_predict'][0].reshape(-1,dimension) 
        y_gt_pred = self.sample_to_learn['y_to_predict'][0].reshape(-1,1) 

        ## modified
        try:
            numexpr_fn = self.env.simplifier.tree_to_numexpr_fn(self.best_eq) 

            y = numexpr_fn(x_gt)[:,0].reshape(-1,1)
            y_pred = numexpr_fn(x_gt_pred)[:,0].reshape(-1,1)
            if np.isnan(y).any():
                mse_fit = float('inf')
            else:
                mse_fit = np.mean((y-y_gt)**2) / ( np.mean(y_gt**2) + 1e-10)
                mse_pred = np.mean((y_pred - y_gt_pred)**2) / ( np.mean(y_gt**2) + 1e-10)

            complexity = len(self.best_eq.prefix().split(','))

            results_fit = compute_metrics(
                {
                    "true": [y_gt],
                    "predicted": [y],
                    # "tree": [tree_gt],
                    "predicted_tree": [self.best_eq],
                },
                metrics=self.params.validation_metrics,
            )

            results_predict = compute_metrics(
                {
                    "true": [y_gt_pred],
                    "predicted": [y_pred],
                    # "tree": [tree_gt],
                    "predicted_tree": [self.best_eq],
                },
                metrics=self.params.validation_metrics,
            )
            for k, v in results_fit.items():
                self.batch_results[k + "_final_fit"].extend(v)
            del results_fit

            for k, v in results_predict.items():
                self.batch_results[k + "_final_predict"].extend(v)
            del results_predict

            self.batch_results["time"].extend([optimization_duration])
        except:
            pass

        return self.batch_results