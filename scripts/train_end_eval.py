import argparse
import os.path
import pickle
import time
from datetime import datetime

import wandb

import pandas as pd
from scipy.stats import spearmanr, kendalltau
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

from graf_nas.features.config import load_from_config
from graf_nas.graf import create_dataset, GRAF
from graf_nas.sample import sampling_strategies
from graf_nas.search_space import searchspace_classes, dataset_api_maps, DARTS
from naslib.utils import get_dataset_api


zcps = ['tenas_cond',
 'tenas_max',
 'tenas_min',
 'tenas_coef',
 'tenas_entropy',
 'tenas_skew',
 'tenas_kurtosis',
 'tenas_kstest',
 'tenas_cramervonmises',
 'tenas_chisquare',
 'tenas_wasserstein_distance',
 'tenas_energy_distance',
 'tenas_prob_cond',
 'tenas_prob_max',
 'tenas_prob_min',
 'tenas_prob_coef',
 'tenas_prob_entropy',
 'tenas_prob_skew',
 'tenas_prob_kurtosis',
 'tenas_prob_kstest',
 'tenas_prob_cramervonmises',
 'tenas_prob_chisquare',
 'tenas_prob_wasserstein_distance',
 'tenas_prob_energy_distance',
 'fisher_cond',
 'fisher_max',
 'fisher_min',
 'fisher_coef',
 'fisher_entropy',
 'fisher_skew',
 'fisher_kurtosis',
 'fisher_kstest',
 'fisher_cramervonmises',
 'fisher_chisquare',
 'fisher_wasserstein_distance',
 'fisher_energy_distance',
 'fisher_prob_cond',
 'fisher_prob_max',
 'fisher_prob_min',
 'fisher_prob_coef',
 'fisher_prob_entropy',
 'fisher_prob_skew',
 'fisher_prob_kurtosis',
 'fisher_prob_kstest',
 'fisher_prob_cramervonmises',
 'fisher_prob_chisquare',
 'fisher_prob_wasserstein_distance',
 'fisher_prob_energy_distance',
 'epe_nas',
 'fisher',
 'flops',
 'grad_norm',
 'grasp',
 'jacov',
 'l2_norm',
 'nwot',
 'params',
 'plain',
 'snip',
 'synflow',
 'zen']

#kwargs = {'cluster_method': 'KMeans', 'weighted': False, 'replace': False, 'n_clusters': 32}
kwargs = {'cluster_method': 'hier', 'weighted': True, 'replace': True, 'n_clusters': 100}
#kwargs = {'cluster_method': 'DBSCAN', 'weighted': False, 'replace': False, 'eps': 10}

def get_train_test_splits(feature_dataset, y, train_size, test_size, seed, strategy):
    train_X, train_y = sampling_strategies[strategy](feature_dataset, y, train_size, seed, **kwargs)
    test_X, test_y = sampling_strategies['random'](feature_dataset, y, test_size, seed + 1)
    return {'train_X': train_X, 'train_y': train_y, 'test_X': test_X, 'test_y': test_y}


def eval_model(model, data_splits):
    start = time.time()
    model.fit(data_splits['train_X'], data_splits['train_y'])
    fit_time = time.time() - start

    data_log = data_splits['test_X'].copy()
    data_log['prediction'] = model.predict(data_log)
    data_log['target'] = data_splits['test_y']

    res = {
        'fit_time': fit_time,
        'r2': r2_score(data_log['target'], data_log['prediction']),
        'mse': mean_squared_error(data_log['target'], data_log['prediction']),
        'corr': spearmanr(data_log['prediction'], data_log['target'])[0],
        'tau': kendalltau(data_log['prediction'], data_log['target'])[0]
    }

    return res, data_log


def get_timestamp():
    return datetime.fromtimestamp(time.time()).strftime("%d-%m-%Y-%H-%M-%S-%f")


def train_end_evaluate(args):
    benchmark, dataset = args['benchmark'], args['dataset']

    # initialize wandb
    if not args['debug_']:
        cfg_args = {k: v for k, v in args.items() if not k.endswith('_')}
        wandb.login(key=args['wandb_key_'])
        wandb.init(project=args['wandb_project_'], config=cfg_args, name=f"train_{benchmark}_{dataset}_{get_timestamp()}", tags=[benchmark, dataset, 'cache'])

    # get iterator of all available networks
    if benchmark == 'darts':
        df = pd.read_csv('../../zc_combine/data/nb301_nets.csv', index_col=0)
        net_iterator = (DARTS(n) for n in df.index)
    else:
        net_cls = searchspace_classes[benchmark]
        dataset_api = get_dataset_api(search_space=dataset_api_maps[benchmark], dataset=dataset)
        net_iterator = net_cls.get_arch_iterator(dataset_api)

        if net_cls.random_iterator:
            raise ValueError("Not implemented for DARTS.")

    # load feature funcs and precomputed data
    feature_funcs = load_from_config(args['config'], benchmark)

    cached_data = None
    cached_zcp = [pd.read_csv(args['cached_zcp_path_'], index_col=0)] if args['cached_zcp_path_'] is not None else []
    cached_features = [pd.read_csv(args['cached_features_path_'], index_col=0)] if args['cached_features_path_'] is not None else []
    if len(cached_zcp) > 0 or len(cached_features) > 0:
        cached_data = pd.concat([*cached_zcp, *cached_features], axis=1)

    # load target data, compute features
    filename_args = ['cache_prefix_', 'benchmark', 'dataset', 'use_features', 'use_zcp', 'use_onehot']
    cache_path = f"{'_'.join([str(args[fa]) for fa in filename_args])}_{os.path.splitext(os.path.basename(args['config']))}.pickle"
    if args['cache_dataset_'] and os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            cd = pickle.load(f)
            feature_dataset, y = cd['dataset'], cd['y']
    else:
        y = pd.read_csv(args['target_path_'], index_col=0)
        graf_model = GRAF(feature_funcs, benchmark, cached_data=cached_data, cache_features=False, no_zcp_raise=True)
        feature_dataset, y = create_dataset(graf_model, net_iterator, target_df=y, zcp_names=zcps,
                                            target_name=args['target_name'], use_features=args['use_features'],
                                            use_zcp=args['use_zcp'], use_onehot=args['use_onehot'],
                                            drop_unreachables='micro' in benchmark or benchmark == 'nb201')
        if args['cache_dataset_']:
            with open(cache_path, 'wb') as f:
                pickle.dump({'dataset': feature_dataset, 'y': y}, f)

    # get regressor
    model = xgb.XGBRegressor(
        objective='reg:squarederror',  # Define the objective as regression with squared error
        n_estimators=1000,  # Number of boosting rounds
        learning_rate=0.05,  # Small learning rate to reduce overfitting
        max_depth=5,  # Maximum depth of a tree to avoid overfitting
        subsample=0.8,  # Subsample ratio of the training instances
        colsample_bytree=1,  # Subsample ratio of columns when constructing each tree
        gamma=0.001,  # Minimum loss reduction required to make a further partition
        reg_alpha=1.0,  # L1 regularization term on weights
        reg_lambda=1.0,  # L2 regularization term on weight
    )

    # fit and eval N times
    for i in range(args['n_train_evals']):
        data_seed = args['seed'] + i
        data_splits = get_train_test_splits(feature_dataset, y, args['train_size'], args['test_size'],
                                            data_seed, args['sample_strategy'])

        res, data_log = eval_model(model, data_splits)
        if not args['debug_']:
            wandb.log(res, step=data_seed)

            if i==0:
                data_log.to_csv(args['cached_zcp_path_'].replace('cached_vkdnw', 'pred_'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run fit-eval of features/zero-cost proxies predictor with different sampling methods."
    )
    parser.add_argument('--benchmark', default='nb201', help="Which NAS benchmark to use (e.g. nb201).")
    parser.add_argument('--dataset', default='cifar10', help="Which dataset from the benchmark to use (e.g. cifar10).")
    parser.add_argument('--config', required=True, help="Path to the feature configuration file.")
    parser.add_argument('--cached_features_path_', default=None, help="Path to the cached features file.")
    parser.add_argument('--cached_zcp_path_', default=None, help="Path to the cached zcp score file.")
    parser.add_argument('--target_path_', required=True,
                        help="Path to network targets (e.g. accuracy). It should be a .csv file with net hashes as "
                             "index and `target_name` among the columns.")
    parser.add_argument('--target_name', default='val_accs', help="Name of the target column.")
    parser.add_argument('--seed', default=42,
                        help="Random seed for sampling the training data. For test data, `seed + 1` is used instead.")
    parser.add_argument('--n_train_evals', default=50, type=int,
                        help="Number of training samples on which the model is trained and evaluated.")
    parser.add_argument('--train_size', default=100, type=int,
                        help="Number of architectures to sample for the training set.")
    parser.add_argument('--test_size', default=1000, type=int,
                        help="Number of architectures to sample for the test set.")
    parser.add_argument('--sample_strategy', default="random", help="Strategy for sampling train networks.")
    parser.add_argument('--wandb_key_', required=True, help='Login key to wandb.')
    parser.add_argument('--wandb_project_', default='graf_sampling', help="Wandb project name.")
    parser.add_argument('--debug_', action='store_true', help="If True, do not sync to wandb.")
    parser.add_argument('--use_features', action='store_true', help="If True, use features from GRAF.")
    parser.add_argument('--use_zcp', action='store_true', help="If True, use zero-cost proxies.")
    parser.add_argument('--use_onehot', action='store_true', help="If True, use the onehot encoding.")
    parser.add_argument('--cache_prefix_', default=None, help="Cache path filename prefix.")
    parser.add_argument('--cache_dataset_', action='store_true', help="If True, cache everything including zps.")

    args = parser.parse_args()
    args = vars(args)
    train_end_evaluate(args)