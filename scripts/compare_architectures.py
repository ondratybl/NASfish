import click
import pandas as pd

from graf_nas.features.zero_cost import get_zcp_dataloader

from graf_nas import GRAF
from graf_nas.features.config import load_from_config
from graf_nas.graf import create_dataset
from graf_nas.search_space import searchspace_classes, dataset_api_maps, DARTS
from naslib.utils import get_dataset_api

def get_n_classes(dataset):
    if dataset == 'cifar10':
        return 10
    elif dataset == 'cifar100':
        return 100
    elif dataset == 'ImageNet16-120':
        return 120
    else:
        return "Dataset not recognized"

@click.command()
@click.option('--benchmark', default='nb201')
@click.option('--dataset', default='cifar10', help='Required only for the dataset api.')
@click.option('--config', default='../graf_nas/configs/nb201.json')
@click.option('--target_path', required=True,
                        help="Path to network targets (e.g. accuracy). It should be a .csv file with net hashes as "
                             "index and `target_name` among the columns.")
@click.option('--start_batch', default=0)
@click.option('--batch_size', default=32)
@click.option('--num_batches', default=1)
def main(benchmark, dataset, config, target_path, start_batch, batch_size, num_batches):

    zcps = ['vkdnw_hist', 'epe_nas', 'fisher', 'flops', 'grad_norm', 'grasp', 'jacov', 'l2_norm', 'nwot', 'params', 'plain',
            'snip', 'synflow', 'zen']

    # OP_NAMES = ["Identity", "Zero", "ReLUConvBN3x3", "ReLUConvBN1x1", "AvgPool1x1"], edges = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
    zcps = ['vkdnw_hist']
    #nets = ['(2, 2, 2, 2, 2, 2)', '(3, 3, 3, 3, 3, 3)', '(2, 2, 2, 0, 0, 0), (3, 3, 3, 0, 0, 0)', '(0, 0, 0, 2, 2, 2)', '(0, 0, 0, 3, 3, 3)']
    nets = ['(3, 3, 3, 1, 3, 3)', '(3, 3, 3, 0, 3, 3)', '(3, 3, 3, 3, 3, 3)', '(2, 2, 2, 1, 2, 2)', '(2, 2, 2, 0, 2, 2)', '(2, 2, 2, 2, 2, 2)']

    feature_funcs = load_from_config(config, benchmark)

    dataloader = get_zcp_dataloader(dataset=dataset, zc_cfg='../NASLib/naslib/runners/zc/zc_config.yaml', batch_size=batch_size, data='../NASLib/naslib' if dataset=='ImageNet16-120' else '../zero_cost/NASLib')

    graf_model = GRAF(feature_funcs, benchmark, dataloader=dataloader, cache_features=False, compute_new_zcps=True, num_batches=num_batches)

    net_cls = searchspace_classes[benchmark]

    if benchmark == 'darts':
        df = pd.read_csv('../../zc_combine/data/nb301_nets.csv', index_col=0)
        net_iterator = (DARTS(n) for n in df.index)
    else:
        dataset_api = get_dataset_api(search_space=dataset_api_maps[benchmark], dataset=dataset)
        net_iterator = net_cls.get_arch_iterator(dataset_api, n_classes=get_n_classes(dataset))
        net_iterator = filter(lambda x: x.net in nets, net_iterator)
        if net_cls.random_iterator:
            raise ValueError("Not implemented for DARTS.")

    y = pd.read_csv(target_path, index_col=0)
    print(f'Dataloader: batch size {dataloader.batch_size}, batch number {start_batch}')
    feature_dataset, y = create_dataset(graf_model, net_iterator, target_df=y, zcp_names=zcps, use_zcp=True,
                                        drop_unreachables='micro' in benchmark or benchmark == 'nb201')
    feature_dataset = feature_dataset.join(y.rename('val_acc'), how='left') # feature_dataset.loc[feature_dataset.index==nets[0], range(32)].iloc[0, :].hist()
    a = 1


if __name__ == "__main__":
    main()
