import click
import pandas as pd
from torch.utils.data import DataLoader, Sampler
import wandb
import time

from graf_nas.features.zero_cost import get_zcp_dataloader
from datetime import datetime

from graf_nas import GRAF
from graf_nas.features.config import load_from_config
from graf_nas.graf import create_dataset
from graf_nas.search_space import searchspace_classes, dataset_api_maps, DARTS
from naslib.utils import get_dataset_api


class LimitedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, start_batch, num_batches):
        self.dataset_size = len(dataset)
        self.batch_size = batch_size
        self.start_batch = start_batch
        self.num_batches = num_batches
        self.total_batches = (self.dataset_size + batch_size - 1) // batch_size  # Total number of batches

    def __iter__(self):
        # Start from the nth batch and continue for k batches
        start_idx = self.start_batch * self.batch_size
        end_idx = min(self.dataset_size, (self.start_batch + self.num_batches) * self.batch_size)

        indices = range(start_idx, end_idx)
        return iter(indices)

    def __len__(self):
        # Return number of elements for k batches starting from the nth batch
        start_idx = self.start_batch * self.batch_size
        return min(self.dataset_size - start_idx, self.num_batches * self.batch_size)


def get_n_classes(dataset):
    if dataset == 'cifar10':
        return 10
    elif dataset == 'cifar100':
        return 100
    elif dataset == 'ImageNet16-120':
        return 120
    else:
        return "Dataset not recognized"

def get_timestamp():
    return datetime.fromtimestamp(time.time()).strftime("%d-%m-%Y-%H-%M-%S-%f")

@click.command()
@click.option('--benchmark', default='nb201')
@click.option('--dataset', default='cifar10', help='Required only for the dataset api.')
@click.option('--config', default='../graf_nas/configs/nb201.json')
@click.option('--wandb_key', required=True)
@click.option('--wandb_project', default='graf_sampling')
@click.option('--target_path', required=True,
                        help="Path to network targets (e.g. accuracy). It should be a .csv file with net hashes as "
                             "index and `target_name` among the columns.")
@click.option('--out_path', required=True)
@click.option('--start_batch', default=0)
@click.option('--batch_size', default=32)
@click.option('--num_batches', default=1)
def main(benchmark, dataset, config, wandb_key, wandb_project, target_path, out_path, start_batch, batch_size, num_batches):

    timestamp = get_timestamp()
    out_path = out_path + f'{timestamp}.csv'

    # initialize wandb
    wandb.login(key=wandb_key)
    wandb.init(project=wandb_project, config={'benchmark': benchmark, 'dataset': dataset, 'config': config, 'target_path': target_path, 'out_path': out_path, 'start_batch': start_batch, 'batch_size': batch_size, 'num_batches': num_batches}, name=f"cache_{benchmark}_{dataset}_{get_timestamp()}", tags=[benchmark, dataset, 'cache'])

    zcps = ['vkdnw', 'epe_nas', 'fisher', 'flops', 'grad_norm', 'grasp', 'jacov', 'l2_norm', 'nwot', 'params', 'plain',
            'snip', 'synflow', 'zen']

    feature_funcs = load_from_config(config, benchmark)

    dataloader_raw = get_zcp_dataloader(dataset=dataset, zc_cfg='../NASLib/naslib/runners/zc/zc_config.yaml', batch_size=batch_size, data='../NASLib/naslib' if dataset=='ImageNet16-120' else '../zero_cost/NASLib')
    dataloader = DataLoader( # create dataloader that will return batch_number-th batch
        dataset=dataloader_raw.dataset,  # Use the same dataset from the original loader
        batch_size=dataloader_raw.batch_size,  # Keep the same batch size
        sampler=LimitedBatchSampler(dataloader_raw.dataset, dataloader_raw.batch_size, start_batch, num_batches),  # Use a SequentialSampler
        num_workers=dataloader_raw.num_workers,  # Optionally keep the same number of workers
        pin_memory=dataloader_raw.pin_memory  # Optionally keep the same pin_memory setting
    )

    graf_model = GRAF(feature_funcs, benchmark, dataloader=dataloader, cache_features=False, compute_new_zcps=True, num_batches=num_batches)

    net_cls = searchspace_classes[benchmark]

    if benchmark == 'darts':
        df = pd.read_csv('../../zc_combine/data/nb301_nets.csv', index_col=0)
        net_iterator = (DARTS(n) for n in df.index)
    else:
        dataset_api = get_dataset_api(search_space=dataset_api_maps[benchmark], dataset=dataset)
        net_iterator = net_cls.get_arch_iterator(dataset_api, n_classes=get_n_classes(dataset))

        if net_cls.random_iterator:
            raise ValueError("Not implemented for DARTS.")

    y = pd.read_csv(target_path, index_col=0)
    print(f'Dataloader: batch size {dataloader.batch_size}, batch number {start_batch}')
    feature_dataset, y = create_dataset(graf_model, net_iterator, target_df=y, zcp_names=zcps, use_zcp=True,
                                        drop_unreachables='micro' in benchmark or benchmark == 'nb201')
    feature_dataset.join(y.rename('val_acc'), how='left').to_csv(out_path)


if __name__ == "__main__":
    main()
