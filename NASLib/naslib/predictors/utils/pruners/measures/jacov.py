# Copyright 2021 Samsung Electronics Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""
This contains implementations of jacov based on
https://github.com/BayesWatch/nas-without-training (jacov).
This script was based on
https://github.com/SamsungLabs/zero-cost-nas/blob/main/foresight/pruners/measures/jacob_cov.py
We found this version of jacov tends to perform
better.
Author: Robin Ru @ University of Oxford

"""

import torch
import numpy as np

from . import measure
from scipy.stats import skew, kurtosis, kstest, cramervonmises, chisquare, wasserstein_distance, energy_distance, \
    gaussian_kde


def get_batch_jacobian(net, x, target):
    net.zero_grad()

    x.requires_grad_(True)

    y = net(x)

    y.backward(torch.ones_like(y))
    jacob = x.grad.detach()

    return jacob, target.detach()


def eval_score(jacob, labels=None):
    corrs = np.corrcoef(jacob)
    v, _ = np.linalg.eigh(corrs)
    k = 1e-5
    return -np.sum(np.log(v + k) + 1.0 / (v + k))


@measure("jacov", bn=True)
def compute_jacob_cov(net, inputs, targets, split_data=1, loss_fn=None):
    try:
        # Compute gradients (but don't apply them)
        jacobs, labels = get_batch_jacobian(net, inputs, targets)
        jacobs = jacobs.reshape(jacobs.size(0), -1).cpu().numpy()
        jc = eval_score(jacobs, labels)
    except Exception as e:
        print(e)
        jc = np.nan

    return jc

@measure("jacov_full", bn=True)
def compute_jacob_cov_full(net, inputs, targets, split_data=1, loss_fn=None):
    # Compute gradients (but don't apply them)
    jacobs, _ = get_batch_jacobian(net, inputs, targets)
    jacobs = jacobs.reshape(jacobs.size(0), -1)
    return get_matrix_stats(torch.corrcoef(jacobs), 'jacov')

def get_matrix_stats(matrix, matrix_name, ret_all=False):

    try:
        lambdas = torch.linalg.eigvalsh(matrix).detach()
    except RuntimeError as e:
        if ("CUDA error: an illegal memory access was encountered" in str(e)) or isinstance(e, torch._C._LinAlgError):
            print(str(e))
            lambdas = torch.empty((2,), device=matrix.device)
        else:
            raise  # re-raise the exception if it's not the specific RuntimeError you want to catch

    rtn = {}

    # Dispersion
    rtn.update({
            matrix_name + '_cond': lambdas.max().item() / lambdas.min().item() if lambdas.min().item() > 0 else None,
            matrix_name + '_max': lambdas.max().item(),
            matrix_name + '_min': lambdas.min().item(),
            matrix_name + '_coef': lambdas.std().item() / lambdas.mean().item() if lambdas.mean().item() > 0 else None,
    })

    # Statistics
    rtn.update(
        {matrix_name+'_'+key: value for key, value in get_statistical_tests(lambdas.cpu().numpy()).items()}
    )

    # Eigenvalues
    if ret_all:
        rtn.update({matrix_name + '_lambda': lambdas})
    return rtn

def get_statistical_tests(lambdas):

    if (lambdas.max() == np.inf) or (lambdas.min() == 0):
        return {
            'skew' : None,
            'kurtosis' : None,
            'kstest' : None,
            'cramervonmises' : None,
            'chisquare' : None,
            'wasserstein_distance' : None,
            'energy_distance' : None,
            'entropy' : None,
        }

    # Normalize
    lambdas = lambdas - lambdas.min()
    lambdas = lambdas / lambdas.max()

    # Compute
    rtn = {
        'skew': skew(lambdas),
        'kurtosis': kurtosis(lambdas),
        'kstest': kstest(lambdas, 'uniform')[1],
        'cramervonmises': cramervonmises(lambdas, 'uniform').pvalue,
        'chisquare': chisquare(lambdas).pvalue,
        'wasserstein_distance': wasserstein_distance(lambdas, np.linspace(0, 1, len(lambdas))),
        'energy_distance': energy_distance(lambdas, np.linspace(0, 1, len(lambdas))),
        'entropy': estimate_entropy_kde(lambdas),
    }

    return rtn

def estimate_entropy_kde(data, method='scott'):
    if (data.max()-data.min() == 0.) or (sum(np.isnan(data)) + sum(np.isinf(data)) > 0):
        return None
    data = (data-data.min())/(data.max()-data.min())
    try:
        kde = gaussian_kde(data, bw_method=method)
        return -np.mean(np.log(kde(np.linspace(0, 1, 100))))
    except np.linalg.LinAlgError:
        print("Error: Eigenvalues have zero variance.")
        return None