
import torch
from functorch import make_functional_with_buffers, vmap, grad, jacrev
from scipy.stats import skew, kurtosis, kstest, cramervonmises, chisquare, wasserstein_distance, energy_distance, \
    gaussian_kde
import gc
import numpy as np

from . import measure

@measure("vkdnw")
def compute_vkdnw(net, inputs, targets, loss_fn, split_data=1):

    #tenas = get_tenas(net, net(inputs))
    #tenas_prob = get_tenas(net, net(inputs), use_logits=False)
    fisher = get_fisher(net, inputs)
    fisher_prob = get_fisher(net, inputs, use_logits=False)

    rtn = {}
    outputs = torch.softmax(net(inputs), dim=1)
    outputs_mean = outputs.mean(dim=0)
    rtn.update({
        'class_nunique': float(torch.unique(torch.argmax(outputs, dim=1)).numel()),
        'output_entropy': -torch.sum(outputs_mean[outputs_mean>0] * torch.log(outputs_mean[outputs_mean>0])).item(),
    })
    #rtn.update(get_matrix_stats(tenas, 'tenas'))
    #rtn.update(get_matrix_stats(tenas_prob, 'tenas_prob'))
    rtn.update(get_matrix_stats(fisher, 'fisher', ret_quantiles=True))
    rtn.update(get_matrix_stats(fisher, 'fisher_svd', ret_quantiles=True, svd=True))
    rtn.update({'fisher_dim': float(len(list(net.named_parameters())))})
    rtn.update(get_matrix_stats(fisher_prob, 'fisher_prob', ret_quantiles=True))
    rtn.update(get_matrix_stats(fisher_prob, 'fisher_prob_svd', ret_quantiles=True, svd=True))

    return rtn

@measure("vkdnw_hist")
def compute_vkdnw_hist(net, inputs, targets, loss_fn, split_data=1):

    tenas = get_tenas(net, net(inputs))
    #tenas_prob = get_tenas(net, net(inputs), use_logits=False)
    #fisher = get_fisher(net, inputs)
    #fisher_prob = get_fisher(net, inputs, use_logits=False)

    lambdas =  torch.linalg.eigvalsh(tenas).detach().cpu().numpy()
    return {i: lambdas[i] for i in range(len(lambdas))}

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

def get_matrix_stats(matrix, matrix_name, ret_quantiles=False, svd=False):

    if svd:
        lambdas = torch.svd(matrix).S.detach()
    else:
        try:
            lambdas = torch.linalg.eigvalsh(matrix).detach()
        except RuntimeError as e:
            if ("CUDA error: an illegal memory access was encountered" in str(e)) or isinstance(e, torch._C._LinAlgError):
                print(f'Matrix {matrix_name}: ' + str(e))
                lambdas = torch.zeros((2,), device=matrix.device)
            else:
                raise  # re-raise the exception if it's not the specific RuntimeError you want to catch

    rtn = {}

    # Dispersion
    rtn.update({
            matrix_name + '_cond': lambdas.max().item() / lambdas.min().item() if lambdas.min().item() > 0 else None,
            matrix_name + '_max': lambdas.max().item(),
            matrix_name + '_min': lambdas.min().item(),
            matrix_name + '_coef': lambdas.std().item() / lambdas.mean().item() if lambdas.mean().item() > 0 else None,
            matrix_name + '_norm_fro': torch.log(torch.norm(matrix, p='fro')).item(),
            matrix_name + '_norm_max': torch.log(torch.max(torch.sum(torch.abs(matrix), dim=0))).item(),
            matrix_name + '_norm_inf': torch.log(torch.max(torch.sum(torch.abs(matrix), dim=1))).item(),
            matrix_name + '_norm_spec': torch.log(torch.max(torch.svd(matrix).S)).item(),
            matrix_name + '_norm_nuc': torch.log(torch.norm(matrix, p='nuc')).item(),
    })

    # Aggregation
    rtn.update({matrix_name + '_agg': (torch.log(lambdas + 1e-32) + 1.0 / (lambdas + 1e-32)).sum().item()})

    # Statistics
    rtn.update(
        {matrix_name+'_'+key: value for key, value in get_statistical_tests(lambdas.cpu().numpy()).items()}
    )

    # Eigenvalues
    if ret_quantiles:
        quantiles = torch.quantile(lambdas, torch.arange(0.1, 1.0, 0.1, device=lambdas.device))
        rtn.update({matrix_name + '_lambda_' + str(i): v.item() for (i,v) in enumerate(quantiles)})
    return rtn


def get_statistical_tests(lambdas):

    if (lambdas.max() == np.inf) or (lambdas.min() == 0) or (lambdas.min() == lambdas.max()):
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
        'kstest': kstest(lambdas, 'uniform')[0],
        'cramervonmises': cramervonmises(lambdas, 'uniform').statistic,
        'chisquare': chisquare(lambdas).statistic,
        'wasserstein_distance': wasserstein_distance(lambdas, np.linspace(0, 1, len(lambdas))),
        'energy_distance': energy_distance(lambdas, np.linspace(0, 1, len(lambdas))),
        'entropy': estimate_entropy_kde(lambdas),
    }

    return rtn

def get_tenas(model, output, use_logits=True):

    if not use_logits:
        output = torch.softmax(output, dim=1)

    grads = torch.empty(len(output), sum(p.numel() for n, p in model.named_parameters() if 'weight' in n),
                        device=output.device)
    for idx in range(len(output)):
        model.zero_grad()  # Clear previous gradients
        output[idx].backward(torch.ones_like(output[idx]), retain_graph=True)

        grad_list = []
        for name, param in model.named_parameters():
            if 'weight' in name and param.grad is not None:
                grad_list.append(param.grad.view(-1))

        if grad_list:
            grads[idx] = torch.cat(grad_list, dim=0)
        else:
            raise ValueError("No gradients found for any parameters.")

    # Compute NTK and its eigenvalues
    ntk = torch.mm(grads, grads.t())

    del grads, grad_list
    gc.collect()
    torch.cuda.empty_cache()

    return ntk

def get_fisher(model, input, use_logits=True):

    model.eval()

    jacobian = get_jacobian_index(model, input, 0)
    if not use_logits:
        jacobian = torch.matmul(cholesky_covariance(model(input)), jacobian).detach()

    #ntk = torch.mean(torch.matmul(jacobian, torch.transpose(jacobian, dim0=1, dim1=2)), dim=0).detach()
    fisher = torch.mean(torch.matmul(torch.transpose(jacobian, dim0=1, dim1=2), jacobian), dim=0).detach()

    del jacobian
    gc.collect()
    torch.cuda.empty_cache()

    return fisher

def cholesky_covariance(output):

    # Cholesky decomposition of covariance matrix (notation from Theorem 1 in https://sci-hub.se/10.2307/2345957)
    alpha = torch.tensor(0.05, dtype=torch.float16, device=output.device)
    prob = torch.nn.functional.softmax(output, dim=1) * (1 - alpha) + alpha / output.shape[1]
    q = torch.ones_like(prob) - torch.cumsum(prob, dim=1)
    q[:, -1] = torch.zeros_like(q[:, -1])
    q_shift = torch.roll(q, shifts=1, dims=1)
    q_shift[:, 0] = torch.ones_like(q_shift[:, 0])
    d = torch.sqrt(prob * q / q_shift)

    L = -torch.matmul(torch.unsqueeze(prob, dim=2), 1 / torch.transpose(torch.unsqueeze(q, dim=2), dim0=1, dim1=2))
    L = torch.nan_to_num(L, neginf=0.)
    L = L * (1 - torch.eye(L.shape[1], device=output.device, dtype=output.dtype).repeat(L.shape[0], 1, 1)) + \
        torch.eye(L.shape[1], device=output.device, dtype=output.dtype).repeat(L.shape[0], 1,
                                                                               1)  # replace diagonal elements by 1.
    L = L * (1 - torch.triu(torch.ones(L.shape[1], L.shape[2], device=output.device, dtype=output.dtype),
                            diagonal=1).repeat(L.shape[0], 1, 1))  # replace upper diagonal by 0
    L = torch.matmul(L, torch.diag_embed(d))  # multiply columns

    # Test
    cov_true = torch.diag_embed(prob) - torch.matmul(torch.unsqueeze(prob, dim=2),
                                                     torch.transpose(torch.unsqueeze(prob, dim=2), dim0=1, dim1=2))
    cov_cholesky = torch.matmul(L, torch.transpose(L, dim0=1, dim1=2))

    max_error = torch.abs(cov_true - cov_cholesky).max().item()
    if max_error > 1.0e-4:
        print(f'Cholesky decomposition back-test error with max error {max_error}')

    return L.detach()

def get_jacobian(model, input):

	fmodel, params_grad, buffers = make_functional_with_buffers(model)
	def compute_prediction(params, sample):
		return fmodel(params, buffers, sample.unsqueeze(0)).squeeze(0)
	def jacobian_sample(sample):
		return jacrev(compute_prediction, argnums=0)(params_grad, sample)

	jacobian = vmap(jacobian_sample)(input)
	return torch.cat([torch.flatten(v, start_dim=2, end_dim=-1) for v in jacobian], dim=2).detach()


def get_jacobian_index(model, input, param_idx):
    # Convert model to functional form
    func_model, params, buffers = make_functional_with_buffers(model)

    # Extract the gradient parameter subset
    params_grad = {k: v.flatten()[param_idx:param_idx + 1].detach() for (k, v) in model.named_parameters()}
    params_grad = dict(list(params_grad.items())[:250])

    def jacobian_sample(sample):
        def compute_prediction(params_grad_tmp):
            # Copy the original parameters and modify the specified gradients
            params_copy = [p.clone() for p in params]
            for i, (k, v) in enumerate(params_grad_tmp.items()):
                param_shape = params_copy[i].shape
                param = params_copy[i].flatten()
                param[param_idx:param_idx + 1] = v
                params_copy[i] = param.view(param_shape)

            # Compute the prediction using the functional model
            return func_model(params_copy, buffers, sample.unsqueeze(0)).squeeze(0)

        return jacrev(compute_prediction)(params_grad)

    # Apply vmap to efficiently compute Jacobians for each input in the batch
    jacobian_dict = vmap(jacobian_sample)(input)

    # Concatenate the Jacobian results across parameters
    ret = torch.cat([torch.flatten(v, start_dim=2, end_dim=-1) for v in jacobian_dict.values()], dim=2)

    return ret.detach()