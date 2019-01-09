"""
https://wiseodd.github.io/techblog/2017/12/23/annealed-importance-sampling/
"""

from typing import List

import numpy as np
import tensorflow as tf

#from iRBM.misc import utils
from rbm.rbm import RBM
from rbm.util.rbm_base_rate import RBMBaseRate, BaseRateType


BETAS = np.r_[
    np.linspace(0, 0.5, num=500),
    np.linspace(0.5, 0.9, num=4000),
    np.linspace(0.9, 1, num=10000)
]


def estimate_lnZ(rbm, M=100, betas=BETAS):
    return compute_AIS(rbm, M, betas)['logcummean_Z'][-1]


def estimate_lnZ_with_std(rbm, M=100, betas=BETAS):
    info = compute_AIS(rbm, M, betas)
    return info['logcummean_Z'][-1], (info['logcumstd_Z_down'][-1], info['logcumstd_Z_up'][-1]), (info['std_lnZ'][-1], info['std_lnZ'][-1])


def compute_AIS(model: RBM, M=100, betas=BETAS, seed=1234):
    # Try different size of batch size.
    batch_size = M

    while batch_size >= 1:
        "Computing AIS using batch size of {}".format(batch_size)

        try:
            return _compute_AIS(model, M=M, betas=betas, batch_size=int(batch_size), seed=seed)
        except MemoryError as e:
            # Probably not enough memory on GPU
            print(e)
        except ValueError as e:
            # Probably because of the limited Multinomial op
            print(e)

        print("*An error occured while computing AIS. Will try a smaller batch size to compute AIS.")
        batch_size = batch_size // 2

    raise RuntimeError("Cannot find a suitable batch size to compute AIS. Try using CPU instead.")


def _compute_AIS(model: RBM, M=100, betas=BETAS, batch_size=None, seed=1234):
    if batch_size is None:
        batch_size = M

    # Will be executing M AIS's runs.
    last_sample_chain = np.zeros((M, model.visible_size), dtype=np.float32)
    M_log_w_ais = np.zeros((M, 1), dtype=np.float64)

    tf.random.set_random_seed(seed)

    ais_results = {}

    # Iterate through all AIS runs.
    for i in range(0, M, batch_size):
        if i <= ais_results.get('batch_id', -1):
            continue

        actual_size = min(M - i, batch_size)
        print("AIS run: {}/{} (using batch size of {})".format(i, M, batch_size))
        ais_partial_results = _compute_AIS_samples(model, M=actual_size, betas=betas)

        M_log_w_ais[i:i+batch_size] = ais_partial_results['M_log_w_ais']
        last_sample_chain[i:i+batch_size] = ais_partial_results['last_sample_chain'].T
        lnZ_trivial = ais_partial_results['lnZ_trivial']

        ais_results = {
            'batch_id': i,
            'M': M,
            'batch_size': batch_size,
            'last_sample_chain': last_sample_chain.T,
            'M_log_w_ais': M_log_w_ais,
            'lnZ_trivial': lnZ_trivial
        }

    # We compute the mean of the estimated `r_AIS`
    Ms = np.arange(1, M+1)
    log_sum_w_ais = np.logaddexp.accumulate(M_log_w_ais)
    logcummean_Z = log_sum_w_ais - np.log(Ms)

    # We compute the standard deviation of the estimated `r_AIS`
    logstd_AIS = np.zeros_like(M_log_w_ais)
    for k in Ms[1:]:
        m = np.max(M_log_w_ais[:k])
        logstd_AIS[k-1] = np.log(np.std(np.exp(M_log_w_ais[:k]-m), ddof=1)) - np.log(np.sqrt(k))
        logstd_AIS[k-1] += m

    logstd_AIS[0] = np.nan  # Standard deviation of only one sample does not exist.

    # The authors report AIS error using ln(Ẑ ± 3\sigma)
    m = max(np.nanmax(logstd_AIS), np.nanmax(logcummean_Z))
    logcumstd_Z_up = np.log(np.exp(logcummean_Z-m) + 3*np.exp(logstd_AIS-m)) + m - logcummean_Z
    logcumstd_Z_down = -(np.log(np.exp(logcummean_Z-m) - 3*np.exp(logstd_AIS-m)) + m) + logcummean_Z

    # Compute the standard deviation of ln(Z)
    std_lnZ = np.array([np.std(M_log_w_ais[:k], ddof=1) for k in Ms[1:]])
    std_lnZ = np.r_[np.nan, std_lnZ]  # Standard deviation of only one sample does not exist.

    return {
        "logcummean_Z": logcummean_Z,#.astype(tf.float32),
        "logcumstd_Z_down": logcumstd_Z_down,#.astype(tf.float32),
        "logcumstd_Z_up": logcumstd_Z_up,#.astype(tf.float32),
        "logcumstd_Z": logstd_AIS,#.astype(tf.float32),
        "M_log_w_ais": M_log_w_ais,
        "lnZ_trivial": lnZ_trivial,
        "std_lnZ": std_lnZ,
        "last_sample_chain": last_sample_chain,
        "batch_size": batch_size,
        "seed": seed,
        "nb_temperatures": len(betas),
        "nb_samples": M
    }


def _compute_AIS_samples(model: RBM, M=100, betas=BETAS):
    """
    ref: Salakhutdinov & Murray (2008), On the quantitative analysis of deep belief networks
    """
    model.batch_size = M  # Will be executing `M` AIS's runs in parallel.

    # Base rate with same visible biases as the model
    model_base_rate, annealable_params = RBMBaseRate(model).generate(BaseRateType.b_v)

    betas = tf.Variable(name="Betas", initial_value=betas, dtype=tf.float32)

    # cadÊ?
    #k = T.iscalar('k')
    k = tf.placeholder(name='k', dtype=tf.int32)
    v = tf.Variable(name="v", initial_value=tf.zeros((model.visible_size, M)), dtype=tf.float32)

    #sym_log_w_ais_k, updates = _log_annealed_importance_sample(model, v, k, betas, annealable_params), None
    sym_log_w_ais_k = _log_annealed_importance_sample(model, v, k, betas, annealable_params)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        lnZ_trivial = model_base_rate.compute_lnZ().eval()

        # Will be executing M AIS's runs.
        #last_sample_chain = tf.zeros((M, model.visible_size), dtype=tf.float32)
        M_log_w_ais = np.zeros((M, 1), dtype=np.float64)

        # First sample V0
        h0 = model_base_rate.sample_h_given_v(tf.zeros((model.visible_size, M), dtype=tf.float32))
        v0 = model_base_rate.sample_v_given_h(h0).eval()
        v = v.assign(v0)  # Set initial v for AIS

        # Iterate through all betas (temperature parameter)
        print('Total of betas:', betas.shape[0])
        for k_index in range(1, betas.shape[0]):
            M_log_w_ais += session.run(sym_log_w_ais_k, feed_dict={k: k_index})
            print(k_index)

        M_log_w_ais += lnZ_trivial

        return {
            "M_log_w_ais": M_log_w_ais,
            "last_sample_chain": v.eval(),
            "lnZ_trivial": lnZ_trivial
        }


def _log_annealed_importance_sample(model: RBM, v, k, betas, annealable_params: List[tf.Variable]):
    beta_k = betas[k]
    beta_k_minus_1 = betas[k-1]

    # Important to backup model's parameters as we modify them
    params = [(param.name, tf.identity(param)) for param in annealable_params]

    # Set `param * beta_k_minus_1`
    for name, param in params:
        setattr(model, name, param * beta_k_minus_1)

    log_pk_minus_1 = - model.F(v)

    # Set `param * beta_k`
    for name, param in params:
        setattr(model, name, param * beta_k)

    log_pk = - model.F(v)

    # Restore original parameters of the model.
    for name, param in params:
        setattr(model, name, param)

    del params

    return log_pk - log_pk_minus_1
