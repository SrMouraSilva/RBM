import tensorflow as tf
from GPyOpt.methods import BayesianOptimization

from experiments.data.load_data_util import load_data_one_hot_encoding, get_specific_fold
from experiments.experiment import train
from rbm.learning.adam import Adam
from rbm.rbmcf import RBMCF
from rbm.regularization.regularization import L2
from rbm.sampling.contrastive_divergence import ContrastiveDivergence
from rbm.util.util import rmse

bounds = [
    {'name': 'hidden_size',     'type': 'discrete',   'domain': (50, 500, 1000, 2000)},
    {'name': 'learning_rate',   'type': 'continuous', 'domain': (0.02, 0.0001)},
    {'name': 'sampling_method', 'type': 'discrete',   'domain': (1, 3, 5)},
    {'name': 'regularization',  'type': 'continuous', 'domain': (0.01, 0.00001)},
    {'name': 'batch_size',      'type': 'discrete',   'domain': (10, 16, 32)},
]


data_train, data_test = get_specific_fold(fold=1, data=load_data_one_hot_encoding())


def f(x):
    tf.reset_default_graph()
    x = x[0]

    batch_size = int(x[4])

    model, path = train(
        kfold='1',
        data_train=data_train,
        data_validation=data_test,

        model_class=RBMCF,

        hidden_size=int(x[0]),
        learning_rate=Adam(x[1]),
        sampling_method=ContrastiveDivergence(int(x[2])),
        regularization=L2(x[3]),
        batch_size=batch_size,

        epochs=batch_size*150,
        persist=True
    )

    with tf.Session() as session:
        model.load(session, path)

        p_h = model.P_h_given_v(data_test.T.values)
        reconstruction = model.P_v_given_h(p_h)
        operation = rmse(data_test.T.values, reconstruction)

        return session.run(operation)


optimization = BayesianOptimization(f=f, domain=bounds, verbosity=True)
optimization.run_optimization(max_iter=10)

print(optimization.fx_opt, optimization.x_opt)
