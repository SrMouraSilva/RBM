import tensorflow as tf

from rbm.learning.learning_rate import LearningRate
from rbm.rbm import RBM
from rbm.util.util import σ


class CaRBM(RBM):
    """
    Ele fez para suportar tanto Bernouli, quanto Gausisan
    """

    def __init__(self, **kwargs):
        print('Training the CaRBM.')

        # The learning rate, usually [1e-3,1].
        self.learning_rate = LearningRate(0.01)

        mo = kwargs.get('mo', 0.9)  # Momentum to speed up learning (can be unstable), usually [0,0.9].
        num_hid = kwargs.get('num_hid', 100)  # num_hid = args.nh #The number of hidden units.
        num_iters = kwargs.get('num_iters', 100)  # The number of epochs of learning.
        batch_size = kwargs.get('batch_size', 100)  # The size of each mini-batch.
        kl = kwargs.get('kl', 10)  # The strength of the column-wise KL penalty to prevent dead units.
        l2 = kwargs.get('l2', 1e-5)  # The strength of the l2 weight-decay penalty.
        sp = kwargs.get('sp', 0.1)  # The amount of sparsity, i.e. nh=100 and sp=0.1 means 10 hidden units will get activated.
        sig = kwargs.get('sig', 1)

        num_on = float(np.floor(sp * num_hid))
        p = num_on / batch_size
        num_drop = num_hid - num_on
        num_examples = X.shape[0]
        num_batches = np.ceil(np.double(num_examples) / batch_size)
        ca = caw.ChainAlg(num_hid, int(num_on), int(num_on), batch_size)

    def learn_alguma_coisa(self):
        W = self.W
        b = self.b_h
        c = self.b_v

        dW = 0
        db = 0
        dc = 0

        η = self.learning_rate
        
        m = X.mean(0)
        errs = []
        for i in range(num_iters):
            obj = 0
            randIndices = np.random.permutation(num_examples)
            for batch in range(int(num_batches)):
                v = X[randIndices[np.mod(range(batch * batch_size, (batch + 1) * batch_size), num_examples)]] - m

                self.learn(v)

        return W, b, c

    def learn(self, v):
        with tf.name_scope('gibbs_step'):
            # POSITIVE PHASE
            h_pos = self.sample_h_given_v(v, sig, storage=0)

            # NEGATIVE PHASE
            v_neg = self.sample_v_given_h(h_pos, m)
            h_neg = self.sample_h_given_v(v_neg, sig, storage=1)

        # Keep a weighted running average of the hidden unit activations
        if i > 0:
            q = 0.9 * q + 0.1 * h_pos.mean(0)
        else:
            q = h_pos.mean(0)

        # Cálculo dos gradientes
        dW = mo * dW + (η * (v.T @ h_pos) - (v_neg.T @ h_neg)) / batch_size
        db = mo * db + η * np.mean(h_pos - h_neg, 0)
        dc = mo * dc + η * np.mean((v - v_neg), 0)

        dW = dW + (η * kl * np.dot(v.T, np.tile(p - q, (batch_size, 1))) / batch_size)
        dW = dW - l2 * W
        db = db + η * kl * (p - q)

        # As atribuições
        W = W + dW
        b = b + db
        c = c + dc

        # Isso aqui deve ser uma forma de ver mean(F(v)) - mean(F(samples))
        obj = obj + np.sum((v - v_neg) ** 2) / (X.shape[0])
        self.condicao_de_parada()
        print('Iteration %d complete. Objective value: %s' % (i + 1, obj))
        errs.append(obj)

        self.plot()

    def sample_v_given_h(self, h, m):
        mv_neg = self.P_v_given_h(h)
        return np.double(mv_neg > np.random.rand(*mv_neg.shape))

    def sample_h_given_v(self, v, sig, storage):
        node_pots_h = ((v @ W) / sig) + b
        exp_node_pots_h = np.exp(node_pots_h)
        mh_pos, h_pos = ca.infer(exp_node_pots_h, storage=storage)

        return h_pos

    def condicao_de_parada(self):
        if obj > 1e10 or not np.isfinite(obj):
            print('\nLearning has diverged.')
            if os.path.isfile(save_file):
                print('Deleting %s' % save_file)
                os.remove(save_file)
                print('File deleted: %s' % ~os.path.isfile(save_file))
            return W, b, c

    def plot(self):
        """
        Plot some diagnostics for this epoch.
        """
        if self.plotting:
            pylab.ion()
            pylab.subplot(1, 2, 1)
            d.print_aligned(W[:, 0:np.minimum(100, W.shape[1])])
            pylab.subplot(1, 2, 2)
            d.print_aligned((v_neg + m).T)
            pylab.draw()
