# -*- coding: utf-8 -*-
from sklearn.utils import shuffle
from theano.tensor.shared_randomstreams import RandomStreams

import numpy as np
import theano
import theano.tensor as T

import dataload.mnist as dataload

import matplotlib.pyplot as plt

debug_test = False

rng = np.random.RandomState(101)
theano_rng = RandomStreams(rng.randint(102))

train_X, test_X, train_y, test_y, data_shape = dataload.prepare_dataset(debug_test=debug_test, y_onehot=True)

one_data_size = 1
for i in data_shape:
    one_data_size *= i

num_datatypes = 10

##########

class Layer:
    # Constructor
    def __init__(self, in_dim, out_dim, function=lambda x: x):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.function = function

        # Xavier
        self.W = theano.shared(rng.uniform(
            low=-np.sqrt(6. / (in_dim + out_dim)),
            high=np.sqrt(6. / (in_dim + out_dim)),
            size=(in_dim, out_dim)
        ).astype('float32'), name='W')

        self.b = theano.shared(np.zeros(out_dim).astype('float32'), name='b')

        self.params = [self.W, self.b]

    # Forward Propagation
    def f_prop(self, x):
        self.u = T.dot(x, self.W) + self.b
        self.z = self.function(self.u)
        return self.z


class VAE:
    def __init__(self, q, p, random=103):
        self.q = q
        self.p = p
        self.srng = RandomStreams(seed=random)

    def q_f_prop(self, x, y):
        # Gaussian MLP
        params = []
        layer_out = T.concatenate([x, y], axis=1)
        for i, layer in enumerate(self.q[:-2]):
            params += layer.params
            layer_out = layer.f_prop(layer_out)

        params += self.q[-2].params
        mean = self.q[-2].f_prop(layer_out)

        params += self.q[-1].params
        var = self.q[-1].f_prop(layer_out)

        return mean, var, params

    def p_f_prop(self, x, y):
        # Bernoulli MLP
        params = []
        layer_out = T.concatenate([x, y], axis=1)
        for i, layer in enumerate(self.p):
            params += layer.params
            layer_out = layer.f_prop(layer_out)
        mean = layer_out

        return mean, params

    def lower_bound(self, x, y):
        # Encode
        mean, var, q_params = self.q_f_prop(x, y)
        KL = -0.5 * T.mean(T.sum(1 + T.log(var) - mean ** 2 - var, axis=1))

        epsilon = self.srng.normal(mean.shape)
        z = mean + T.sqrt(var) * epsilon

        # Decode
        _x, p_params = self.p_f_prop(z, y)
        log_likelihood = T.mean(T.sum(x * T.log(_x) + (1 - x) * T.log(1 - _x),
                                      axis=1))

        params = q_params + p_params

        lower_bound = [-KL, log_likelihood]

        return lower_bound, params


# 更新則(Adam)
# https://gist.github.com/Newmu/acb738767acb4788bac3
def adam(params, g_params, lr=0.0002, b1=0.1, b2=0.001, e=1e-8):
    updates = []
    i = theano.shared(np.float32(0.), name='i')
    i_t = i + 1.
    fix1 = 1. - (1. - b1) ** i_t
    fix2 = 1. - (1. - b2) ** i_t
    lr_t = lr * (T.sqrt(fix2) / fix1)
    for p, g in zip(params, g_params):
        m = theano.shared(p.get_value() * 0., name='m')
        v = theano.shared(p.get_value() * 0., name='v')
        m_t = (b1 * g) + ((1. - b1) * m)
        v_t = (b2 * T.sqr(g)) + ((1. - b2) * v)
        g_t = m_t / (T.sqrt(v_t) + e)
        p_t = p - (lr_t * g_t)
        updates.append((m, m_t.astype(dtype="float32")))
        updates.append((v, v_t.astype(dtype="float32")))
        updates.append((p, p_t.astype(dtype="float32")))

    updates.append((i, i_t))
    return updates


# ネットワークの定義
z_dim = 10
# Gaussian MLP
q = [
    Layer(one_data_size + num_datatypes, 200, T.nnet.relu),
    Layer(200, 200, T.nnet.relu),
    Layer(200, z_dim),  # mean
    Layer(200, z_dim, T.nnet.softplus)  # variance
]
# Bernoulli MLP
p = [
    Layer(z_dim + num_datatypes, 200, T.nnet.relu),
    Layer(200, 200, T.nnet.relu),
    Layer(200, one_data_size, T.nnet.sigmoid)
]

# train関数とtest関数
model = VAE(q, p)

x = T.fmatrix('x')
y = T.fmatrix('y')
lr = T.fscalar('lr')
lower_bound, params = model.lower_bound(x, y)

g_params = T.grad(-T.sum(lower_bound), params)
updates = adam(params, g_params, lr=lr)

train = theano.function(inputs=[x, y, lr], outputs=lower_bound, updates=updates,
                        allow_input_downcast=True, name='train')
test = theano.function(inputs=[x, y], outputs=T.sum(lower_bound),
                       allow_input_downcast=True, name='test')

# 学習
batch_size = 500

n_batches = train_X.shape[0] // batch_size
lr = 0.001

hoge = []
fuga = []

epoch_num = 50
lr_change_num = 3

for _ in range(lr_change_num):
    for epoch in range(epoch_num):
        shuffle(train_X, train_y, random_state=104)
        lowerbound_all = []
        for i in range(n_batches):
            start = i * batch_size
            end = start + batch_size
            lowerbound = train(train_X[start:end], train_y[start:end], lr)
            lowerbound_all.append(lowerbound)

        lowerbound_all = np.mean(lowerbound_all, axis=0)
        test_lowerbound = test(test_X, test_y)
        print('Epoch:%d, Train Lower Bound:%lf (%lf, %lf), Test Lower Bound:%lf' %
              (epoch, sum(lowerbound_all), lowerbound_all[0], lowerbound_all[1],
               test_lowerbound))

        hoge.append(- np.sum(lowerbound_all))
        fuga.append(- np.sum(test_lowerbound))

    lr /= 2

plt.plot(range(epoch_num * lr_change_num), hoge, color='r')
plt.plot(range(epoch_num * lr_change_num), fuga, color='b')
plt.savefig('output.png')
