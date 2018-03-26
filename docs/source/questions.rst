Questions
=========

Why a binomial distribution in ``P(v|h)`` and ``P(h|v)``?
 - https://github.com/MarcCote/iRBM/blob/master/iRBM/models/rbm.py#L59
 - https://github.com/MarcCote/iRBM/blob/master/iRBM/models/rbm.py#L69
 - Maybe because: https://github.com/wuaalb/keras_extensions/blob/master/keras_extensions/rbm.py#L163-L166
 - Because the value will be float, but v and h are binary!

Why in setup the initialization is ``1e^2``
- https://github.com/MarcCote/iRBM/blob/master/iRBM/models/rbm.py#L38
