
RBM
===

.. image:: https://travis-ci.org/SrMouraSilva/RBM.svg?branch=master
    :target: https://travis-ci.org/SrMouraSilva/RBM
    :alt: Build Status

.. image:: https://readthedocs.org/projects/srmourasilva-rbm/badge/?version=latest
    :target: http://srmourasilva-rbm.readthedocs.io/?badge=latest
    :alt: Documentation Status

.. image:: https://codecov.io/gh/SrMouraSilva/RBM/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/SrMouraSilva/RBM
    :alt: Code coverage

An implementation of RBM. Check the documentation :)


The implementation are based from:

* **Hinton RBM implementation**: http://www.cs.toronto.edu/~hinton/code/rbm.m
* **Infinite RBM**: https://github.com/MarcCote/iRBM/blob/master/iRBM/models/rbm.py
* **Scikit-learn RBM**: https://github.com/scikit-learn/scikit-learn/blob/a24c8b46/sklearn/neural_network/rbm.py
* **Keras extensions - RBM**: https://github.com/wuaalb/keras_extensions/blob/master/keras_extensions/rbm.py


Docker
------

Clone repo

.. code-block:: bash

    git clone https://github.com/SrMouraSilva/RBM --depth=1

Go to docker path

.. code-block:: bash

    cd docker

Install dependencies

.. code-block:: bash

    docker-compose up install

Run experiment

.. code-block:: bash

    docker-compose up tensorflow

Inspect learning (tensorboard)

.. code-block:: bash

    docker-compose up tensorboard
