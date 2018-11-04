import time

import tensorflow as tf

from rbm.learning.constant_learning_rate import ConstantLearningRate
from rbm.rbm import RBM
from rbm.regularization.regularization import L1Regularization, L2Regularization
from rbm.train.task.beholder_task import BeholderTask
from rbm.train.task.inspect_images_task import InspectImagesTask
from rbm.train.task.persistent_task import PersistentTask
from rbm.train.task.rbm_inspect_histograms_task import RBMMeasureTask
from rbm.train.task.summary_task import SummaryTask
from rbm.train.trainer import Trainer
from tests.mnist import *

tf.set_random_seed(42)

dataset = train_images()

total_elements = len(dataset)
size_element = len(dataset[0]) ** 2

# To binary
dataset = dataset > 127

dataset = dataset.astype(np.uint8)
dataset = dataset.reshape((total_elements, size_element))

# Batch_size = 10 or 100
# https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf
batch_size = 100

rbm = RBM(
    visible_size=size_element,
    #hidden_size=10,
    hidden_size=1000,
    #regularization=L1Regularization(0.01),
    regularization=L2Regularization(0.01),
    learning_rate=ConstantLearningRate(0.05)
)

trainer = Trainer(rbm, dataset, batch_size=batch_size)

trainer.stopping_criteria.append(lambda epoch: epoch > 50)

log = "../experiments/logs/{}/{}".format(batch_size, time.time())
trainer.tasks.append(RBMMeasureTask())
trainer.tasks.append(InspectImagesTask())
trainer.tasks.append(SummaryTask(log=log))
trainer.tasks.append(BeholderTask(log='../experiments/logs'))
#trainer.tasks.append(PersistentTask(path="../experiments/model/{}/rbm.ckpt".format(batch_size)))

trainer.train()
