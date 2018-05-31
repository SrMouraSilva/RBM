from rbm.rbm import RBM
from rbm.train.tensorflow_task import TensorFlowTask
from rbm.train.trainer import Trainer

from tests.mnist import *
import time

dataset = train_images()

total_elements = len(dataset)
size_element = len(dataset[0]) ** 2

dataset = 0 > dataset
dataset = dataset.reshape((total_elements, size_element))

dataset = np.asarray([list([0]*size_element), list([0]*size_element)])
#dataset = dataset.reshape((len(dataset), 1, size_element))
#print(dataset.reshape((len(dataset), size_element, 1)))

rbm = RBM(visible_size=28**2, hidden_size=28**2)
trainer = Trainer(rbm, dataset)
trainer.stopping_criteria.append(lambda epoch: epoch > 100)
trainer.tasks.append(TensorFlowTask(log="../graph/mnist/{}".format(time.time())))

trainer.train()
