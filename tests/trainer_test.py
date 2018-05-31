from rbm.rbm import RBM
from rbm.train.tensorflow_task import TensorFlowTask
from rbm.train.trainer import Trainer

from tests.mnist import *
import time

#dataset = np.asarray([list([0]*size_element), list([1]*size_element), list([1]*(size_element//2))+list([0]*(size_element//2))])
dataset = train_images()

total_elements = len(dataset)
size_element = len(dataset[0]) ** 2

dataset = 0 > dataset
dataset = dataset.reshape((total_elements, size_element))

rbm = RBM(visible_size=size_element, hidden_size=size_element)
trainer = Trainer(rbm, dataset, batch_size=10000)
trainer.stopping_criteria.append(lambda epoch: epoch > 10)
trainer.tasks.append(TensorFlowTask(log="../graph/mnist/vera/{}".format(time.time())))

trainer.train()
