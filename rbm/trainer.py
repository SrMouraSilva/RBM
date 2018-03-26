import numpy as np


class Batch(object):

    def __init__(self, data, start, size):
        self.data = data

        self.start = start
        self.number = start
        self.size = size

    def next(self):
        """
        Implements as iterate
        :return:
        """
        return self.data[self.number*self.size: (self.number+1)*self.size]


class Trainer(object):

    def __init__(self, model, dataset, batch_size=None, starting_epoch=1):
        """
        :param Model model:
        :param dataset:
        :param batch_size:
        :param starting_epoch:
        """

        #self.learn = None

        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size if batch_size is not None else len(dataset)

        self.nb_updates = int(np.ceil(len(dataset) / self.batch_size))
        self.starting_epoch = starting_epoch

        #self.stopping_criteria = []
        #self.tasks = []

        self.epoch = 0
        self.no_update = 0
        self.final_epoch = None

        # Build learner
        self.input = T.matrix('input')
        self.no_batch = T.iscalar('no_batch')
        self.updates = self.model.get_updates(self.input)

    def train(self):
        if self.learn is None:
            self.build()

        #self.init()

        # Learning
        for self.epoch in count(start=self.starting_epoch):
            # Check stopping criteria
            if any([stopping_criterion.check(self.epoch) for stopping_criterion in self.stopping_criteria]):
                break

            #self.pre_epoch()

            for self.no_update in range(1, self.nb_updates+1):
                #self.pre_update()
                self.learn(self.no_update-1)
                #self.post_update()

            #self.post_epoch()

        self.final_epoch = self.epoch - 1
        #self.finished()

    def learn(self):
        theano.function(
            [self.no_batch],
            updates=self.updates,
            givens={
                self.input: self.dataset.inputs[self.no_batch * self.batch_size:(self.no_batch + 1) * self.batch_size]
            },
            name="learn"
        )
