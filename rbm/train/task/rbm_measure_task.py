from rbm.predictor.expectation.expectation_predictor import RoundMethod, NormalizationRoundingMethod


from rbm.predictor.expectation.rbm_expectation_predictor import RBMExpectationPredictor, ClassicalNormalization
from rbm.predictor.topk.rbm_top_k_predictor import RBMTop1Predictor, RBMTopKPredictor
from rbm.train.task.rbmcf_measure_task import RBMCFMeasureTask


class RBMMeasureTask(RBMCFMeasureTask):

    def __init__(self, movies_size, ratings_size, data_train, data_validation):
        super().__init__(data_train, data_validation)
        self._movie_size = movies_size
        self._rating_size = ratings_size
        self.shape_visibleT = [-1, movies_size*ratings_size]

    @property
    def shape_softmax(self):
        return [-1, self.movie_size, self.rating_size]

    @property
    def rating_size(self):
        return self._rating_size

    @property
    def movie_size(self):
        return self._movie_size

    @property
    def predictors(self):
        return {
            'top-1': RBMTop1Predictor(self.model, self.movie_size, self.rating_size),
            'top-5': RBMTopKPredictor(self.model, self.movie_size, self.rating_size, k=5),
            #'expectation/round': RBMExpectationPredictor(
            #    self.model, self.movie_size, self.rating_size, normalization=RoundMethod(),
            #    pre_normalization=ClassicalNormalization()
            #),
            'expectation/normalized': RBMExpectationPredictor(
                self.model, self.movie_size, self.rating_size, normalization=NormalizationRoundingMethod(),
                pre_normalization=ClassicalNormalization()
            ),
        }

        '''
        'expectation/round/classical_normalization': RBMExpectationPredictor(
            self.model, self.movie_size, self.rating_size, normalization=RoundMethod(),
            pre_normalization=ClassicalNormalization()
        ),
        'expectation/round/softmax_normalization': RBMExpectationPredictor(
            self.model, self.movie_size, self.rating_size, normalization=RoundMethod(),
            pre_normalization=SoftmaxNormalization()
        ),
        'expectation/normalized/classical_normalization': RBMExpectationPredictor(
            self.model, self.movie_size, self.rating_size, normalization=NormalizationRoundingMethod(),
            pre_normalization=ClassicalNormalization()
        ),
        'expectation/normalized/softmax_normalization': RBMExpectationPredictor(
            self.model, self.movie_size, self.rating_size, normalization=NormalizationRoundingMethod(),
            pre_normalization=SoftmaxNormalization()
        ),
        '''
