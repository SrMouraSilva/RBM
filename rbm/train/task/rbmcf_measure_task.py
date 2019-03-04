from rbm.predictor.expectation.expectation_predictor import NormalizationRoundingMethod
from rbm.predictor.expectation.rbmcf_expectation_predictor import RBMCFExpectationPredictor
from rbm.predictor.topk.rbm_top_k_predictor import RBMTop1Predictor, RBMTopKPredictor
from rbm.train.task.rbm_base_measure_task import RBMBaseMeasureTask


class RBMCFMeasureTask(RBMBaseMeasureTask):

    @property
    def shape_softmax(self):
        return self.model.shape_softmax

    @property
    def rating_size(self):
        return self.model.rating_size

    @property
    def movie_size(self):
        return self.model.movie_size

    @property
    def predictors(self):
        return {
            'top-1': RBMTop1Predictor(self.model, self.movie_size, self.rating_size),
            'top-5': RBMTopKPredictor(self.model, self.movie_size, self.rating_size, k=5),
            #'top-50': RBMTopKPredictor(self.model, self.movie_size, self.rating_size, k=50),
            #'expectation/round': RBMCFExpectationPredictor(
            #    self.model, self.movie_size, self.rating_size, normalization=RoundMethod()
            #),
            #'expectation/normalized': RBMCFExpectationPredictor(
            #    self.model, self.movie_size, self.rating_size, normalization=NormalizationRoundingMethod()
            #),
        }
