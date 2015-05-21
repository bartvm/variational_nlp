import logging

from blocks.algorithms import GradientDescent, Scale, CompositeRule
from blocks.dump import load_parameter_values
from blocks.dump import MainLoopDumpManager
from blocks.extensions import Printing
from blocks.extensions.monitoring import (DataStreamMonitoring,
                                          TrainingDataMonitoring)
from blocks.extensions.training import SharedVariableModifier
from blocks.graph import ComputationGraph
from blocks.main_loop import MainLoop
from blocks.model import Model
from theano import tensor


from variational import VariationalInference, VARIATIONAL_COST

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)


class Decreaser(object):
    def __init__(self, scalar):
        self.scalar = scalar

    def __call__(self, t, x):
        return x * self.scalar


class LearningRateHalver(SharedVariableModifier):
    def __init__(self, record_name, lr, scalar, **kwargs):
        self.record_name = record_name
        self.lr = lr
        self.prev_value = float('inf')
        super(LearningRateHalver, self).__init__(self.lr,
                                                 Decreaser(scalar).__call__,
                                                 **kwargs)

    def do(self, which_callback, *args):
        log = self.main_loop.log
        self.cur_value = log.current_row[self.record_name]
        if self.cur_value > self.prev_value:
            super(LearningRateHalver, self).do(which_callback, *args)
        self.prev_value = self.cur_value


def train_model(cost, train_stream, valid_stream, freq_likelihood,
                sigmas=None, num_batches=None, load_location=None,
                save_location=None, learning_rate=0.1):
    cost.name = 'nll'
    perplexity = 2 ** (cost / tensor.log(2))
    perplexity.name = 'ppl'

    # Define the model
    model = Model(cost)

    # Load the parameters from a dumped model
    if load_location is not None:
        logger.info('Loading parameters...')
        model.set_param_values(load_parameter_values(load_location))

    # Set up the training procedure
    cg = ComputationGraph(cost)
    if VARIATIONAL_COST in cost.tag.roles:
        if sigmas is None:
            raise ValueError('need sigmas to train variational cost')
        step_rule = CompositeRule([VariationalInference(cg.outputs[0], sigmas,
                                                        num_batches,
                                                        learning_rate),
                                   Scale(learning_rate=learning_rate)])
        lr = step_rule.components[1].learning_rate
    else:
        step_rule = Scale(learning_rate=learning_rate)
        lr = step_rule.learning_rate

    lr.name = "learning_rate"
    monitor_lr = TrainingDataMonitoring([lr], after_epoch=True)
    lr_halver = LearningRateHalver("valid_nll", lr, 0.9, after_epoch=True,
                                   after_batch=False)
    algorithm = GradientDescent(cost=cost, step_rule=step_rule,
                                params=cg.parameters)
    main_loop = MainLoop(
        model=model,
        data_stream=train_stream,
        algorithm=algorithm,
        extensions=[
            DataStreamMonitoring([freq_likelihood, cost, perplexity],
                                 valid_stream, prefix='valid'),
            monitor_lr,
            lr_halver,
            Printing()
        ]
    )
    main_loop.run()

    # Save the main loop
    if save_location is not None:
        logger.info('Saving the main loop...')
        dump_manager = MainLoopDumpManager(save_location)
        dump_manager.dump(main_loop)
        logger.info('Saved')
