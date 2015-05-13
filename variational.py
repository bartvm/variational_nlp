import numpy
from collections import OrderedDict

from blocks.algorithms import StepRule
from blocks.graph import ComputationGraph
from blocks.roles import add_role, ParameterRole, CostRole
from blocks.utils import shared_floatx
from theano import tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams


class VarianceRole(ParameterRole):
    pass


class VariationalCostRole(CostRole):
    pass



VARIANCE = VarianceRole()
VARIATIONAL_COST = VariationalCostRole()


def make_variational_model(cost, init_sigma=0.1):
    # Consider the weights to be the means, create variances
    cg = ComputationGraph(cost)
    sigmas = {}
    for param in cg.parameters:
        sigmas[param] = shared_floatx(
            numpy.ones_like(param.get_value()) * init_sigma,
            name=param.name + '_sigma'
        )
        add_role(sigmas[param], VARIANCE)

    # Replace weights with samples from Gaussian
    rng = MRG_RandomStreams()
    new_cg = cg.replace({param: rng.normal(param.shape, param, sigmas[param])
                         for param in cg.parameters})
    new_cost = new_cg.outputs[0]
    add_role(new_cost, VARIATIONAL_COST)
    return new_cost, sigmas


class VariationalInference(StepRule):
    def __init__(self, cost, sigmas, num_batches):
        self.cost = cost
        self.sigmas = sigmas
        self.num_batches = num_batches

    def compute_steps(self, previous_steps):
        # previous_steps contains parameters and their gradients
        params = self.sigmas.keys()

        # Create mu and sigma for prior
        mu = shared_floatx(0, name='mu')
        sigma = shared_floatx(0.1, name='sigma')

        # Optimal values for the prior
        N = numpy.array(sum([param.get_value().size for param in params]),
                        dtype='float32')  # Else mean_param is float64
        mean_param = tensor.sum([param.sum() for param in params]) / N
        update_mu = (mu, mean_param)
        update_sigma = (sigma, tensor.sum([tensor.sum(self.sigmas[param] +
                        tensor.sqr(param - mean_param)) for param in params]) /
                        N)

        # Update parameters using gradient + regularization
        steps = OrderedDict(
            [(param, (param - mu) / sigma / self.num_batches +
              previous_steps[param])
             for param in params])

        # Update variance based on cost
        # NOTE: Sigma is actually sigma^2
        sigma_error_losses = {param: 0.5 * tensor.sqr(grad)
                              for param, grad in previous_steps.items()}
        steps.update(OrderedDict([(
            self.sigmas[param],
            0.5 * (1 / sigma - 1 / self.sigmas[param]) /
            self.num_batches +
            sigma_error_losses[param]) for param in params]))

        return steps, [update_mu, update_sigma]
