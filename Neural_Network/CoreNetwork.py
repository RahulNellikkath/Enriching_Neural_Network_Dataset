
import torch
from torch import nn
from collections import OrderedDict


class Normalise(nn.Module):
    def __init__(self, n_neurons):
        super(Normalise, self).__init__()
        self.mean = nn.Parameter(data=torch.zeros(n_neurons), requires_grad=False)
        self.standard_deviation = nn.Parameter(data=torch.ones(n_neurons), requires_grad=False)
        self.eps = 1e-8

    def forward(self, input):
        return (input - self.mean) / (self.standard_deviation + self.eps)

    def set_normalisation(self, mean, standard_deviation):
        if not len(standard_deviation.shape) == 1 or not len(mean.shape) == 1:
            raise Exception('Input statistics are not 1-D tensors.')

        if not torch.nonzero(self.standard_deviation).shape[0] == standard_deviation.shape[0]:
            raise Exception('Standard deviation in normalisation contains elements equal to 0.')

        self.mean = nn.Parameter(data=mean, requires_grad=False)
        self.standard_deviation = nn.Parameter(data=standard_deviation, requires_grad=False)


class Denormalise(nn.Module):
    def __init__(self, n_neurons):
        super(Denormalise, self).__init__()
        self.mean = nn.Parameter(data=torch.zeros(n_neurons), requires_grad=False)
        self.standard_deviation = nn.Parameter(data=torch.ones(n_neurons), requires_grad=False)
        self.eps = 1e-8

    def forward(self, input):
        return self.mean + input * self.standard_deviation

    def set_normalisation(self, mean, standard_deviation):
        if not len(standard_deviation.shape) == 1 or not len(mean.shape) == 1:
            raise Exception('Input statistics are not 1-D tensors.')

        if not torch.nonzero(self.standard_deviation).shape[0] == standard_deviation.shape[0]:
            raise Exception('Standard deviation in normalisation contains elements equal to 0.')

        self.mean = nn.Parameter(data=mean, requires_grad=False)
        self.standard_deviation =  nn.Parameter(data=standard_deviation, requires_grad=False)


class NeuralNetwork(torch.nn.Module):
    def __init__(self,n_input_neurons, n_output_neurons, hidden_layer_size, n_hidden_layers, pytorch_init_seed):
        super(NeuralNetwork, self).__init__()

        if type(pytorch_init_seed) is int:
            torch.manual_seed(pytorch_init_seed)

        self.n_input_neurons = n_input_neurons
        self.n_output_neurons = n_output_neurons
        self.n_hidden_layers = n_hidden_layers
        self.hidden_layer_size = hidden_layer_size

        neurons_in_layers = [self.n_input_neurons] + [hidden_layer_size] * n_hidden_layers + [self.n_output_neurons]
        activation_function = 'ReLU'
        layer_normalisation = False

        layer_dictionary = OrderedDict()

        layer_dictionary['input_normalisation'] = Normalise(self.n_input_neurons)

        for ii, (neurons_in, neurons_out) in enumerate(zip(neurons_in_layers[:-2], neurons_in_layers[1:-1])):
            layer_dictionary[f'dense_{ii}'] = nn.Linear(in_features=neurons_in,
                                                        out_features=neurons_out,
                                                        bias=True)

            if activation_function == "ReLU":
                if layer_normalisation:
                    layer_dictionary[f'layer_norm_{ii}'] = nn.LayerNorm(neurons_out)
                layer_dictionary[f'activation_{ii}'] = nn.ReLU()
                nn.init.kaiming_normal_(layer_dictionary[f'dense_{ii}'].weight, mode='fan_in',
                                        nonlinearity='relu')

            elif activation_function == "Tanh":
                layer_dictionary[f'activation_{ii}'] = nn.Tanh()
                if layer_normalisation:
                    layer_dictionary[f'layer_norm_{ii}'] = nn.LayerNorm(neurons_out)
                nn.init.kaiming_normal_(layer_dictionary[f'dense_{ii}'].weight, mode='fan_in',
                                        nonlinearity='tanh')
            else:
                raise Exception('Enter valid activation function! (ReLU or Tanh)')

        layer_dictionary['output_layer'] = nn.Linear(in_features=neurons_in_layers[-2],
                                                     out_features=neurons_in_layers[-1],
                                                     bias=True)
        nn.init.xavier_normal_(layer_dictionary['output_layer'].weight, gain=1.0)

        layer_dictionary['output_de_normalisation'] = Denormalise(self.n_output_neurons)

        self.dense_layers = nn.Sequential(layer_dictionary)

    def normalise_input(self, input_statistics):
        self.dense_layers.input_normalisation.set_normalisation(mean=input_statistics[1],
                                                                standard_deviation=input_statistics[0])

    def normalise_output(self, output_statistics):
            self.dense_layers.output_de_normalisation.set_normalisation(mean=output_statistics[1],
                                                                    standard_deviation=output_statistics[0])

    def forward(self, x):
        return self.dense_layers(x)



