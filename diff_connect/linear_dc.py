import torch
from torch import nn

from torch.nn import Linear
from torch.nn import init
from torch.nn import functional

def make_binary_labels(n):
    matrix = [[int(s) for s in "{0:0{1}b}".format(i, n)] for i in range(2**n)]
    return torch.FloatTensor(matrix)

class LinearDC(nn.Module):
    def __init__(self, num_input, num_output, num_binary_labels, state_space_size):
        super().__init__()

        #Switched "Tensor" to 'randn' in a few cases as 'tensor' was giving nans when running in pynb/colab

        self.num_input = num_input
        self.num_output = num_output
        self.num_units = 2**num_binary_labels
        self.num_hidden = self.num_units - num_input - num_output

        assert(self.num_hidden >= 0)

        self.num_binary_labels = num_binary_labels
        self.state_space_size = state_space_size

        self.embedding = Linear(num_binary_labels, state_space_size)
        self.kernel = nn.Parameter(torch.randn(state_space_size, state_space_size)) #Note: may switich randn back to Tensor later?

        # note this is an non-meta parameter: it is just an ordinary bias vector learned by sgd
        # in the usual way
        self.bias1 = nn.Parameter(torch.randn(self.num_hidden)) #Note: may switich randn back to Tensor later?
        self.bias2 = nn.Parameter(torch.randn(self.num_output)) #Note: may switich randn back to Tensor later?

        # pre-set up label matrices for speed
        binary_label_matrix = make_binary_labels(num_binary_labels)
        input_label_matrix = binary_label_matrix[0:self.num_input]
        hidden_label_matrix = binary_label_matrix[self.num_input:-self.num_output]
        output_label_matrix = binary_label_matrix[-self.num_output:]
        self.register_buffer('input_label_matrix', input_label_matrix)
        self.register_buffer('hidden_label_matrix', hidden_label_matrix)
        self.register_buffer('output_label_matrix', output_label_matrix)

        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_normal_(self.kernel, gain=0.01)
        init.xavier_normal_(self.embedding.weight, gain=0.01)
        init.uniform_(self.bias1, 0, 0)
        init.uniform_(self.bias2, 0, 0)

    def compute_weights(self, src, dst):
        return torch.matmul(torch.matmul(dst, self.kernel), torch.t(src))

    def forward(self, input):
        assert len(input.shape) == 2 and input.shape[1] == self.num_input, \
            "input should be matrix of shape (_, %d) but was %s instead" % (self.num_input, input.shape)

        input_embedded = torch.tanh(self.embedding(self.input_label_matrix))
        hidden_embedded = torch.tanh(self.embedding(self.hidden_label_matrix))
        output_embedded = torch.tanh(self.embedding(self.output_label_matrix))

        input_to_hidden = self.compute_weights(input_embedded, hidden_embedded)
        hidden_to_output = self.compute_weights(hidden_embedded, output_embedded)

        hidden_activations = torch.relu(functional.linear(input, input_to_hidden, self.bias1))
        return functional.linear(hidden_activations, hidden_to_output, self.bias2)

    def extra_repr(self):
        return 'num_input={}, num_hidden={}, num_output={}, state_space_size={}'.format(
            self.num_input, self.num_hidden, self.num_output, self.state_space_size
        )
