import torch
import torch.nn as nn
from torch.autograd import Variable


class bilstm(torch.nn.Module):
    def __init__(self, batch_size, output_size, hidden_size, num_layers, embed_dim, bidirectional, dropout,
                 use_cuda, sequence_length):
        super(bilstm, self).__init__()
        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.embed_dim = embed_dim
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.use_cuda = use_cuda
        self.sequence_length = sequence_length
        self.layer_size = num_layers
        self.lstm_forward = nn.LSTM(self.embed_dim, self.hidden_size, self.layer_size, dropout=self.dropout,
                                    bidirectional=False)
        self.lstm_backward = nn.LSTM(self.embed_dim, self.hidden_size, self.layer_size, dropout=self.dropout,
                                     bidirectional=False)
        self.num_directions = 2 if bidirectional else 1
        self.dense1 = nn.Linear(self.hidden_size * self.num_directions * self.sequence_length, self.sequence_length)
        self.dense2 = nn.Linear(self.sequence_length, self.output_size)

    def reverse_sequence(self, tensor):
        idx = [i for i in range(tensor.size(0) - 1, -1, -1)]
        idx = torch.LongTensor(idx).cuda()
        tensor_reversed = tensor.index_select(0, idx)
        del tensor
        return tensor_reversed

    def self_attention(self, att_input):  # [batch_size, seq_len, hidden_size*2]
        weight = torch.softmax(torch.div(torch.sum(att_input.mul(att_input), dim=2), self.hidden_size ** 0.5), dim=1)
        weight = weight.reshape([self.batch_size, -1, 1]).repeat(1, 1, self.hidden_size * 2)
        att_output = att_input.mul(weight)
        return att_output  # [batch_size, seq_len, hidden_size*2]

    def forward(self, input_tensor):
        input_tensor = input_tensor.permute(1, 0, 2)
        if self.use_cuda:
            h_0_f = Variable(torch.zeros(self.layer_size, self.batch_size, self.hidden_size).cuda())
            c_0_f = Variable(torch.zeros(self.layer_size, self.batch_size, self.hidden_size).cuda())
            h_0_b = Variable(torch.zeros(self.layer_size, self.batch_size, self.hidden_size).cuda())
            c_0_b = Variable(torch.zeros(self.layer_size, self.batch_size, self.hidden_size).cuda())
        else:
            h_0_f = Variable(torch.zeros(self.layer_size, self.batch_size, self.hidden_size))
            c_0_f = Variable(torch.zeros(self.layer_size, self.batch_size, self.hidden_size))
            h_0_b = Variable(torch.zeros(self.layer_size, self.batch_size, self.hidden_size))
            c_0_b = Variable(torch.zeros(self.layer_size, self.batch_size, self.hidden_size))
        h_n_f, _ = self.lstm_forward(input_tensor, (h_0_f, c_0_f))
        input_reversed = self.reverse_sequence(input_tensor)
        h_n_b, _ = self.lstm_backward(input_reversed, (h_0_b, c_0_b))
        h_n_b_reverse = self.reverse_sequence(h_n_b)
        bilstm_output = torch.cat((h_n_f, h_n_b_reverse), dim=2).permute(1, 0, 2)  # [ 16, 100, 512]
        att_output = self.self_attention(bilstm_output)
        logits_dense1 = torch.relu(self.dense1(att_output.reshape(int(self.batch_size), -1)))
        logits_dense2 = self.dense2(logits_dense1)
        self.logits_dense1 = logits_dense1
        return logits_dense2
