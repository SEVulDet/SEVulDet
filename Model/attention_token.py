import math
import torch
import torch.nn as nn
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class AttentionToken(nn.Module):
    def __init__(self, embed_dim):
        super(AttentionToken, self).__init__()
        self.embed_dim = embed_dim
        self.weight_W_word = nn.Parameter(torch.Tensor(embed_dim, embed_dim))
        self.bias_word = nn.Parameter(torch.Tensor(embed_dim, 1))
        self.weight_proj_word = nn.Parameter(torch.Tensor(embed_dim, 1))
        self.softmax = nn.Softmax(dim=1)
        self.weight_W_word.data.uniform_(-0.1, 0.1)
        self.weight_proj_word.data.uniform_(-0.1, 0.1)

    def forward(self, input):
        batch_size = input.size(1)
        word_squish = self.batch_matmul_bias(input, self.weight_W_word, self.bias_word).view(-1, batch_size,
                                                                                             self.embed_dim)
        word_attn = self.batch_matmul(word_squish, self.weight_proj_word).view(-1, batch_size)
        # print(word_attn)
        word_attn_norm = self.softmax(word_attn.transpose(1, 0))
        # print(word_attn_norm.shape, word_attn_norm)
        input_atted = self.attention_mul(input, word_attn_norm)
        return input_atted

    def batch_matmul_bias(self, seq, weight, bias):
        s = None
        bias_dim = bias.size()
        for i in range(seq.size(0)):
            _s = torch.mm(seq[i], weight)
            _s_bias = _s + bias.expand(bias_dim[0], _s.size()[0]).transpose(0, 1)
            _s_bias = torch.tanh(_s_bias)
            _s_bias = _s_bias.unsqueeze(0)
            if s is None:
                s = _s_bias
            else:
                s = torch.cat((s, _s_bias), 0)
        return s

    def batch_matmul(self, seq, weight):
        s = None
        for i in range(seq.size(0)):
            _s = torch.mm(seq[i], weight)
            _s = torch.tanh(_s)
            _s = _s.unsqueeze(0)
            if s is None:
                s = _s
            else:
                s = torch.cat((s, _s), 0)
        return s

    def attention_mul(self, input, att_weights):
        input = input.permute(1, 0, 2)
        att_weights = att_weights.unsqueeze(2).repeat(1, 1, 30)
        attn_vectors = input.mul(att_weights)
        return torch.cat((input.unsqueeze(1), attn_vectors.unsqueeze(1)), dim=1)