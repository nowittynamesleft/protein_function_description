import torch
import torch.nn.functional as F
from torch import nn


class ProtEmbedding(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim=0):
        super(ProtEmbedding, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        print('input dim:', input_dim)
        print('output dim:', output_dim)

        self.conv_layer_1 = nn.Conv1d(self.input_dim, self.hidden_dim, 5)
        self.conv_layer_2 = nn.Conv1d(self.hidden_dim, self.hidden_dim, 5)
        self.linear = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x_in):
        x = F.relu(self.conv_layer_1(x_in))
        x = F.relu(self.conv_layer_2(x))
        x, _ = torch.max(x, -1)
        x_out = self.linear(x)

        return x_out
    

class KeywordEmbedding(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim=100):
        super(KeywordEmbedding, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        print('vocab size:', input_dim)
        print('output dim:', output_dim)

        self.word_embed = nn.Embedding(self.input_dim, self.hidden_dim)
        self.linear = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, keyword_tensor_list):
        batch_word_embeds = []
        for words_tensor in keyword_tensor_list:
            curr_word_embeds = self.word_embed(words_tensor)
            combined_word_embeds = torch.mean(curr_word_embeds, 0)
            batch_word_embeds.append(combined_word_embeds)

        batch_word_embeds = torch.stack(batch_word_embeds)
        x_out = self.linear(batch_word_embeds)

        return x_out


class seqCLIP(nn.Module):

    def __init__(self, prot_alphabet_dim, keyword_vocab_size, embed_dim):
        super(seqCLIP, self).__init__()

        self.prot_alphabet_dim = prot_alphabet_dim
        self.embed_dim = embed_dim
        self.keyword_vocab_size = keyword_vocab_size

        print('prot alphabet size:', prot_alphabet_dim)
        print('vocab size:', keyword_vocab_size)
        print('embed dim:', embed_dim)

        self.prot_embed = ProtEmbedding(self.prot_alphabet_dim, self.embed_dim, hidden_dim=100)
        self.keyword_embed = KeywordEmbedding(self.keyword_vocab_size, self.embed_dim)
        self.temperature = nn.Parameter(torch.tensor(0.07))
        self.temperature.requires_grad = True

    def forward(self, x_prot, x_keyword_list):
        x_out_prot = self.prot_embed(x_prot)
        x_out_keyword = self.keyword_embed(x_keyword_list)

        return x_out_prot, x_out_keyword
