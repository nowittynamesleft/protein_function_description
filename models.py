import torch
import torch.nn.functional as F
from torch import nn
import pytorch_lightning as pl


class LitSeqCLIP(pl.LightningModule):

    def __init__(self, prot_alphabet_dim, keyword_vocab_size, embed_dim, learning_rate=0.01):
        super(LitSeqCLIP, self).__init__()

        self.prot_alphabet_dim = prot_alphabet_dim
        self.embed_dim = embed_dim
        self.keyword_vocab_size = keyword_vocab_size
        self.lr = learning_rate

        print('prot alphabet size:', prot_alphabet_dim)
        print('vocab size:', keyword_vocab_size)
        print('embed dim:', embed_dim)

        self.prot_embed = ProtEmbedding(self.prot_alphabet_dim, self.embed_dim, hidden_dim=100)
        self.keyword_embed = KeywordEmbedding(self.keyword_vocab_size, self.embed_dim)
        self.temperature = nn.Parameter(torch.tensor(0.07))
        self.temperature.requires_grad = True
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x_prot, x_keyword_list):
        x_out_prot = self.prot_embed(x_prot)
        x_out_keyword = self.keyword_embed(x_keyword_list)

        return x_out_prot, x_out_keyword


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def training_step(self, train_batch, batch_idx):
        batch_seqs, batch_keywords = train_batch

        seq_embeds = self.prot_embed(batch_seqs)
        keyword_embeds = self.keyword_embed(batch_keywords) # keyword_embeds (Nxk) are averaged over all assigned keywords for a protein
        seq_embeds = nn.functional.normalize(seq_embeds)
        keyword_embeds = nn.functional.normalize(keyword_embeds)

        curr_batch_size, embed_dim = seq_embeds.size()
        similarity = torch.mm(seq_embeds, keyword_embeds.transpose(0,1)).squeeze()
        similarity *= torch.exp(self.temperature)

        labels = torch.arange(start=0, end=similarity.shape[0], dtype=torch.long).to(self.device)
        #labels = torch.arange(start=0, end=similarity.shape[0], dtype=torch.long)
        loss = (self.loss_fn(similarity, labels) + self.loss_fn(similarity.transpose(0,1), labels))/2
        self.log('train_loss', loss)
        return {'loss': loss, 'seq_embeds': seq_embeds, 'keyword_embeds': keyword_embeds}

    def validation_step(self, val_batch, batch_idx):
        batch_seqs, batch_keywords = val_batch

        seq_embeds = self.prot_embed(batch_seqs)
        keyword_embeds = self.keyword_embed(batch_keywords) # keyword_embeds (Nxk) are averaged over all assigned keywords for a protein
        seq_embeds = nn.functional.normalize(seq_embeds)
        keyword_embeds = nn.functional.normalize(keyword_embeds)

        curr_batch_size, embed_dim = seq_embeds.size()
        similarity = torch.mm(seq_embeds, keyword_embeds.transpose(0,1)).squeeze()
        similarity *= torch.exp(net.temperature)

        labels = torch.arange(start=0, end=similarity.shape[0], dtype=torch.long).to(self.device)
        #labels = torch.arange(start=0, end=similarity.shape[0], dtype=torch.long)
        loss = (self.loss_fn(similarity, labels) + self.loss_fn(similarity.transpose(0,1), labels))/2
        self.log('val_loss', loss)

    def predict_step(self, pred_batch, batch_idx):
        batch_seqs, _ = pred_batch

        seq_embeds = self.prot_embed(batch_seqs)

        # get only individual keyword embeddings for predict step
        total_keyword_embeds = []
        for i in range(0, self.keyword_vocab_size):
            ind = torch.tensor(i, dtype=torch.long).unsqueeze(0).to(self.device)
            keyword_embed = self.keyword_embed([ind])
            keyword_embeds = nn.functional.normalize(keyword_embed)
            total_keyword_embeds.append(keyword_embeds)
        keyword_embeds = torch.cat(total_keyword_embeds, dim=0)
        seq_embeds = nn.functional.normalize(seq_embeds)

        return seq_embeds, keyword_embeds


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