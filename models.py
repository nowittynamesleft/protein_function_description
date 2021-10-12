import torch
import torch.nn.functional as F
from torch import nn
import pytorch_lightning as pl
from layers import MultiheadAttention, EncoderBlock, TransformerEncoder, PositionalEncoding

from layers import EncoderBlock, PositionalEncoding


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
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
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
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

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


class TransformerProtEmbedding(nn.Module):

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


class PretrainedProtBERTClassifier(nn.Module):
    def __init__(self, pretrained_model, output_dim, freeze_pretrained=False, freeze_partial=False):
        super(PretrainedProtBERTClassifier, self).__init__()
        self.pretrained_model = pretrained_model
        self.freeze_pretrained = freeze_pretrained
        self.length_transform = LengthConverter()
        if freeze_partial:
            for name, param in self.pretrained_model.named_parameters():
                if 'norm' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

    def forward(self, x, seq_masks):
        repr_layer = 5 # layer to extract representations from
        if self.freeze_pretrained:
            with torch.no_grad():
                rep = self.pretrained_model(x, repr_layers=[repr_layer])['representations'][repr_layer]
        else:
            rep = self.pretrained_model(x, repr_layers=[repr_layer])['representations'][repr_layer]

        target_lens = 1000*torch.ones(rep.shape[0])
        embeddings = self.length_transform(rep, target_lens, seq_masks)
        return embeddings


class LengthConverter(nn.Module):
    """
    Implementation of Length Transformation. From Shu et al 2019
    """

    def __init__(self):
        super(LengthConverter, self).__init__()
        self.sigma = nn.Parameter(torch.tensor(1., dtype=torch.float), requires_grad=True)

    def forward(self, z, ls, z_mask):
        """
        Adjust the number of vectors in `z` according to `ls`.
        Return the new `z` and its mask.
        Args:
            z - latent variables, shape: B x L_x x hidden
            ls - target lengths, shape: B
            z_mask - latent mask, shape: B x L_x
        """
        n = z_mask.sum(1)
        arange_l = torch.arange(ls.max().long())
        arange_z = torch.arange(z.size(1))
        if torch.cuda.is_available():
            arange_l = arange_l.cuda()
            arange_z = arange_z.cuda()
        arange_l = arange_l[None, :].repeat(z.size(0), 1).float()
        mu = arange_l * n[:, None].float() / ls[:, None].float()
        arange_z = arange_z[None, None, :].repeat(z.size(0), ls.max().long(), 1).float()

        # assuming else statement..
        distance = torch.clamp(arange_z - mu[:, :, None], -100, 100)
        logits = - torch.pow(2, distance) / (2. * self.sigma ** 2)
        '''
        if OPTS.fp16:
            arange_l = arange_l.half()
            mu = mu.half()
            arange_z = arange_z.half()
        if OPTS.fixbug1:
            logits = - torch.pow(arange_z - mu[:, :, None], 2) / (2. * self.sigma ** 2)
        else:
            distance = torch.clamp(arange_z - mu[:, :, None], -100, 100)
            logits = - torch.pow(2, distance) / (2. * self.sigma ** 2)
        '''
        logits = logits * z_mask[:, None, :] - 999. * (1 - z_mask[:, None, :])
        weight = torch.softmax(logits, 2)
        # z_prime = (z[:, None, :, :] * weight[:, :, :, None]).sum(2) # use torch.multiply instead, to do dot product between two matrices
        z_prime = torch.matmul(weight, z)
        """
        if OPTS.fp16:
            z_prime_mask = (arange_l < ls[:, None].half()).half()
        else:
            z_prime_mask = (arange_l < ls[:, None].float()).float()
        """
        z_prime_mask = (arange_l < ls[:, None].float()).float()
        z_prime = z_prime * z_prime_mask[:, :, None]
        return z_prime, z_prime_mask


class NMTDescriptionGen(nn.Module):
    def __init__(self, prot_alphabet_dim, vocab_size, embed_dim, num_heads=8, dim_feedforward=512, max_len=1000):
        super(NMTDescriptionGen, self).__init__()

        self.prot_alphabet_dim = prot_alphabet_dim
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size

        print('prot alphabet size:', prot_alphabet_dim)
        print('vocab size:', vocab_size)
        print('embed dim:', embed_dim)

        # pre-trained model
        # assumes block of: Embedding + Positinoal Encoding + Self-Attention
        self.embedding_layer = nn.Embedding(self.prot_alphabet_dim + 1, embed_dim)
        self.prot_embed = EncoderBlock(embed_dim, num_heads, dim_feedforward)

        # pos encoding
        self.pos_encoding = PositionalEncoding(embed_dim, max_len=max_len)

        # len transform
        self.len_convert = LengthConverter()

        # decoder
        self.decoder = EncoderBlock(embed_dim, num_heads, dim_feedforward)

    def _max_lens(self, x_mask):
        ls = torch.max(torch.sum(x_mask.float(), dim=1))*torch.ones(x_mask.shape[0])
        return ls

    def forward(self, x, x_mask):
        # x: Tensor: [seq_set, L]
        # x_mask: Tensor: [seq_set, L]
        print(x.shape)
        #x = torch.transpose(x, 1, 2)
        #print(x.shape)
        embed = torch.zeros(x.shape[0], x.shape[1], x.shape[2], self.embed_dim)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                print('Sample ' + str(j) + ' of set ' + str(i))
                print(x[i, j, :].shape)
                embed[i, j, ...] = self.embedding_layer(x[i, j, :].type(torch.long))
                embed[i, j, ...] = self.prot_embed(embed[i, j, ...], x_mask[i, j, :])  # Tensor: [seq_set, L, embed_dim]
        embed_len_trans = []
        for i in range(x.shape[0]):
            curr_sample_embed = embed[i, ...]
            curr_sample_max_len = int(torch.max(torch.sum(x_mask[i, ...], dim=1)).item())
            print(curr_sample_max_len)
            embed_len_trans.append(torch.zeros((curr_sample_embed.shape[0], curr_sample_max_len, curr_sample_embed.shape[2])))
            embed_len_trans[-1] = self.len_convert(embed[i, ...], self._max_lens(x_mask[i, ...]), x_mask[i, ...]) # ls: Tensor: []
        avg_embed = [torch.mean(embed[0], dim=0) for embed in embed_len_trans]
        pos_encode_embed = [self.pos_encoding(embed) for embed in avg_embed] # Tensor: [1, L_max, embed_dim]

        out = [self.decoder(embed) for embed in pos_encode_embed]

        return out
