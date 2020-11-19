import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import string
import re
import collections
import numpy as np

random.seed(0)


def prepare_sequence(lst, idx_mapping):
    """ 
    Map individual items from `sent` using `idx_mapping`
    Return value is the same length as `sent`
    Usage: 
        >> prepare_sequence(['a', 'b', 'c'], {'a':0, 'b':1, 'c':2})
        [0, 1, 2]
    """
    idxs = []
    for item in lst:
        if item not in idx_mapping:
            assert '<unk>' in idx_mapping or 0 in idx_mapping, "cannot map unknown token:" + item
            if '<unk>' in idx_mapping:
                idxs.append(idx_mapping['<unk>'])
            else:
                idxs.append(idx_mapping[0])
        else:
            idxs.append(idx_mapping[item])
    try:
        return torch.tensor(idxs, dtype=torch.long)
    except:
        return idxs


class LSTMEncoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, char_embedding_dim,
                 char_hidden_dim, char_size, vocab_size, lstm_layers=1,
                 bidirectional=False, dropout=0):
        super(LSTMEncoder, self).__init__()
        #############################################################################
        # TODO: Define and initialize anything needed for the forward pass.
        # You are required to create a model with:
        # embedding layer: that maps words to the embedding space
        # char embedding layer: maps chars to embedding space 
        # an char level LSTM: that finds the character level embedding for a word
        # an LSTM layer: that takes the combined embeddings as input and outputs hidden states
        # Remember, this needs to be done for both context and query (our input)
        # (DO NOT apply bidirectionality to character LSTMs)
        #############################################################################
        # Encoder
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.char_embedding = nn.Embedding(char_size, char_embedding_dim)
        self.context_lstm = nn.LSTM(embedding_dim + char_hidden_dim, hidden_dim, num_layers=lstm_layers,
                                    dropout=dropout, bidirectional=bidirectional)
        self.query_lstm = nn.LSTM(embedding_dim + char_hidden_dim, hidden_dim, num_layers=lstm_layers,
                                    dropout=dropout, bidirectional=bidirectional)
        self.char_lstm = nn.LSTM(char_embedding_dim, char_hidden_dim)
        for param in self.parameters():
            param.requires_grad = True
        # Remember: bidirectional makes the output hidden_dim * 2

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

    def segment_word(self, chars):
        buffer = []
        new_chars = []
        words = []
        for i in range(len(chars)):
            if chars[i] != 1:
                buffer.append(chars[i].item())
            else:
                if len(buffer):
                    new_chars.extend(buffer)
                    words.append(buffer)
                    buffer = []
                    new_chars.append(chars[i].item())
        return words

    def forward(self, context, context_chars, query, query_chars):
        lstm_context_vectors = None  # for each word
        lstm_query_vectors = None  # for each word
        #############################################################################
        # TODO: Implement the forward pass.
        # Given a tokenized index-mapped sentence and a character sequence as the arguments,
        # `context` and `query` are word sequences at are index mapped
        # `context_chars` and `query_chars` are char sequence of each word that are
        #   index mapped
        # The output is very similar to HW4.
        # Return values:
        #   `lstm_query_vectors` : Txd or Nx(d*2) if bidirectional LSTM used
        #       T is # of words in context, and d is size of hidden states
        #   `lstm_query_vectors` : Jxd or Jx(d*2) if bidirectional LSTM used
        #       J is # of words in query, and d is size of hidden states
        #
        #############################################################################
        # Context
        context_embed_out = self.embedding(context)
        context_char_embed_out = self.char_embedding(context_chars)
        context_char_embed_out = self.char_lstm(context_char_embed_out.unsqueeze(0))[0].squeeze(0)
        # concatenation
        word_len = int(context_chars.shape[0] / context.shape[0])
        list_char_embed = [torch.mean(context_char_embed_out[i * word_len: (i + 1) * word_len, :], dim=0)
                   for i in range(context_embed_out.shape[0])]
        char_embeds = torch.stack(list_char_embed, dim=0)
        context_embed_out = torch.cat((context_embed_out, char_embeds), dim=1).unsqueeze(0)
        # print(context_embed_out.shape)
        lstm_context_vectors = self.context_lstm(context_embed_out)[0].squeeze(0)

        # Query
        query_embed_out = self.embedding(query)
        query_char_embed_out = self.char_embedding(query_chars)
        query_char_embed_out = self.char_lstm(query_char_embed_out.unsqueeze(0))[0].squeeze(0)
        list_char_embed = [torch.mean(query_char_embed_out[i * word_len: (i + 1) * word_len, :], dim=0)
                           for i in range(query_embed_out.shape[0])]
        char_embeds = torch.stack(list_char_embed, dim=0)
        query_embed_out = torch.cat((query_embed_out, char_embeds), dim=1).unsqueeze(0)
        lstm_query_vectors = self.query_lstm(query_embed_out)[0].squeeze(0)
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return lstm_context_vectors, lstm_query_vectors


class AttentionFlow(nn.Module):
    def __init__(self, w_dim):
        """
        w_dim : is the same as 6d in the paper. Should be 6*hidden_dim if bidirectionality is True
        """
        #############################################################################
        # TODO: Define and initialize anything needed for the forward pass.
        # For this part, we need a linear layer to compute the similarity matrix (no bias according to the paper)
        #############################################################################
        super(AttentionFlow, self).__init__()
        self.sim_weight = nn.Linear(w_dim, 1, bias=False)
        self.softmax = nn.Softmax()
        for param in self.parameters():
            param.requires_grad = True
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

    def forward(self, context, query):
        G = None
        #############################################################################
        # T : number of tokens in context
        # J : number of tokens in query
        # d : hidden_dimensions (context and query will have d*2 hidden dimensions if using bidirectional LSTM)
        # Parameters (from the encoder):
        #     context: of size Txd or (d*2) if using bidirectional LSTM
        #     query: of size Jxd or (d*2) if using bidirectional LSTM
        # For this part, you need to compute a similarity matrix, S
        #     S : TxJ
        # then use S to build context2query and query2context attention
        #     context2query : T x (d*2)
        #     query2context : T x (d*2)
        #         hint: query2context will be (1, d*2) but it will need to be repeated
        #               T times so the dimension is T x (d*2)
        # Return :: G which is the query aware context vectors of size (T x d*8)
        #     G is obtained by combining `context`, `context2query` and `query2context`
        #       as defined in the paper.
        #############################################################################
        S = np.zeros((context.shape[0], query.shape[0]))
        for t in range(context.shape[0]):
            for j in range(query.shape[0]):
                # print(context[t], context[t].shape)
                # print(query[j], query[j].shape)
                mult = torch.tensor(np.multiply(context[t].detach().numpy(), query[j].detach().numpy()))
                # print(mult, mult.shape)
                input_vec = torch.cat((context[t], query[j], mult), dim=0)
                # print(input_vec)
                S[t][j] = self.sim_weight(input_vec.squeeze()).squeeze()
                # print(S[t][j])
        # Context to query
        context2query = torch.tensor(np.zeros((context.shape[0], context.shape[1])))
        for t in range(context.shape[0]):
            a_t = self.softmax(torch.tensor(S[t]))
            for j in range(query.shape[0]):
                context2query[t] += torch.tensor(np.multiply(a_t[j].detach().numpy(), query[j].detach().numpy())).squeeze()
        # query2context = np.zeros((context.shape[0], context.shape[1]))
        b = np.max(S, axis=1)
        h_tilde = torch.tensor(np.zeros(context.shape[1]))
        for t in range(context.shape[0]):
            h_tilde = h_tilde + torch.tensor(np.multiply(b[t], context[t].detach().numpy()))
        for t in range(context.shape[0]):
            new_vec = torch.cat((context[t], context2query[t],
                                 torch.tensor(np.multiply(context[t].detach().numpy(), context2query[t].detach().numpy())),
                                 torch.tensor(np.multiply(context[t].detach().numpy(), h_tilde.detach().numpy()))), dim=0)
            if G is None:
                G = new_vec.unsqueeze(0)
            else:
                G = torch.cat((G, new_vec.unsqueeze(0)), dim=0)
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

        return G


class ModelingLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=2, dropout=0.2,
                 bidirectional=True):
        super(ModelingLayer, self).__init__()
        #############################################################################
        # TODO: Define and initialize anything needed for the forward pass.
        # For the modeling layer, we just need to pass our query aware context vectors
        # You need to initialize an LSTM layer here
        #############################################################################
        self.lstm = nn.LSTM(input_dim, output_dim, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional)
        self.lstm = self.lstm.double()
        for param in self.parameters():
            param.requires_grad = True
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

    def forward(self, G):
        M = None
        #############################################################################
        # TODO: Implement the forward pass.
        # G : query aware context word embeddings
        # returns :: of size Tx(output_dim*2) (T is # words in context)
        #############################################################################
        # G = G.double()
        M = self.lstm(G.double().unsqueeze(0))
        M = M[0].squeeze(0)
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return M


class OutputLayer(nn.Module):
    def __init__(self, fc_dim, LSTM_input_size, LSTM_output_size, num_layers=1, bidirectional=True):
        super(OutputLayer, self).__init__()
        #############################################################################
        # TODO: Define and initialize anything needed for the forward pass.
        # For the OutputLayer, we need:
        #   Linear layer (to predict start idx; no bias for these linear layers according to paper)
        #   LSTM + Linear (to predict end idx)
        #############################################################################
        self.start_linear = nn.Linear(fc_dim, 1, bias=False).double()
        self.start_softmax = nn.Softmax()
        self.end_lstm = nn.LSTM(LSTM_input_size, LSTM_output_size, num_layers=num_layers, bidirectional=bidirectional).double()
        self.end_linear = nn.Linear(fc_dim, 1, bias=False).double()
        self.end_softmax = nn.Softmax()
        for param in self.parameters():
            param.requires_grad = True
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

    def forward(self, G, M):
        start, end = None, None
        #############################################################################
        # TODO: Implement the forward pass.
        # G : query aware context word embeddings
        # M : output of modeling layer
        # returns :: `start` and `end` of size (T,)
        #############################################################################
        start = self.start_linear(torch.cat((G, M), dim=1).double()).squeeze(1)
        end = self.end_lstm(M.unsqueeze(0).double())
        # print(end)
        end = end[0].squeeze(0)
        end = self.end_linear(torch.cat((G, end), dim=1).double()).squeeze(1)
        start = self.start_softmax(start)
        end = self.end_softmax(end)
        # print(end, end.shape)
        # print(start, start.shape)
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return start, end


class BiDAF(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, char_embedding_dim,
                 char_hidden_dim, char_size, vocab_size, bidirectional=True, phrase_LSTM_layers=1,
                 modeling_LSTM_layers=2, dropout=0):
        super(BiDAF, self).__init__()
        #############################################################################
        # TODO: Define and initialize anything needed for the forward pass.
        # Initialize all the modules you created so far and link their inputs and
        #       outputs properly:
        #   LSTMEncoder
        #   AttentionFlow
        #   ModelingLayer
        #   Output
        #############################################################################
        directions = 2 if bidirectional else 1
        self.lstm_encoder = LSTMEncoder(embedding_dim, hidden_dim, char_embedding_dim,
                                        char_hidden_dim, char_size, vocab_size, lstm_layers=phrase_LSTM_layers,
                                        bidirectional=bidirectional, dropout=dropout)
        self.attention_flow = AttentionFlow(w_dim=hidden_dim * directions * 3)
        self.modeling = ModelingLayer(hidden_dim * directions * 4, hidden_dim, dropout=dropout,
                                      bidirectional=bidirectional, num_layers=modeling_LSTM_layers)
        self.output = OutputLayer(hidden_dim * directions * 5, hidden_dim * directions, hidden_dim, bidirectional=bidirectional)
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

    def forward(self, context, context_chars, query, query_chars):
        start, end = None, None
        #############################################################################
        # TODO: Implement the forward pass.
        # Given a tokenized index-mapped sentence and a character sequence as the arguments,
        # find the corresponding scores for tags
        # returns:: `start` and `end` of size (T,)
        #   where, T is the number of words/tokens in context
        #############################################################################
        context_vec, query_vec = self.lstm_encoder.forward(context, context_chars, query, query_chars)
        G = self.attention_flow.forward(context_vec, query_vec)
        M = self.modeling.forward(G)
        start, end = self.output.forward(G, M)
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return start, end


##### From Official SQUAD Evauation evaluation script version 2.0 #####

def normalize_answer(s):
    """Lower text and remove punctuation, artcles and extra whitespace"""

    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)]b', re.UNICODE)
        try:
            return re.sub(regex, ' ', text)
        except:
            return text

    def white_space_fix(text):
        try:
            return ' '.join(text.split())
        except:
            return text

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        try:
            return text.lower()
        except:
            return text

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1
