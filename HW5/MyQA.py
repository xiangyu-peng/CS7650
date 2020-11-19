import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import string
import re
import collections
random.seed(0)
import numpy as np


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

        # Remember: bidirectional makes the output hidden_dim * 2
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.char_embedding_dim = char_embedding_dim
        self.char_hidden_dim = char_hidden_dim
        
        directions = 2 if bidirectional else 1
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=1)
        self.lstm = nn.GRU(char_hidden_dim * directions + embedding_dim, hidden_dim, bidirectional=bidirectional, num_layers=lstm_layers, dropout=dropout)  
        
        self.embedding_c = nn.Embedding(char_size + 1, char_embedding_dim, padding_idx=1) #char_size + 1, char_embedding_dim
        self.lstm_c = nn.GRU(char_embedding_dim * 30, char_hidden_dim, dropout=dropout, bidirectional=bidirectional) 

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

    def forward(self, context, context_chars, query, query_chars):
        lstm_context_vectors = None # for each word
        lstm_query_vectors = None # for each word 
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
        # embedding for words
        embedded = self.embedding(context)      
        
        # embedding for chars
        context_chars = context_chars.reshape(-1,30)  # reshape first (# of words * 30(max_word_len))

        embedded_c = self.embedding_c(context_chars)
        embedded_c = embedded_c.view(embedded_c.shape[0], 1, -1)
        
        # lstm for char
        outputs_c, _ = self.lstm_c(embedded_c)
        
        # cat word embed with lstm[-1] of chars      
        joint_emb = torch.cat((embedded, outputs_c.reshape(outputs_c.shape[0], -1)), dim=1)
        joint_emb = joint_emb.reshape(joint_emb.shape[0], 1, -1)
        
        # lstm for word and chars
        lstm_context_vectors, _ = self.lstm(joint_emb)
        lstm_context_vectors = lstm_context_vectors.reshape(lstm_context_vectors.shape[0], -1)
        
        # query
        embedded = self.embedding(query)  
        
        query_chars = query_chars.reshape(-1,30)
        embedded_c = self.embedding_c(query_chars)

        embedded_c = embedded_c.view(embedded_c.shape[0], 1, -1)
        outputs_c, _ = self.lstm_c(embedded_c)
        
        joint_emb = torch.cat((embedded, outputs_c.reshape(outputs_c.shape[0], -1)), dim=1)
        joint_emb = joint_emb.reshape(joint_emb.shape[0], 1, -1)
        lstm_query_vectors, _ = self.lstm(joint_emb)
        lstm_query_vectors = lstm_query_vectors.reshape(lstm_query_vectors.shape[0], -1)
               
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
        self.hidden_size = int(w_dim/6)
        self.W = nn.Linear(w_dim, 1)
        self.softmax = nn.Softmax()
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

#         # Make a similarity matrix  
#         if len(context.shape) == 3:
#             N = context.shape[0]
#             T = context.shape[1]
#             J = query.shape[1]
#         else:
#             T = context.shape[0]
#             J = query.shape[0]
#             N = 1
        
#         context = context.reshape(1, T, 1, -1)  # (N, T, 1, 2d)
#         query = query.reshape(1, 1, J, -1)  # (N, 1, J, 2d)
        
#         shape = (1, T, J, 2*self.hidden_size)            # (N, T, J, 2d)
 
#         embd_context_ex = context.expand(shape) # (N, T, J, 2d)
#         embd_query_ex = query.expand(shape)     # (N, T, J, 2d)
        
#         a_elmwise_mul_b = torch.mul(embd_context_ex, embd_query_ex) # (N, T, J, 2d)
#         cat_data = torch.cat((embd_context_ex, embd_query_ex, a_elmwise_mul_b), 3) # (N, T, J, 6d), [h;u;h◦u]
#         S = self.W(cat_data).view(1, T, J) # (N, T, J)

#         # Context2Query
#         query = query.reshape(1, J, -1)
#         c2q = torch.bmm(F.softmax(S, dim=-1), query) # (N, T, 2d) = bmm( (N, T, J), (N, J, 2d) )
        
#         # Query2Context
#         # b: attention weights on the context
#         context = context.reshape(1, T, -1)
#         b = F.softmax(torch.max(S, 2)[0], dim=-1) # (N, T)
#         q2c = torch.bmm(b.unsqueeze(1), context) # (N, 1, 2d) = bmm( (N, 1, T), (N, T, 2d) )
#         q2c = q2c.repeat(1, T, 1) # (N, T, 2d), tiled T times

#         # G: query aware representation of each context word
#         G = torch.cat((context, c2q, context.mul(c2q), context.mul(q2c)), 2) # (N, T, 8d)



#         context = context.reshape(N, T, 1, -1)  # (N, T, 1, 2d)
#         query = query.reshape(N, 1, J, -1)  # (N, 1, J, 2d)

#         shape = (N, T, J, 2 * self.hidden_size)  # (N, T, J, 2d)

#         embd_context_ex = context.expand(shape)  # (N, T, J, 2d)
#         embd_query_ex = query.expand(shape)  # (N, T, J, 2d)

#         a_elmwise_mul_b = torch.mul(embd_context_ex, embd_query_ex)  # (N, T, J, 2d)
#         cat_data = torch.cat((embd_context_ex, embd_query_ex, a_elmwise_mul_b), 3)  # (N, T, J, 6d), [h;u;h◦u]
#         S = self.W(cat_data).view(N, T, J)  # (N, T, J)

#         # Context2Query
#         query = query.reshape(N, J, -1)
#         c2q = torch.bmm(F.softmax(S, dim=-1), query)  # (N, T, 2d) = bmm( (N, T, J), (N, J, 2d) )

#         # Query2Context
#         # b: attention weights on the context
#         context = context.reshape(N, T, -1)
#         b = F.softmax(torch.max(S, 2)[0], dim=-1)  # (N, T)
#         q2c = torch.bmm(b.unsqueeze(1), context)  # (N, 1, 2d) = bmm( (N, 1, T), (N, T, 2d) )
#         q2c = q2c.repeat(1, T, 1)  # (N, T, 2d), tiled T times

#         # G: query aware representation of each context word
#         # print('c2q', c2q.shape)
#         # print('q2c', q2c.shape)
#         # print('context', context.shape)
#         # print('context, c2q, context.mul(c2q), context.mul(q2c)', context.shape, c2q.shape, context.mul(c2q).shape, context.mul(q2c).shape)
#         G = torch.cat((context, c2q, context.mul(c2q), context.mul(q2c)), 2)
#         if N == 1:
#             G = G.reshape(G.shape[1], -1).double()
#         else:
#             G = G.reshape(N, G.shape[1], -1).double()


        T = context.shape[0]
        J = query.shape[0]
        S = np.zeros((T, J))
        for t in range(T):
            for j in range(J):
                sim = torch.mul(context[t], query[j])
                input_cq = torch.cat((context[t], query[j], sim), dim=0)
                S[t][j] = self.W(input_cq.squeeze()).squeeze()

        # Context to query
        context2query = torch.tensor(np.zeros((context.shape[0], context.shape[1]))) #(T, 2d)
        for t in range(T):
            a_t = self.softmax(torch.tensor(S[t]))
            for j in range(J):
                context2query[t] += torch.mul(a_t[j], query[j]).squeeze()
        
        # Query2Context
        # b: attention weights on the context
        b = np.max(S, axis=1)
        q2c = torch.tensor(np.zeros(context.shape[1]))
        for t in range(T):
            q2c += torch.tensor(np.multiply(b[t], context[t].detach().numpy()))
        
        for t in range(T):
            new_vec = torch.cat((context[t], context2query[t],
                                 torch.mul(context[t], context2query[t]),
                                 torch.mul(context[t], q2c)), dim=0)
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
        
#         self.lstm_1 = nn.LSTM(input_size=input_dim,
#                                    hidden_size=output_dim * 2,
#                                    bidirectional=bidirectional,
#                                    dropout=dropout,
#                                    num_layers=num_layers)

#         self.lstm_2 = nn.LSTM(input_size=output_dim * 2,
#                                    hidden_size=output_dim,
#                                    bidirectional=bidirectional,
#                                    dropout=dropout,
#                                    num_layers=num_layers)
        self.lstm = nn.GRU(input_size=input_dim,
                           hidden_size=output_dim,
                           bidirectional=bidirectional,
                           dropout=dropout,
                           num_layers=num_layers)
        self.lstm = self.lstm.double()

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
        # G size len(c) * (8 * hiddensize)
        G = G.double().unsqueeze(0)
        M, _ =  self.lstm(G)
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
        
        # predict start idx
        self.fc_GM = nn.Linear(fc_dim, 1).double()
        
        # predict end idx
        self.lstm = nn.GRU(LSTM_input_size, LSTM_output_size, bidirectional=bidirectional, num_layers=num_layers).double()
        self.fc_GM_lstm = nn.Linear(fc_dim, 1).double()
        
        
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
        
        GM = torch.cat((G,M), dim=-1)  # (c_len ,hidden_size * 10)
        GM = GM.reshape(1, GM.shape[0], -1)  # (batch, c_len, hidden_size * 10)
        start = self.fc_GM(GM.double()).reshape(1, -1)
        start = F.softmax(start, dim=-1)
        
        
        G = G.reshape(1, G.shape[0], -1)  # (batch, c_len, hidden_size * 8)
        M = M.reshape(1, M.shape[0], -1)  # (batch, c_len ,hidden_size * 2)
        
        output_M, _ =  self.lstm(M.double())
        GM_lstm = torch.cat((G,output_M), dim=2)
        end = self.fc_GM_lstm(GM_lstm.double()).reshape(1, -1)
        end = F.softmax(end, dim=-1)
        start, end =  start.reshape(-1), end.reshape(-1)
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return start, end

 
class BiDAF(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, char_embedding_dim,
                 char_hidden_dim, char_size, vocab_size, bidirectional=True, phrase_LSTM_layers=1, modeling_LSTM_layers=2, dropout=0):
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
        self.encoder = LSTMEncoder(embedding_dim=embedding_dim, 
                                   hidden_dim=hidden_dim, 
                                   char_embedding_dim=char_embedding_dim,
                                   char_hidden_dim=char_hidden_dim, 
                                   char_size=char_size, 
                                   vocab_size=vocab_size, 
                                   lstm_layers=phrase_LSTM_layers, 
                                   bidirectional=bidirectional, 
                                   dropout=dropout)
        
        directions = 2 if bidirectional else 1
        
        self.attention = AttentionFlow(w_dim=3 * hidden_dim * directions)
        
        self.modelLayer = ModelingLayer(input_dim=hidden_dim * directions * 4, 
                                        output_dim=hidden_dim, 
                                        num_layers=modeling_LSTM_layers, 
                                        dropout=dropout, 
                                        bidirectional=bidirectional)
        self.outputLayer = OutputLayer(fc_dim=hidden_dim*5*directions, 
                                       LSTM_input_size=hidden_dim * directions, 
                                       LSTM_output_size=hidden_dim, 
                                       bidirectional=bidirectional)
        
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
        
        # encode the words and chars
        context_enc, query_enc = self.encoder(context, context_chars, query, query_chars)
        
        # use attention layer
        G = self.attention(context_enc, query_enc)
        
        # put G into 2 layers of LSTM
        M = self.modelLayer(G)
        
        # output layer
        start, end = self.outputLayer(G, M)
        
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
