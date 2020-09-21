import math, random
from typing import List, Tuple

################################################################################
# Part 0: Utility Functions
################################################################################

def start_pad(n):
    ''' Returns a padding string of length n to append to the front of text
        as a pre-processing step to building n-grams '''
    return ['~'] * n

Pair = Tuple[str, str]
Ngrams = List[Pair]

def ngrams(n, text:str) -> Ngrams:
    text = text.strip().split()
    ''' Returns the ngrams of the text as tuples where the first element is
        the n-word sequence (i.e. "I love machine") context and the second is the word '''
    ngrams_list = []
    for i , word in enumerate(text):
        if n > i:
            ngram = (' '.join(start_pad(n - i) + text[:i]), word)
        else:
            ngram = (' '.join(text[i - n : i]) , word)
        ngrams_list.append(ngram)
    return ngrams_list
        

def create_ngram_model(model_class, path, n=2, k=0):
    ''' Creates and returns a new n-gram model trained on the path file '''
    model = model_class(n, k)
    with open(path, encoding='utf-8') as f:
        model.update(f.read())
    return model

################################################################################
# Part 1: Basic N-Gram Model
################################################################################

class NgramModel(object):
    ''' A basic n-gram model using add-k smoothing '''

    def __init__(self, n, k):
        self.n = n
        self.k = k
        self.vocab = set()
        self.ngrams = dict()
        self.ngrams_char_count = dict()

    def get_vocab(self):
        ''' Returns the set of words in the vocab '''
        return self.vocab
    
    def update_ngram_char_count(self, ngram):
        if ngram[0] not in self.ngrams_char_count:
            self.ngrams_char_count[ngram[0]] = dict()
        if ngram[1] in self.ngrams_char_count[ngram[0]]:
            self.ngrams_char_count[ngram[0]][ngram[1]] += 1
        else:
            self.ngrams_char_count[ngram[0]][ngram[1]] = 1
            
    def update(self, text:str):
        ''' Updates the model n-grams based on text '''
        ngram_list = ngrams(self.n, text)
        for ngram in ngram_list:
            self.vocab.add(ngram[1])  # update vocab
            self.update_ngram_char_count(ngram)  # update ngrams_char_count
            if ngram[0] in self.ngrams:  # update ngram dict     
                self.ngrams[ngram[0]] += 1
            else:
                self.ngrams[ngram[0]] = 1

    def prob(self, context:str, word:str):
        ''' Returns the probability of word appearing after context '''        
        if context in self.ngrams:
            count = self.ngrams[context]
            count_ngram = 0
            if word in self.ngrams_char_count[context]:
                count_ngram = self.ngrams_char_count[context][word]
            
            return (self.k + count_ngram) / (count + self.k * len(self.vocab))

        else:
            return 1 / len(self.vocab)
        
    def random_word(self, context):
        ''' Returns a random word based on the given context and the
            n-grams learned by this model '''
#         random.seed(1)
        r = random.random()
        vocab_list = sorted(list(self.vocab))
        p = 0
        for v in vocab_list:
            p += self.prob(context, v)
            if p >= r:
                return v
        return None

    def random_text(self, length):
        ''' Returns text of the specified word length based on the
            n-grams learned by this model '''
        start_context = ' '.join(['~'] * self.n)
        text = ''
        for l in range(length):
            word = self.random_word(start_context)  # add new generated char
            text += word + ' '
            if self.n > 1:
                start_context = ' '.join(start_context.split(' ')[1:]) + ' ' + word
            else:
                start_context = word
        return text.strip()

    def perplexity(self, text):
#         text = text.strip().split()
        ''' Returns the perplexity of text based on the n-grams learned by
            this model '''
        
        p_list = []
        ngram_list = ngrams(self.n, text)
        
        for context, word in ngram_list:
            p = self.prob(context, word)
            if p == 0:
                return float('inf')
            p_list.append(p)
        
        perplexity = sum([math.log(p,2) for p in p_list]) / len(p_list)
        return math.pow(2, -1 * perplexity)

################################################################################
# Part 2: N-Gram Model with Interpolation
################################################################################

class NgramModelWithInterpolation(NgramModel):
    ''' An n-gram model with interpolation '''

    def __init__(self, n, k):
        self.n = n
        self.k = k
        self.vocab = set()
        self.ngrams = dict()
        self.ngrams_char_count = dict()
        self.lamb = [1 / (n + 1) for i in range(n + 1)]
        

    def get_vocab(self):
        return self.vocab

    def update(self, text:str):
        for n in range(self.n + 1):
            ngram_list = ngrams(n, text)
            for ngram in ngram_list:
                self.vocab.add(ngram[1])  # update vocab
                self.update_ngram_char_count(ngram)  # update ngrams_char_count
                if ngram[0] in self.ngrams:  # update ngram dict     
                    self.ngrams[ngram[0]] += 1
                else:
                    self.ngrams[ngram[0]] = 1

    def prob(self, context:str, word:str):
        p_list = []
        context = context.split(' ')
        for n in range(self.n + 1): 
            context_start = ' '.join(context[n:])  # get the context string            
            if context_start in self.ngrams:
                count = self.ngrams[context_start]
#                 print('count ', context[:n], ' is ', count)
                count_ngram = 0
                if word in self.ngrams_char_count[context_start]:
#                     print('char',char)                  
                    count_ngram = self.ngrams_char_count[context_start][word]
#                     print('count_ngram', count_ngram)
#                 print(char, ' in count_ngram', context[:n], ' is ', count_ngram)
                p_list.append((self.k + count_ngram) / (count + self.k * len(self.vocab)))

            else:
#                 print('no ngram ', context[:n])
                p_list.append(1 / len(self.vocab))
        p_list = [p * self.lamb[i] for i, p in enumerate(p_list)]
        return sum(p_list)
    
    def set_lambda(self, lamb_list):
        self.lamb = lamb_list

################################################################################
# Part 3: Your N-Gram Model Experimentation
################################################################################

if __name__ == '__main__':
    pass