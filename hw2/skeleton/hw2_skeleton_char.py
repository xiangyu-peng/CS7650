import math, random

################################################################################
# Part 0: Utility Functions
################################################################################

def start_pad(n):
    ''' Returns a padding string of length n to append to the front of text
        as a pre-processing step to building n-grams '''
    return '~' * n

def ngrams(n, text):
    ''' Returns the ngrams of the text as tuples where the first element is
        the length-n context and the second is the character '''
    
    n_gram_list = []
    for i, char in enumerate(text):
        if n > i:
            n_gram = (start_pad(n - i) + text[:i], char)
        else:
            n_gram = (text[i - n : i] , char)
        n_gram_list.append(n_gram)
        
    return n_gram_list

def create_ngram_model(model_class, path, n=2, k=0):
    ''' Creates and returns a new n-gram model trained on the path file '''
    model = model_class(n, k)
    with open(path, encoding='utf-8', errors='ignore') as f:
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
        ''' Returns the set of characters in the vocab '''
        return self.vocab

    def update(self, text):
        ''' Updates the model n-grams based on text '''
        ngram_list = ngrams(self.n, text)
        for ngram in ngram_list:
            self.vocab.add(ngram[1])  # update vocab
            self.update_ngram_char_count(ngram)  # update ngrams_char_count
            if ngram[0] in self.ngrams:  # update ngram dict     
                self.ngrams[ngram[0]] += 1
            else:
                self.ngrams[ngram[0]] = 1

    def prob(self, context, char):
        ''' Returns the probability of char appearing after context '''
        # get the count of ngram - context
        if context in self.ngrams:
            count = self.ngrams[context]
            count_ngram = 0
            if char in self.ngrams_char_count[context]:
                count_ngram = self.ngrams_char_count[context][char]
            
            return (self.k + count_ngram) / (count + self.k * len(self.vocab))

        else:
            return 1 / len(self.vocab)
        
        
    def update_ngram_char_count(self, ngram):
        if ngram[0] not in self.ngrams_char_count:
            self.ngrams_char_count[ngram[0]] = dict()
        if ngram[1] in self.ngrams_char_count[ngram[0]]:
            self.ngrams_char_count[ngram[0]][ngram[1]] += 1
        else:
            self.ngrams_char_count[ngram[0]][ngram[1]] = 1
        

    def random_char(self, context):
        ''' Returns a random character based on the given context and the 
            n-grams learned by this model '''
#         random.seed(1)
        r = random.random()
        vocab_list = sorted(list(self.vocab))
        p = 0
        for v in vocab_list:
            p += self.prob(context, v)
            if p > r:
                return v
        return None
        

    def random_text(self, length):
        ''' Returns text of the specified character length based on the
            n-grams learned by this model '''
        start_context = '~' * self.n
        text = ''
        for l in range(length):
            char = self.random_char(start_context)  # add new generated char
            text += char
            start_context = start_context[1:] + char
        return text

    def perplexity(self, text):
        ''' Returns the perplexity of text based on the n-grams learned by
            this model '''
        
        p_list = []
        ngram_list = ngrams(self.n, text)
        
        for context, char in ngram_list:
            p = self.prob(context, char)
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

    def update(self, text):
        # the ngrams should be caled based on the all the n
        for n in range(self.n + 1):
            ngram_list = ngrams(n, text)
            for ngram in ngram_list:
                if n == 0:
                    self.vocab.add(ngram[1])  # update vocab
                self.update_ngram_char_count(ngram)  # update ngrams_char_count
                if ngram[0] in self.ngrams:  # update ngram dict     
                    self.ngrams[ngram[0]] += 1
                else:
                    self.ngrams[ngram[0]] = 1
                
    def prob(self, context, char):
        # get the count of ngram - context
        p_list = []
        for n in range(self.n + 1):  
            if context[n:] in self.ngrams:
#                 print('context is ', context[n:])
                count = self.ngrams[context[n:]]
#                 print('count ', context[n:], ' is ', count)
                count_ngram = 0
                if char in self.ngrams_char_count[context[n:]]:
#                     print('char',char)
                    
                    count_ngram = self.ngrams_char_count[context[n:]][char]
#                     print('count_ngram', count_ngram)
#                 print(char, ' in count_ngram', context[n:], ' is ', count_ngram)
                p_list.append((self.k + count_ngram) / (count + self.k * len(self.vocab)))

            else:
#                 print('no ngram ', context[n:])
                p_list.append(1 / len(self.vocab))
#             print('=====')
#         print(p_list)
#         print(p_list)
        p_list = [p * self.lamb[i] for i, p in enumerate(p_list)]
#         print(p_list)
        return sum(p_list)
    
    def set_lambda(self, lamb_list):
        self.lamb = lamb_list
    
    

################################################################################
# Part 3: Your N-Gram Model Experimentation
################################################################################

if __name__ == '__main__':
    pass