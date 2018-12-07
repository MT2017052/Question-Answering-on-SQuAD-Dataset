import keras
from keras.models import Model
from keras.layers import Input, LSTM, Bidirectional
from numpy import array
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
from nltk.tokenize import WhitespaceTokenizer
import numpy as np

# Using RNNEncoder Class so that both Context And Questions can be trained on same weights


class RNNEncoder:
    def getHiddenVectors(self, input_shape, data):
        lstm, forward_h, forward_c, backward_h, backward_c = Bidirectional(
            LSTM(2, return_sequences=True, return_state=True))(input_shape)
        model = Model(inputs=input_shape, outputs=[lstm])
        return model.predict(data)

# define model
# input_shape = Input(shape=(3, 1))
# print(input_shape)

# define input data
# context_emb = array([0.1, 0.2, 0.3]).reshape((1,3,1))
# question_emb = array([0.1, 0.2, 0.3]).reshape((1,3,1))

# Get Hidden vectors for Context and Question using RNNEncoder Object defined earlier
# encoder = RNNEncoder()
# hidden_context = encoder.getHiddenVectors(input_shape, context_emb)
# hidden_question = encoder.getHiddenVectors(input_shape, question_emb)

# print("__________Context_hidden__________")
# print(hidden_context)
# print("__________Question_hidden__________")
# print(hidden_question)


# load the Stanford GloVe model
filename = 'glove.6B.100d.txt.word2vec'

model = KeyedVectors.load_word2vec_format(filename, binary=False)

# calculate: (king - man) + woman = ?
result = model.most_similar(positive=['man', 'queen'], negative=['woman'], topn=1)
# print(result)

# print GloVe vector representation for word 'umbrella'
print("GloVe vector representation for word 'umbrella', vector length: {0}, vector: {1}".format(
    len(model['umbrella']), model['umbrella']))

maxlength = 0
context_emb = []
# embeddings = []
w = 0
with open("context.txt") as f:
    data = f.readlines()
    for line in data:
        tokens = WhitespaceTokenizer().tokenize(line)
        paragraph_embeddings = []
        for word in tokens:
            if word not in model:
                w = w+1
            else:
                embedding = model[word]
                paragraph_embeddings.append(embedding)
        context_emb.append(paragraph_embeddings)
        if (maxlength < len(tokens)):
            maxlength = len(tokens)
        # context_emb.append(embeddings)

# print(context_emb)

print len(context_emb[1])
# print context_emb[1].shape

maxlength2 = 0
question_emb = []
# q_embeddings = []
with open("question.txt") as f:
    data = f.readlines()
    for line in data:
        tokens = WhitespaceTokenizer().tokenize(line)
        question_embeddings = []
        for word in tokens:
            embedding = model[word]
            question_embeddings.append(embedding)
        question_emb.append(question_embeddings)
        if (maxlength2 < len(tokens)):
            maxlength2 = len(tokens)
        # question_emb.append(q_embeddings)

# print(question_emb)

print len(question_emb[1])
# print question_emb[1].shape

# define model
input_shape_context = Input(shape=(120, 100))
input_shape_question = Input(shape=(8, 100))
# print(input_shape)

# define input data
context_emb_1 = array(context_emb[1]).reshape((1, 120, 100))
question_emb_1 = array(question_emb[1]).reshape((1, 8, 100))

context_emb_test = array(context_emb[1]).reshape((1, 120, 100))
question_emb_test = array(question_emb[1]).reshape((1, 8, 100))

# Get Hidden vectors for Context and Question using RNNEncoder Object defined earlier
encoder = RNNEncoder()
hidden_context = encoder.getHiddenVectors(input_shape_context, context_emb_1)
hidden_question = encoder.getHiddenVectors(input_shape_question, question_emb_1)

hidden_context_test = encoder.getHiddenVectors(input_shape_context, context_emb_test)
hidden_question_test = encoder.getHiddenVectors(input_shape_question, question_emb_test)

hidden_context_vector = hidden_context[0]
hidden_question_vector = hidden_question[0]

hidden_context_vector_test = hidden_context_test[0]
hidden_question_vector_test = hidden_question_test[0]

print("__________Context_hidden__________")
print(hidden_context_vector)
print("__________Question_hidden__________")
print(hidden_question_vector)

rows = []
for hc in range(len(hidden_context_vector)):
    columns = []
    for hq in range(len(hidden_question_vector)):
        context_vector = hidden_context_vector[hc]
        question_vector = hidden_question_vector[hq]
        mul = (context_vector[0]*question_vector[0])+(context_vector[1]*question_vector[1]) + \
            (context_vector[2]*question_vector[2])+(context_vector[3]*question_vector[3])
        columns.append(mul)
    rows.append(columns)

# print(rows)

for row in rows:
    row = np.asarray(row, dtype=float)

rows = np.asarray(rows)

print rows

rows_test = []
for hc in range(len(hidden_context_vector_test)):
    columns_test = []
    for hq in range(len(hidden_question_vector_test)):
        context_vector = hidden_context_vector_test[hc]
        question_vector = hidden_question_vector_test[hq]
        mul = (context_vector[0]*question_vector[0])+(context_vector[1]*question_vector[1]) + \
            (context_vector[2]*question_vector[2])+(context_vector[3]*question_vector[3])
        columns_test.append(mul)
    rows_test.append(columns_test)

# print(rows)

for row in rows_test:
    row = np.asarray(row, dtype=float)

rows_test = np.asarray(rows_test)


def softmax(x):
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=0, keepdims=True)
    s = x_exp/x_sum
    return s


attn_query_to_context = softmax(rows)
attn_query_to_context_test = softmax(rows_test)
# print attn_query_to_context

ground_truth = np.zeros(120, dtype=int)

with open('span.txt') as f:
    line1 = f.readline()
    line2 = f.readline()

span = map(int, line2.strip().split(' '))

# print span
start = span[0]
end = span[1]
for i in range(start, end+1):
    ground_truth[i] = 1

# print ground_truth

model = keras.Sequential()
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(attn_query_to_context,
                    ground_truth,
                    epochs=40,
                    batch_size=512,
                    verbose=1)

results = model.predict(attn_query_to_context_test)

print(results)
