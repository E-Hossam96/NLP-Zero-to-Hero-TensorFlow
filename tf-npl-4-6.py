import urllib
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
import matplotlib.pyplot as plt

# downloading the file and converting to strings
url = 'https://storage.googleapis.com/laurencemoroney-blog.appspot.com/irish-lyrics-eof.txt'
response = urllib.request.urlopen(url)
datastore = response.read().decode('utf-8')
corpus = datastore.lower().split('\n')

# tokenizing the words
tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1
print(f'total number of words: {total_words}')

# generating a step by step input sequences
input_sequences = []

for line in corpus:
    line_sequences = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(line_sequences)):
        input_sequences.append(line_sequences[ : i + 1])

# padding the sequences
## get max sequence length
max_sequence_len = max([len(i) for i in input_sequences])
## padding all the sequences with zeros in the beginning
input_sequences = pad_sequences(input_sequences, maxlen = max_sequence_len, padding = 'pre')

# creating observations and labels
X, y = input_sequences[:, :-1], input_sequences[:, -1]
# encoding the targets using keras
y = to_categorical(y, num_classes = total_words)

# building the neural network model
model = Sequential([
    Embedding(total_words, 240, input_length = max_sequence_len - 1),
    Bidirectional(LSTM(150)),
    Dense(total_words, activation = 'softmax')
])
print(model.summary())
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
history = model.fit(X, y, epochs = 30)

# plotting the propagation of accuracy over epochs
def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.show()

plot_graphs(history, 'accuracy')

# generating new text
seed_text = "I've got mail"
num_words = 100
# defining the loop to sequence the words
for _ in range(num_words):
    sequence = tokenizer.texts_to_sequences([seed_text])[0]
    padded = pad_sequences([sequence], maxlen = max_sequence_len - 1, padding = 'pre')
    predicted = np.argmax(model.predict(padded, verbose = 0))
    output_word = ''
    for word, index in tokenizer.word_index.items():
        if predicted == index:
            output_word = word
            break
    seed_text += ' ' + output_word
# printing the final output
print(seed_text)
