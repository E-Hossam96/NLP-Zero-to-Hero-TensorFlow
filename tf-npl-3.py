import urllib, json
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
import matplotlib.pyplot as plt


# downloading the json file
url = 'https://storage.googleapis.com/laurencemoroney-blog.appspot.com/sarcasm.json'
response = urllib.request.urlopen(url)
datastore = json.loads(response.read())

# getting valuable data from the json file
sentences, labels = [], []
for item in datastore:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])

# initializing some parameters
training_size = 20000
vocab_size = 10000
oov_tok = "<OOV>"
max_length = 100
padding_type = 'post'
trunc_type = 'post'
embedding_dim = 16

# splitting the data into training and testing sets
training_sentences = sentences[ : training_size]
testing_sentences = sentences[training_size : ]
training_labels = labels[ : training_size]
testing_labels = labels[training_size : ]

# transforming to word indices
tokenizer = Tokenizer(num_words = vocab_size, oov_token = oov_tok)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(
    training_sequences, maxlen = max_length, padding = padding_type, truncating = trunc_type
)
testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(
    testing_sequences, maxlen = max_length, padding = padding_type, truncating = trunc_type
)

# transforming to numpy arrays to work with tensorflow
training_padded = np.array(training_padded)
training_labels = np.array(training_labels)
testing_padded = np.array(testing_padded)
testing_labels = np.array(testing_labels)

# building the neural network
model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length = max_length),
    GlobalAveragePooling1D(),
    Dense(32, activation = 'relu'),
    Dense(1, activation = 'sigmoid')
])
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
print(model.summary())
history = model.fit(
    training_padded, training_labels, epochs = 10, 
    validation_data = (testing_padded, testing_labels),
    verbose = 2
)

# plotting the results
def plot_graphs(history):
    fig, axes = plt.subplots(1, 2, figsize = (10, 5))
    axes[0].plot(history.history['loss'], label = 'Train Loss')
    axes[0].plot(history.history['val_loss'], label = 'Val. Loss')
    axes[1].plot(history.history['accuracy'], label = 'Train Accuracy')
    axes[1].plot(history.history['val_accuracy'], label = 'Val. Accuracy')
    axes[0].legend(loc = 'best')
    axes[1].legend(loc = 'best')
    fig.supxlabel('Epochs')
    fig.supylabel('Propagation')
    plt.show()

plot_graphs(history)

# predictions using the trained model
sentence = [
    "granny starting to fear spiders in the garden might be real", 
    "game of thrones season finale showing this sunday night"
]
sequences = tokenizer.texts_to_sequences(sentence)
padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
print(f'predictions:\n{model.predict(padded)}')
