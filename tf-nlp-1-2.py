import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = [
    'I love reading',
    'I love coffee',
    'Do you love coffee?',
    'Do you thin he loves coffee?'
]

tokenizer = Tokenizer(num_words = 100, oov_token = '<OOV>')
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(sentences)

test_data = [
    'I really love coffee',
    'he loves coffee too!'
]
test_seq = tokenizer.texts_to_sequences(test_data)

padded = pad_sequences(sequences)
padded_test = pad_sequences(test_seq)

print(f'word index:\n{word_index}')
print(f'sequences:\n{sequences}')
print(f'test sequences:\n{test_seq}')
print(f'padded sequences:\n{padded}')
print(f'padded test sequences:\n{padded_test}')
