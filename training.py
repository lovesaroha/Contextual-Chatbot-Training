# Love Saroha
# lovesaroha1994@gmail.com (email address)
# https://www.lovesaroha.com (website)
# https://github.com/lovesaroha  (github)

# Train keras model for contextual chat.
import json
import numpy
import random
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow.keras.utils as ku

# Parameters.
token_size = 100
epochs = 50
batchSize = 1
totalTags = 0

# Get data from intents.json file.
with open("./intents.json", 'r') as file:
    data = json.load(file)

# Save as sentences and labels.
training_sentences = []
labels = []
training_labels = []
for intent in data['intents']:
    totalTags += 1
    labels.append(intent["tag"])
    for pattern in intent['patterns']:
        training_sentences.append(pattern)
        training_labels.append(labels.index(intent["tag"]))

# Sequence to binary.
def sequenceToBinary(seq):
     d = numpy.zeros(total_words, dtype=numpy.float32)
     for l in seq:
          i = l - 1
          d[i] = 1
     return d


# Create tokenizer.
tokenizer = Tokenizer(num_words=token_size, oov_token="<OOV>")
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
total_words = len(tokenizer.word_index)

# Create training sequences.
training_sequences = tokenizer.texts_to_sequences(training_sentences)

training_data = []
for seq in training_sequences:
    training_data.append(sequenceToBinary(seq))

training_data = numpy.array(training_data)
training_labels = numpy.array(training_labels)

# Create model.
model = keras.Sequential([
     keras.layers.Dense(32, activation="relu",
                       input_shape=(total_words,)),
     keras.layers.Dense(16, activation="relu"),
     keras.layers.Dense(totalTags, activation="softmax")
])

# Set loss function and optimizer.
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="adam", metrics=["accuracy"])

# Train model.
model.fit(training_data, training_labels, epochs=epochs,
          batch_size=batchSize, verbose=1)

print("Chat with bot (type exit to stop")
while True:
     text = input("You: ")
     if text == "exit":
          break
     s = tokenizer.texts_to_sequences([text])
     inputData = numpy.array([sequenceToBinary(s[0])])
     prediction = model.predict(inputData)

     # Get max value.
     label = 0
     for i in range(totalTags):
          if prediction[0][label] < prediction[0][i]:
               label = i
     if prediction[0][label] < 0.5:
          print("Bot: " , "what?")
     else:          
          tag = labels[label]
          for intent in data['intents']:
               if tag == intent["tag"]:
                    print("Bot: " , random.choice(intent["responses"]))
