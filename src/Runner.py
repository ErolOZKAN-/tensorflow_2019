import tensorflow as tf
from util import *

tf.enable_eager_execution()

import numpy as np
import os
import functools

EPOCHS = 1

text = open("../data/irish.abc").read()  # Read Dataset
print(text[:250])  # Print first characters
vocab = sorted(set(text))  # Create vocab. The unique characters in the file
print('{} unique characters'.format(len(vocab)))  # Print vocab

char2idx = {chr: index for index, chr in enumerate(vocab)}  # Create mapping from unique characters to indices
text_as_int = np.array([char2idx[c] for c in text])  # All text as integer
idx2char = np.array(vocab)  # Create a mapping from indices to characters

print('{')
for char, _ in zip(char2idx, range(20)):
    print('  {:4s}: {:3d},'.format(repr(char), char2idx[char]))
print('  ...\n}')

print('{} ---- characters mapped to int ---- > {}'.format(repr(text[:13]), text_as_int[:13]))


def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text


seq_length = 100  # The maximum length sentence we want for a single input in characters
examples_per_epoch = len(text) // seq_length
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)  # Create training examples / targets sing the `tf.data` module!
sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)  # use the batch function to generate sequences of the desired size
dataset = sequences.map(split_input_target)  # the map method to apply your function to the list of sequences to generate the dataset

for input_example, target_example in dataset.take(1):
    for i, (input_idx, target_idx) in enumerate(zip(input_example[:5], target_example[:5])):
        print("Step {:4d}".format(i))
        print("  input: {} ({:s})".format(input_idx, repr(idx2char[input_idx])))
        print("  expected output: {} ({:s})".format(target_idx, repr(idx2char[target_idx])))

BATCH_SIZE = 64  # Batch size
BUFFER_SIZE = 10000

steps_per_epoch = examples_per_epoch // BATCH_SIZE
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
vocab_size = len(vocab)  # Length of the vocabulary in chars

embedding_dim = 256  # The embedding dimension
rnn_units = 1024  # The number of RNN units

LSTM = functools.partial(tf.keras.layers.LSTM, recurrent_activation='sigmoid')
LSTM = functools.partial(LSTM, return_sequences=True, recurrent_initializer='glorot_uniform', stateful=True)


def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
        LSTM(rnn_units),  # TODO: Define the dimensionality of the RNN
        tf.keras.layers.Dense(vocab_size)  # TODO: Define the dimensionality of the Dense layer
    ])
    return model


model = build_model(vocab_size=len(vocab),
                    embedding_dim=embedding_dim,
                    rnn_units=rnn_units,
                    batch_size=BATCH_SIZE)

print(model.summary())

for input_example_batch, target_example_batch in dataset.take(1):
    example_batch_predictions = model(input_example_batch)
    print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

sampled_indices = tf.random.multinomial(example_batch_predictions[0], num_samples=1)
sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()

print("Input: \n", repr("".join(idx2char[input_example_batch[0]])))
print()
print("Next Char Predictions: \n", repr("".join(idx2char[sampled_indices])))


def compute_loss(labels, logits):
    return tf.keras.backend.sparse_categorical_crossentropy(labels, logits, from_logits=True)


example_batch_loss = compute_loss(target_example_batch, example_batch_predictions)  # compute the loss using the example batch and predictions from above
print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
print("scalar_loss:      ", example_batch_loss.numpy().mean())

optimizer = tf.train.AdamOptimizer()
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

history = []
plotter = PeriodicPlotter(sec=1, xlabel='Iterations', ylabel='Loss')
for epoch in range(EPOCHS):

    # Initialize the hidden state at the start of every epoch; initially is None
    hidden = model.reset_states()

    # Enumerate the dataset for use in training
    custom_msg = custom_progress_text("Loss: %(loss)2.2f")
    bar = create_progress_bar(custom_msg)
    for inp, target in bar(dataset):
        # Use tf.GradientTape()
        with tf.GradientTape() as tape:
            '''TODO: feed the current input into the model and generate predictions'''
            predictions = model(inp)  # TODO
            '''TODO: compute the loss!'''
            loss = compute_loss(target, predictions)  # TODO

        # Now, compute the gradients and try to minimize
        '''TODO: complete the function call for gradient computation'''
        grads = tape.gradient(loss, model.trainable_variables)  # TODO
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Update the progress bar!
        history.append(loss.numpy().mean())
        custom_msg.update_mapping(loss=history[-1])
        plotter.plot(history)

    # Update the model with the changed weights!
    model.save_weights(checkpoint_prefix.format(epoch=epoch))

model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

model.build(tf.TensorShape([1, None]))
print(model.summary())


def generate_text(model, start_string, generation_length=1000):
    # Evaluation step (generating ABC text using the learned RNN model)

    '''TODO: convert the start string to numbers (vectorize)'''
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    # Empty string to store our results
    text_generated = []

    # Here batch size == 1
    model.reset_states()
    bar = create_progress_bar()
    for i in bar(range(generation_length)):
        '''TODO: evaluate the inputs and generate the next character predictions'''
        predictions = model(input_eval)  # TODO

        # Remove the batch dimension
        predictions = tf.squeeze(predictions, 0)

        '''TODO: use a multinomial distribution to sample'''
        predicted_id = tf.multinomial(predictions, num_samples=1)[-1, 0].numpy()  # TODO

        # Pass the prediction along with the previous hidden state
        # as the next inputs to the model
        input_eval = tf.expand_dims([predicted_id], 0)

        '''TODO: add the predicted character to the generated text!'''
        # Hint: consider what format the prediction is in, vs. the output
        text_generated.append(idx2char[predicted_id])  # TODO

    return (start_string + ''.join(text_generated))


'''TODO: Use the model to generate ABC format text!'''
# As you may notice, ABC files start with "X" - this may be a good start string
text = generate_text(model, start_string="X")
play_generated_song(text)
