import tensorflow as tf

from seq2seq.decoders.rnn_decoder import RNNDecoder

from seq2seq.decoders.attention_decoder import AttentionDecoderOutput

from tensorflow.python.ops import math_ops
from tensorflow.python.framework import dtypes

from collections import namedtuple

class EnsembleDecoderOutput(
    namedtuple("EnsembleDecoderOutput",
        ['predicted_ids', 'prob', 'original_outputs'])):
    pass

class EnsembleDecoder(RNNDecoder):
    def __init__(self, decoders, params, mode, vocab_size, name='ensemble_decoder'):
        super(EnsembleDecoder, self).__init__(params, mode, name)
        self.decoders = decoders
        self.vocab_size = vocab_size

    @property
    def output_size(self):
        return EnsembleDecoderOutput(
                predicted_ids=tf.TensorShape([]),
                prob=self.vocab_size,
                original_outputs=[decoder.output_size for decoder in self.decoders]
            )

    @property
    def output_dtype(self):
        return EnsembleDecoderOutput(
                predicted_ids=tf.int32,
                prob=tf.float32,
                original_outputs=[decoder.output_dtype for decoder in self.decoders]
            )

    def initialize(self):
        finished = None
        ens_first_inputs = []
        ens_initial_states = []

        for decoder in self.decoders:
            with decoder.variable_scope():
                with tf.variable_scope('decoder'):
                    finished, first_inputs, initial_states = decoder.initialize()
                    ens_first_inputs.append(first_inputs)
                    ens_initial_states.append(initial_states)

        return finished, ens_first_inputs, ens_initial_states
    
    def step(self, time_, ens_inputs, ens_states):
        original_outputs = []
        sum_prob = None
        _ens_states = []

        for decoder, inputs, states in zip(self.decoders, ens_inputs, ens_states):
            with decoder.variable_scope():
                with tf.variable_scope('decoder'):
                    # Run decoder step
                    outputs, state, _, _ = decoder.step(time_, inputs, states)

                    # Save original outputs
                    original_outputs.append(outputs)
                    _ens_states.append(state)

                    # Compute Probability
                    if sum_prob is None:
                        sum_prob = tf.nn.softmax(outputs.logits)
                    else:
                        sum_prob += tf.nn.softmax(outputs.logits)

        # Compute Average Prob
        average_prob = sum_prob / len(self.decoders)

        # Sample word
        sample_ids = math_ops.cast(math_ops.argmax(average_prob, axis=-1), dtype=tf.int32)

        # Compute state and inputs for next step
        ens_next_inputs = []
        ens_next_states = []
        finished = None

        for decoder, output, state in zip(self.decoders, original_outputs, _ens_states):
            updated_output = AttentionDecoderOutput(
                    logits=output.logits,
                    predicted_ids=sample_ids,
                    cell_output=output.cell_output,
                    attention_scores=output.attention_scores,
                    attention_context=output.attention_context)
            
            finished, next_inputs, next_states = decoder.helper.next_inputs(
                    time=time_, outputs=updated_output, state=state, sample_ids=sample_ids)

            ens_next_inputs.append(next_inputs)
            ens_next_states.append(next_states)

        # Get output
        ensemble_output = EnsembleDecoderOutput(
                predicted_ids=sample_ids,
                prob=average_prob,
                original_outputs=original_outputs)

        return ensemble_output, ens_next_states, ens_next_inputs, finished
