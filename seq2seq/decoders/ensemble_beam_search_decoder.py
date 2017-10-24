import tensorflow as tf

from seq2seq.decoders.beam_search_decoder import BeamSearchDecoder, FinalBeamDecoderOutput, BeamDecoderOutput
from seq2seq.inference import beam_search
from seq2seq.inference.beam_search import BeamSearchState, BeamSearchStepOutput, mask_probs, nest

class EnsembleBeamSearchDecoder(BeamSearchDecoder):
    def __init__(self, decoder, config):
        super(EnsembleBeamSearchDecoder, self).__init__(decoder, config)

    def step(self, time_, inputs, state, name=None):
        decoder_state, beam_state = state

        # Run decoder step
        decoder_output, decoder_state, _, _ = self.decoder.step(time_, inputs, decoder_state)

        # Get top k
        bs_output, beam_state = ensemble_beam_search_step(
                time_=time_,
                probs=decoder_output.prob,
                beam_state=beam_state,
                config=self.config)

        # Choosen part
        decoder_state = nest.map_structure(
                lambda x: tf.gather(x, bs_output.beam_parent_ids), decoder_state)
        decoder_output = nest.map_structure(
                lambda x: tf.gather(x, bs_output.beam_parent_ids), decoder_output)
        
        # Combine to get next state
        next_state = (decoder_state, beam_state)

        outputs = BeamDecoderOutput(
                logits=tf.zeros([self.config.beam_width, self.config.vocab_size]),
                predicted_ids=bs_output.predicted_ids,
                log_probs=beam_state.log_probs,
                scores=bs_output.scores,
                beam_parent_ids=bs_output.beam_parent_ids,
                original_outputs=decoder_output)

        ens_next_state = []
        ens_next_inputs = []
        finished = None
        for dec, original_output, state in zip(self.decoder.decoders, decoder_output.original_outputs, decoder_state):
            finished, next_inputs, next_state = dec.helper.next_inputs(
                    time=time_,
                    outputs=original_output,
                    state=state,
                    sample_ids=bs_output.predicted_ids)

            next_inputs.set_shape([self.batch_size, None])

            ens_next_state.append(next_state)
            ens_next_inputs.append(next_inputs)

        return (outputs, (ens_next_state, beam_state), ens_next_inputs, finished)


def ensemble_beam_search_step(time_, probs, beam_state, config):
    # Current length of prediction
    prediction_lengths = beam_state.lengths
    previously_finished = beam_state.finished

    # Calculate log prob for new hypothetheses
    log_probs = tf.log(probs)
    log_probs = mask_probs(log_probs, config.eos_token, previously_finished)
    total_log_prob = tf.expand_dims(beam_state.log_probs, 1) + log_probs

    # Update prediction lengths
    lengths_to_add = tf.one_hot([config.eos_token] * config.beam_width, config.vocab_size, 0, 1)
    add_mask = 1 - tf.to_int32(previously_finished)
    lengths_to_add = tf.expand_dims(add_mask, 1) * lengths_to_add
    new_prediction_lengths = tf.expand_dims(prediction_lengths, 1) + lengths_to_add

    # Get score
    scores = beam_search.hyp_score(
            log_probs=total_log_prob,
            sequence_lengths=new_prediction_lengths,
            config=config)

    # scoreは[現在までのtop k, 今回のtop k]の形なのでそれを[k*k]にする
    scores_flat = tf.reshape(scores, [-1])
    scores_flat = tf.cond(
            tf.convert_to_tensor(time_) > 0,
            lambda: scores_flat,
            lambda: scores[0])

    # Basically get top k
    next_beam_scores, word_indices = config.choose_successors_fn(scores_flat, config)

    next_beam_scores.set_shape([config.beam_width])
    word_indices.set_shape([config.beam_width])


    total_probs_flat = tf.reshape(total_log_prob, [-1], name="total_probs_flat")
    next_beam_probs = tf.gather(total_probs_flat, word_indices)
    next_beam_probs.set_shape([config.beam_width])
    next_word_ids = tf.mod(word_indices, config.vocab_size)
    next_beam_ids = tf.div(word_indices, config.vocab_size)

    # Update Finished
    next_finished = tf.logical_or(
            tf.gather(beam_state.finished, next_beam_ids),
            tf.equal(next_word_ids, config.eos_token))

    # Calculate the length of the next predictions.
    # 1. Finished beams remain unchanged
    # 2. Beams that are now finished (EOS predicted) remain unchanged
    # 3. Beams that are not yet finished have their length increased by 1
    lengths_to_add = tf.to_int32(tf.not_equal(next_word_ids, config.eos_token))
    lengths_to_add = (1 - tf.to_int32(next_finished)) * lengths_to_add
    next_prediction_len = tf.gather(beam_state.lengths, next_beam_ids)
    next_prediction_len += lengths_to_add

    next_state = BeamSearchState(
          log_probs=next_beam_probs,
          lengths=next_prediction_len,
          finished=next_finished)

    output = BeamSearchStepOutput(
          scores=next_beam_scores,
          predicted_ids=next_word_ids,
          beam_parent_ids=next_beam_ids)

    return output, next_state


