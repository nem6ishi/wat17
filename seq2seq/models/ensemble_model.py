import tensorflow as tf

from seq2seq import graph_utils
from seq2seq.models.basic_seq2seq import BasicSeq2Seq

from seq2seq.inference import beam_search
from seq2seq.decoders.ensemble_beam_search_decoder import EnsembleBeamSearchDecoder
from seq2seq.decoders.ensemble_decoder import EnsembleDecoder

from seq2seq.models.model_base import _flatten_dict
from seq2seq.contrib.seq2seq.decoder import _transpose_batch_time

import collections

class EnsembleModel(BasicSeq2Seq):
  def __init__(self, models, params=None, mode=tf.contrib.learn.ModeKeys.INFER, name='ensemble_model'):
    super(EnsembleModel, self).__init__(params, mode, name)
    self.models = models
  
  def encode(self, features, labels):
    return [model.encode(features, labels) for model in self.models]

  def decode(self, encoder_outputs, features, labels):
    # ToDo: Special Decoder for Ensemble
    decoder = EnsembleDecoder(
        decoders=[model.decoder for model in self.models],
        params=self.params['decoder.params'],
        mode=self.mode,
        vocab_size=self.target_vocab_info.total_size)

    # Check for beam search
    if self.use_beam_search:
      decoder = self._get_beam_search_decoder(decoder)

    # Get decoder initial state
    decoder_initial_states = [model.bridge() for model in self.models]

    # Get decode result
    # Note: as each decoder has information of helper, we give None for global helper
    return decoder(decoder_initial_states, None)

  def _get_beam_search_decoder(self, decoder):
    # Create configuration of beam search
    config = beam_search.BeamSearchConfig(
        beam_width=self.params["inference.beam_search.beam_width"],
        vocab_size=self.target_vocab_info.total_size,
        eos_token=self.target_vocab_info.special_vocab.SEQUENCE_END,
        length_penalty_weight=self.params["inference.beam_search.length_penalty_weight"],
        choose_successors_fn=getattr(beam_search, self.params["inference.beam_search.choose_successors_fn"]))

    # ToDo: Special Beam Search Decoder for Ensemble
    return EnsembleBeamSearchDecoder(decoder, config)

  def _create_predictions(self, decoder_output, features, labels, losses=None):
    """Creates the dictionary of predictions that is returned by the model.
    """
    predictions = {}

    # Add features and, if available, labels to predictions
    predictions.update(_flatten_dict({"features": features}))
    if labels is not None:
      predictions.update(_flatten_dict({"labels": labels}))

    if losses is not None:
      predictions["losses"] = _transpose_batch_time(losses)

    # Decoders returns output in time-major form [T, B, ...]
    # Here we transpose everything back to batch-major for the user
    output_dict = collections.OrderedDict(zip(decoder_output._fields, decoder_output))
    decoder_output_flat = _flatten_dict(output_dict)
    decoder_output_flat = {
      k: _transpose_batch_time(v)
      for k, v in decoder_output_flat.items() if not isinstance(v, list) 
    }
    predictions.update(decoder_output_flat)

    # If we predict the ids also map them back into the vocab and process them
    if "predicted_ids" in predictions.keys():
      vocab_tables = graph_utils.get_dict_from_collection("vocab_tables")
      target_id_to_vocab = vocab_tables["target_id_to_vocab"]
      predicted_tokens = target_id_to_vocab.lookup(tf.to_int64(predictions["predicted_ids"]))
      # Raw predicted tokens
      predictions["predicted_tokens"] = predicted_tokens

    return predictions

