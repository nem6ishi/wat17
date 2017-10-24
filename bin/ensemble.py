#! /usr/bin/env python
# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" Generates model predictions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from pydoc import locate

import yaml
from six import string_types

import tensorflow as tf
from tensorflow import gfile

from seq2seq import tasks
from seq2seq import models as model_clsses
from seq2seq.configurable import _maybe_load_yaml, _deep_merge_dict
from seq2seq.data import input_pipeline
from seq2seq.inference import create_inference_graph
from seq2seq.training import utils as training_utils

from seq2seq.models.ensemble_model import EnsembleModel

tf.flags.DEFINE_string("tasks", "{}", "List of inference tasks to run.")
tf.flags.DEFINE_string("model_params", "{}", """Optionally overwrite model
                        parameters for inference""")

tf.flags.DEFINE_string("config_path", None,
                       """Path to a YAML configuration file defining FLAG
                       values and hyperparameters. Refer to the documentation
                       for more details.""")

tf.flags.DEFINE_string("models", "{}", "List of models to do ensemble")

tf.flags.DEFINE_string("input_pipeline", None,
                       """Defines how input data should be loaded.
                       A YAML string.""")

tf.flags.DEFINE_integer("batch_size", 32, "the train/dev batch size")

FLAGS = tf.flags.FLAGS

def main(_argv):
  """Program entry point.
  """

  # Load flags from config file
  if FLAGS.config_path:
    with gfile.GFile(FLAGS.config_path) as config_file:
      config_flags = yaml.load(config_file)
      for flag_key, flag_value in config_flags.items():
        setattr(FLAGS, flag_key, flag_value)

  if isinstance(FLAGS.tasks, string_types):
    FLAGS.tasks = _maybe_load_yaml(FLAGS.tasks)

  if isinstance(FLAGS.input_pipeline, string_types):
    FLAGS.input_pipeline = _maybe_load_yaml(FLAGS.input_pipeline)

  if isinstance(FLAGS.model_params, string_types):
      FLAGS.model_params = _maybe_load_yaml(FLAGS.model_params)

  if isinstance(FLAGS.models, string_types):
      FLAGS.models = _maybe_load_yaml(FLAGS.models)
      for mdict in FLAGS.models:
          if 'params' not in mdict:
              mdict['params'] = {}

  input_pipeline_infer = input_pipeline.make_input_pipeline_from_def(
      FLAGS.input_pipeline, mode=tf.contrib.learn.ModeKeys.INFER,
      shuffle=False, num_epochs=1)

  # ---------- Load Models First to Load Model Paramerters ----------
  model_variables = []

  for mdict in FLAGS.models:
    # Load saved training options
    train_options = training_utils.TrainOptions.load(mdict['dir'])
    
    # Get the model class
    model_cls = locate(train_options.model_class) or getattr(model_clsses, train_options.model_class)
    
    # Load model params
    model_params = train_options.model_params
    model_params = _deep_merge_dict(model_params, mdict['params'])
    model_params = _deep_merge_dict(model_params, FLAGS.model_params)

    # Create model
    model = model_cls(params=model_params, mode=tf.contrib.learn.ModeKeys.INFER)

    # Create computation graph
    predictions, _, _ = create_inference_graph(
            model=model,
            input_pipeline=input_pipeline_infer,
            batch_size=FLAGS.batch_size)

    # Get path to the checkpoint
    checkpoint_path = mdict['checkpoint_path'] if 'checkpoint_path' in mdict else tf.train.latest_checkpoint(mdict['dir'])

    # Get Saver
    saver = tf.train.Saver()

    # Create session to load values
    with tf.Session() as sess:
        # Load model values from checkpoint
        saver.restore(sess, checkpoint_path)

        # List all variables
        variables = {}
        for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            variables[var.name] = var.eval()
        
        model_variables.append(variables)

    # Reset graph
    tf.reset_default_graph()

  # Create computation graph for ensemble
  models = []
  vocab_source = None
  vocab_target = None

  for n, (mdict, variables) in enumerate(zip(FLAGS.models, model_variables)):
    # Load saved training options
    train_options = training_utils.TrainOptions.load(mdict['dir'])
    
    # Get the model class
    model_cls = locate(train_options.model_class) or getattr(model_clsses, train_options.model_class)
    
    # Load model params
    model_params = train_options.model_params
    model_params = _deep_merge_dict(model_params, mdict['params'])
    model_params = _deep_merge_dict(model_params, FLAGS.model_params)

    # Create model
    model = model_cls(params=model_params, mode=tf.contrib.learn.ModeKeys.INFER)

    models.append(model)
    
    # Predefine variables
    with tf.variable_scope('model{}'.format(n)):
      for name, value in variables.items():
        varname = name.split(':')[0]
        tf.get_variable(varname, shape=value.shape, initializer=tf.constant_initializer(value))

    # Create computation graph
    with tf.variable_scope('model{}'.format(n), reuse=True):
      predictions, _, _ = create_inference_graph(
              model=model,
              input_pipeline=input_pipeline_infer,
              batch_size=FLAGS.batch_size)
    
    # Get vocab informatin
    if 'vocab_source' in model_params:
      vocab_source = vocab_source if vocab_source else model_params['vocab_source']
      assert vocab_source == model_params['vocab_source'], 'Vocab Not Match'
    if 'vocab_target' in model_params:
      vocab_target = vocab_target if vocab_target else model_params['vocab_target']
      assert vocab_target == model_params['vocab_target'], 'Vocab Not Match'

  # Fill vocab info of model_params 
  if vocab_source:
    FLAGS.model_params['vocab_source'] = vocab_source
  if vocab_target:
    FLAGS.model_params['vocab_target'] = vocab_target
  
  # Create Ensemble Models
  ensemble_model = EnsembleModel(models=models, params=FLAGS.model_params)

  # Create Computation Graph
  predictions, _, _ = create_inference_graph(ensemble_model, input_pipeline_infer, FLAGS.batch_size)

  # DEBUG
  #for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
  #  print(var.name)

  #exit();

  # Load inference tasks
  hooks = []
  for tdict in FLAGS.tasks:
    if not "params" in tdict:
      tdict["params"] = {}
    task_cls = locate(tdict["class"]) or getattr(tasks, tdict["class"])
    task = task_cls(tdict["params"])
    hooks.append(task)

  with tf.train.MonitoredSession(
      hooks=hooks) as sess:

    # Run until the inputs are exhausted
    while not sess.should_stop():
      sess.run([])

if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
