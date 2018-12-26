# -*-encoding:utf-8-*-
from functools import reduce
from operator import mul

import numpy as np
import tensorflow as tf
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode

from model_utils import result_to_json
from bert_model import BertConfig, BertModel


def gelu(input_tensor):
    cdf = .5 * (1. + tf.erf(input_tensor / tf.sqrt(2.)))
    return input_tensor * cdf


def flatten(tensor, keep):
    fixed_shape = tensor.get_shape().as_list()
    start = len(fixed_shape) - keep
    left = reduce(mul, [fixed_shape[i] or tf.shape(tensor)[i] for i in range(start)])
    out_shape = [left] + [fixed_shape[i] or tf.shape(tensor)[i] for i in range(start, len(fixed_shape))]
    flat = tf.reshape(tensor, out_shape)
    return flat


def reconstruct(tensor, ref, keep, remove_shape=None):
    ref_shape = ref.get_shape().as_list()
    tensor_shape = tensor.get_shape().as_list()
    ref_stop = len(ref_shape) - keep
    tensor_start = len(tensor_shape) - keep
    if remove_shape is not None:
        tensor_start = tensor_start + remove_shape
    pre_shape = [ref_shape[i] or tf.shape(ref)[i] for i in range(ref_stop)]
    keep_shape = [tensor_shape[i] or tf.shape(tensor)[i] for i in range(tensor_start, len(tensor_shape))]
    target_shape = pre_shape + keep_shape
    out = tf.reshape(tensor, target_shape)
    return out


def deep_bidirectional_dynamic_rnn(cells, inputs, sequence_length, name=""):
    def _reverse(_input, seq_lengths):
        return tf.reverse_sequence(input=_input, seq_lengths=seq_lengths, seq_axis=1, batch_axis=0)

    outputs, state = None, None
    with tf.variable_scope("dblstm-%s" % name):
        for i, cell in enumerate(cells):
            if i % 2 == 1:
                with tf.variable_scope("bw-%s" % (i // 2)) as bw_scope:
                    inputs_reverse = _reverse(inputs, seq_lengths=sequence_length)
                    outputs, state = tf.nn.dynamic_rnn(cell=cell, inputs=inputs_reverse,
                                                       sequence_length=sequence_length,
                                                       dtype=tf.float32, scope=bw_scope)
                    # outputs, state = tf.nn.bidirectional_dynamic_rnn(cell_fw=_cell(dims),
                    #                                                  cell_bw=_cell(dims),
                    #                                                  inputs=inputs_reverse,
                    #                                                  sequence_length=sequence_length,
                    #                                                  dtype=tf.float32, scope=bw_scope)
                    outputs = _reverse(outputs, seq_lengths=sequence_length)
            else:
                with tf.variable_scope("fw-%s" % (i // 2)) as fw_scope:
                    outputs, state = tf.nn.dynamic_rnn(cell=cell, inputs=inputs, sequence_length=sequence_length,
                                                       dtype=tf.float32, scope=fw_scope)
                    # outputs, state = tf.nn.bidirectional_dynamic_rnn(cell_fw=_cell(dims),
                    #                                                  cell_bw=_cell(dims),
                    #                                                  inputs=inputs,
                    #                                                  sequence_length=sequence_length,
                    #                                                  dtype=tf.float32, scope=fw_scope)
            inputs = outputs
    return outputs, state


class BiRNN:

    def __init__(self, num_units, keep_prob, cell_type='lstm', scope=None):
        self.keep_prob = keep_prob
        self.cell_fw = tf.contrib.rnn.GRUCell(num_units) if cell_type == 'gru' \
            else self._build_single_cell(num_units)
        self.cell_bw = tf.contrib.rnn.GRUCell(num_units) if cell_type == 'gru' \
            else self._build_single_cell(num_units)
        self.scope = scope or "bi_rnn"

    def _build_single_cell(self, lstm_units):
        cell = tf.contrib.rnn.CoupledInputForgetGateLSTMCell(lstm_units, use_peepholes=True, state_is_tuple=True)
        cell = tf.contrib.rnn.HighwayWrapper(cell)
        cell = tf.contrib.rnn.DropoutWrapper(cell, variational_recurrent=True, output_keep_prob=self.keep_prob,
                                             dtype=tf.float32)
        cell = tf.contrib.rnn.ResidualWrapper(cell)
        return cell

    def __call__(self, inputs, seq_len, use_last_state=False, time_major=False):
        assert not time_major, "BiRNN class cannot support time_major currently"
        with tf.variable_scope(self.scope):
            flat_inputs = flatten(inputs, keep=2)  # reshape to [-1, max_time, dim]
            seq_len = flatten(seq_len, keep=0)  # reshape to [x] (one dimension sequence)
            outputs, ((_, h_fw), (_, h_bw)) = tf.nn.bidirectional_dynamic_rnn(self.cell_fw, self.cell_bw, flat_inputs,
                                                                              sequence_length=seq_len, dtype=tf.float32)
            if use_last_state:  # return last states
                output = tf.concat([h_fw, h_bw], axis=-1)  # shape = [-1, 2 * num_units]
                output = reconstruct(output, ref=inputs, keep=2, remove_shape=1)  # remove the max_time shape
            else:
                output = tf.concat(outputs, axis=-1)  # shape = [-1, max_time, 2 * num_units]
                output = reconstruct(output, ref=inputs, keep=2)  # reshape to same as inputs, except the last two dim
            return output


class DenselyConnectedBiRNN:
    """Implement according to Densely Connected Bidirectional LSTM with Applications to Sentence Classification
       https://arxiv.org/pdf/1802.00889.pdf"""

    def __init__(self, num_layers, num_units, num_last_units, keep_prob, cell_type='lstm', scope=None):
        if type(num_units) == list:
            assert len(num_units) == num_layers, "if num_units is a list, then its size should equal to num_layers"
        self.units = num_units
        self.keep_prob = keep_prob
        self.dense_bi_rnn = []
        for i in range(num_layers):
            units = num_units[i] if type(num_units) == list else num_units
            if i < num_layers - 1:
                self.dense_bi_rnn.append(BiRNN(units, keep_prob, cell_type=cell_type,
                                               scope='bi_rnn_{}'.format(i)))
            else:
                self.dense_bi_rnn.append(BiRNN(num_last_units, keep_prob, cell_type=cell_type,
                                               scope='bi_rnn_{}'.format(i)))
        self.num_layers = num_layers
        self.scope = scope or "densely_connected_bi_rnn"

    def _dblstm_cell(self, dims):
        cell = tf.contrib.rnn.CoupledInputForgetGateLSTMCell(dims, use_peepholes=True, state_is_tuple=True)
        # cell = tf.contrib.rnn.LayerNormBasicLSTMCell(dims)
        # cell = tf.contrib.rnn.GLSTMCell(dims, initializer=self.he_uni, number_of_groups=4)
        # cell = tf.contrib.rnn.HighwayWrapper(cell)
        # cell = tf.contrib.rnn.AttentionCellWrapper(cell, attn_length=dims, state_is_tuple=True)
        cell = tf.contrib.rnn.DropoutWrapper(cell, variational_recurrent=True, output_keep_prob=self.keep_prob,
                                             dtype=tf.float32)
        # cell = tf.contrib.rnn.ResidualWrapper(cell)
        return cell

    def __call__(self, inputs, seq_len, time_major=False):
        assert not time_major, "DenseConnectBiRNN class cannot support time_major currently"
        # this function does not support return_last_state method currently
        with tf.variable_scope(self.scope):
            flat_inputs = flatten(inputs, keep=2)  # reshape to [-1, max_time, dim]
            seq_len = flatten(seq_len, keep=0)  # reshape to [x] (one dimension sequence)
            cur_inputs = flat_inputs
            for i in range(self.num_layers):
                cur_outputs, _ = deep_bidirectional_dynamic_rnn([self._dblstm_cell(self.units) for _ in range(2)],
                                                                cur_inputs, seq_len, "%d" % i)
                # self.dense_bi_rnn[i](cur_inputs, seq_len)
                if i < self.num_layers - 1:
                    cur_inputs = tf.concat([cur_inputs, cur_outputs], axis=-1)
                else:
                    cur_inputs = cur_outputs
            output = reconstruct(cur_inputs, ref=inputs, keep=2)
            return output


class Model(object):

    def __init__(self, sess, config):
        self.sess = sess
        self.config = config
        self.lr = config["lr"]
        self.char_dim = config["char_dim"]
        self.char_lstm_dim = config["char_lstm_dim"]
        self.word_lstm_dim = config["word_lstm_dim"]

        self.num_tags = config["num_tags"]
        self.num_chars = config["num_chars"]

        self.global_step = tf.Variable(0, trainable=False)
        self.best_dev_f1 = tf.Variable(0.0, trainable=False)
        self.best_test_f1 = tf.Variable(0.0, trainable=False)

        # self.batch_size = tf.placeholder(tf.int32, [], name='batch_size')

        self.seed = 1337

        tf.set_random_seed(self.seed)

        self.l2_reg = tf.contrib.layers.l2_regularizer(5e-4)
        self.he_uni = tf.contrib.layers.variance_scaling_initializer(factor=3., mode='FAN_AVG', uniform=True)

        # add placeholders for the model
        self.char_inputs = tf.placeholder(dtype=tf.int32,
                                          # [batch, word_in_sen, char_in_word]
                                          shape=[None, None, config["max_char_length"]],
                                          name="CharInputs")
        self.targets = tf.placeholder(dtype=tf.int32, shape=[None, None], name="Targets")

        # dropout keep prob
        self.dropout = tf.placeholder(dtype=tf.float32, name="Dropout")

        used = tf.sign(tf.abs(self.char_inputs))  # 존재하는 곳에 1인 mask
        char_length = tf.reduce_sum(used, reduction_indices=2)
        word_length = tf.reduce_sum(tf.sign(char_length), reduction_indices=1)
        self.word_lengths = tf.cast(word_length, tf.int32)
        self.batch_size = tf.shape(self.char_inputs)[0]
        self.word_num_steps = tf.shape(self.char_inputs)[-2]

        # embeddings for chinese character and segmentation representation
        # apply dropout before feed to lstm layer
        embedding = self.embedding_layer(self.char_inputs, name="x")
        embedding = tf.nn.dropout(embedding, self.dropout)
        print("[*] embedding size : ", embedding.get_shape().as_list())

        # position-encoding
        word_encoded = self.get_word_representation(embedding)
        print("[*] word encoded size : ", word_encoded.get_shape().as_list())

        """
        # BERT
        word_bert_config = BertConfig(vocab_size=self.config["num_chars"],
                                      hidden_size=self.config["char_dim"],
                                      num_hidden_layers=self.config["num_hidden_layers"],
                                      num_attention_heads=self.config["num_attention_heads"],
                                      intermediate_size=self.config["intermediate_size"],
                                      max_position_embeddings=self.config["max_word_length"],
                                      type_vocab_size=self.config["num_tags"])

        word_bert_model = BertModel(config=word_bert_config,
                                    is_training=self.config["mode"] == "train",
                                    input_ids=word_encoded,
                                    input_mask=None,
                                    token_type_ids=None,
                                    use_one_hot_embeddings=False,
                                    scope="WordBertModel")

        word_bert_output = word_bert_model.get_sequence_output()
        print("[*] Word BERT output : ", word_bert_output.get_shape().as_list())

        # word_encoded = tf.concat([word_encoded, word_bert_output], axis=-1)
        word_encoded += word_bert_output
        print("[*] Concatenated output : ", word_encoded.get_shape().as_list())
        """

        with tf.variable_scope('deep_bidirectional_rnn'):
            # outputs, stat_x = deep_bidirectional_dynamic_rnn([self._dblstm_cell(self.word_lstm_dim)
            #                                                   for _ in range(2)],
            #                                                  word_encoded, sequence_length=self.word_lengths)
            outputs, _ = deep_bidirectional_dynamic_rnn([self._dblstm_cell(self.word_lstm_dim) for _ in range(2)],
                                                        word_encoded, sequence_length=self.word_lengths, name="1")

            print("[*] deep biLSTM size : ", outputs.get_shape().as_list())

        dc_outputs = DenselyConnectedBiRNN(self.config["num_dc_layers"],
                                           self.word_lstm_dim,
                                           self.word_lstm_dim,
                                           self.dropout)(outputs, self.word_lengths)
        # outputs = tf.concat([outputs, dc_outputs], axis=-1)
        outputs = dc_outputs
        # print("[*] last cont size : ", outputs.get_shape().as_list())

        # logits for tags
        outputs = tf.reshape(outputs, (-1, outputs.get_shape().as_list()[-1]))
        self.logits = self.project_layer(outputs)

        # loss of the model
        self.loss = self.loss_layer(self.logits, self.word_lengths)

        with tf.variable_scope("optimizer", reuse=tf.AUTO_REUSE):
            optimizer = self.config["optimizer"]
            lr = self.config["lr"]
            lr_decay = self.config["lr_decay"]

            learning_rate = tf.train.exponential_decay(lr,
                                                       self.global_step,
                                                       1643 * 1,  # 1593 * 1,  # 1 epoch
                                                       lr_decay,
                                                       staircase=True)

            self.lr = tf.clip_by_value(learning_rate,
                                       clip_value_min=8e-5,
                                       clip_value_max=lr,
                                       name='lr-clipped')

            if optimizer == "sgd":
                # self.opt = tf.train.GradientDescentOptimizer(self.lr)
                self.opt = tf.train.MomentumOptimizer(self.lr, momentum=.9, use_nesterov=True)
            elif optimizer == "adam":
                self.opt = tf.train.AdamOptimizer(self.lr, epsilon=1e-6)
            elif optimizer == "adadelta":
                self.opt = tf.train.AdadeltaOptimizer(self.lr, epsilon=1e-6)
            else:
                raise KeyError

            # apply grad clip to avoid gradient explosion
            gradients, variables = zip(*self.opt.compute_gradients(self.loss))
            gradients, _ = tf.clip_by_global_norm(gradients, self.config['clip'])
            self.train_op = self.opt.apply_gradients(zip(gradients, variables), global_step=self.global_step)

            # grads_vars = self.opt.compute_gradients(self.loss)
            # capped_grads_vars = [[tf.clip_by_value(g, -self.config["clip"], self.config["clip"]), v]
            #                      for g, v in grads_vars]
            # self.train_op = self.opt.apply_gradients(capped_grads_vars, self.global_step)

        # saver of the model
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=7)
        self.writer = tf.summary.FileWriter('./model/', self.sess.graph)

    @staticmethod
    def reconstruct(tensor, ref, keep, remove_shape=None):
        ref_shape = ref.get_shape().as_list()
        tensor_shape = tensor.get_shape().as_list()
        ref_stop = len(ref_shape) - keep
        tensor_start = len(tensor_shape) - keep
        if remove_shape is not None:
            tensor_start = tensor_start + remove_shape
        pre_shape = [ref_shape[i] or tf.shape(ref)[i] for i in range(ref_stop)]
        keep_shape = [tensor_shape[i] or tf.shape(tensor)[i] for i in range(tensor_start, len(tensor_shape))]
        target_shape = pre_shape + keep_shape
        out = tf.reshape(tensor, target_shape)
        return out

    @staticmethod
    def _reverse(_input, seq_lengths):
        return tf.reverse_sequence(input=_input, seq_lengths=seq_lengths, seq_axis=1, batch_axis=0)

    def _dblstm_cell(self, dims):
        cell = tf.contrib.rnn.CoupledInputForgetGateLSTMCell(dims,
                                                             use_peepholes=True,
                                                             initializer=self.he_uni,
                                                             state_is_tuple=True)
        # cell = tf.contrib.rnn.LayerNormBasicLSTMCell(dims)
        # cell = tf.contrib.rnn.GLSTMCell(dims, initializer=self.he_uni, number_of_groups=4)
        cell = tf.contrib.rnn.HighwayWrapper(cell)
        # cell = tf.contrib.rnn.AttentionCellWrapper(cell, attn_length=dims, state_is_tuple=True)
        cell = tf.contrib.rnn.DropoutWrapper(cell, variational_recurrent=True, output_keep_prob=self.dropout,
                                             dtype=tf.float32)
        cell = tf.contrib.rnn.ResidualWrapper(cell)
        return cell

    @staticmethod
    def _char_embedding_init(dim):
        scale = tf.sqrt(3. / dim)
        return tf.random_uniform_initializer(-scale, scale)

    @staticmethod
    def _position_encoding(sentence_size, embedding_size):
        encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
        ls = sentence_size + 1
        le = embedding_size + 1
        for i in range(1, le):
            for j in range(1, ls):
                encoding[i - 1, j - 1] = (i - (le - 1) / 2) * (j - (ls - 1) / 2)
        encoding = 1 + 4 * encoding / embedding_size / sentence_size
        return np.transpose(encoding)

    def get_word_representation(self, embedding):
        position_encoding_mat = self._position_encoding(self.config["max_char_length"], self.char_dim)
        position_encoded = tf.reduce_sum(embedding * position_encoding_mat, 2)
        return position_encoded

    def embedding_layer(self, char_inputs, name=None):
        """
        :param char_inputs: one-hot encoding of sentence
        :return: [1, num_steps, embedding size],
        """

        with tf.variable_scope("char_embedding" if not name else name, reuse=tf.AUTO_REUSE), tf.device('/cpu:0'):
            self.char_lookup = tf.get_variable(
                name="char_embedding",
                shape=[self.num_chars, self.char_dim],
                initializer=self._char_embedding_init(self.char_dim))
            embed = tf.nn.embedding_lookup(self.char_lookup, char_inputs)
        return embed

    def biLSTM_layer(self, inputs, lstm_dim, lengths, name=None):
        with tf.variable_scope("char_BiLSTM" if not name else name, reuse=tf.AUTO_REUSE):
            lstm_cell = {}
            for direction in ["forward", "backward"]:
                with tf.variable_scope(direction, reuse=tf.AUTO_REUSE):
                    lstm_cell[direction] = self._dblstm_cell(lstm_dim)

            outputs, final_states = tf.nn.bidirectional_dynamic_rnn(
                lstm_cell["forward"],
                lstm_cell["backward"],
                inputs,
                dtype=tf.float32,
                sequence_length=lengths)

            outputs = tf.concat(outputs, axis=2)

        return outputs

    def project_layer(self, lstm_outputs, name=None):
        """
        hidden layer between lstm layer and logits
        :param lstm_outputs: [batch_size, num_steps, emb_size]
        :return: [batch_size, num_steps, num_tags]
        """
        with tf.variable_scope("project" if not name else name, reuse=tf.AUTO_REUSE):
            with tf.variable_scope("hidden", reuse=tf.AUTO_REUSE):
                x = lstm_outputs
                x = tf.layers.dense(x, x.get_shape().as_list()[-1] // 2,
                                    kernel_initializer=self.he_uni,
                                    kernel_regularizer=self.l2_reg,
                                    bias_initializer=tf.zeros_initializer())
                x = tf.nn.leaky_relu(x, alpha=0.2)
                # x = tf.nn.tanh(x)
                # x = gelu(x)

                # x += lstm_outputs

            # project to score of tags
            with tf.variable_scope("logits", reuse=tf.AUTO_REUSE):
                x = tf.layers.dense(x, units=self.num_tags,
                                    kernel_initializer=self.he_uni,
                                    kernel_regularizer=self.l2_reg,
                                    bias_initializer=tf.zeros_initializer())

                pred = tf.reshape(x, (-1, self.word_num_steps, self.num_tags))

            return pred

    def loss_layer(self, project_logits, lengths, name=None):
        """
        calculate crf loss
        :param project_logits: [1, num_steps, num_tags]
        :return: scalar loss
        """
        with tf.variable_scope("crf_loss" if not name else name, reuse=tf.AUTO_REUSE):
            small = -1000.0

            # pad logits for crf loss
            start_logits = tf.concat([
                small * tf.ones(shape=[self.batch_size, 1, self.num_tags]),
                tf.zeros(shape=[self.batch_size, 1, 1])
            ], axis=-1)
            pad_logits = tf.cast(small * tf.ones([self.batch_size, self.word_num_steps, 1]), dtype=tf.float32)

            logits = tf.concat([project_logits, pad_logits], axis=-1)
            logits = tf.concat([start_logits, logits], axis=1)
            targets = tf.concat(
                [tf.cast(self.num_tags * tf.ones([self.batch_size, 1]), tf.int32), self.targets], axis=-1)

            self.trans = tf.get_variable(
                "transitions",
                shape=[self.num_tags + 1, self.num_tags + 1],
                initializer=self.he_uni,
                regularizer=self.l2_reg
            )

            log_likelihood, self.trans = crf_log_likelihood(
                inputs=logits,
                tag_indices=targets,
                transition_params=self.trans,
                sequence_lengths=lengths + 1)
            return tf.reduce_mean(-log_likelihood)

    def create_feed_dict(self, is_train, batch):
        """
        :param is_train: Flag, True for train batch
        :param batch: list train/evaluate data
        :return: structured data to feed
        """
        _, chars, tags = batch
        input_chars = np.array(chars)
        feed_dict = {
            self.char_inputs: input_chars,
            self.dropout: 1.0,
            # self.batch_size: self.config["batch_size"] if is_train else input_chars.shape[0],
        }
        '''
        print ('chars')
        print (chars)
        print ('after chars')
        print (feed_dict[self.char_inputs])
        print 
        '''
        if is_train:
            feed_dict[self.targets] = np.asarray(tags)
            feed_dict[self.dropout] = self.config["dropout_keep"]
            '''
            print ('tags')
            print (tags)
            print ('after tags')
            print (feed_dict[self.targets])
            print
            '''
        return feed_dict

    def run_step(self, sess, is_train, batch):
        """
        :param sess: session to run the batch
        :param is_train: a flag indicate if it is a train batch
        :param batch: a dict containing batch data
        :return: batch result, loss of the batch or logits
        """
        feed_dict = self.create_feed_dict(is_train, batch)

        if is_train:
            global_step, loss, _ = sess.run(
                [self.global_step, self.loss, self.train_op],
                feed_dict)
            return global_step, loss
        else:
            lengths, logits = sess.run([self.word_lengths, self.logits], feed_dict)
            return lengths, logits

    def decode(self, logits, lengths, matrix):
        """
        :param logits: [batch_size, num_steps, num_tags]float32, logits
        :param lengths: [batch_size]int32, real length of each sequence
        :param matrix: transaction matrix for inference
        :return:
        """
        # inference final labels usa viterbi Algorithm
        paths = []
        small = -1000.0
        start = np.asarray([[small] * self.num_tags + [0]])
        for score, length in zip(logits, lengths):
            score = score[:length]
            pad = small * np.ones([length, 1])
            logits = np.concatenate([score, pad], axis=1)
            logits = np.concatenate([start, logits], axis=0)
            path, _ = viterbi_decode(logits, matrix)
            paths.append(path[1:])
        return paths

    def evaluate_model(self, sess, data_manager, id_to_tag):
        """
        :param sess: session  to run the model
        :param data: list of data
        :param id_to_tag: index to tag name
        :return: evaluate result
        """
        results = []
        trans = self.trans.eval(session=sess)
        for batch in data_manager.iter_batch():
            strings = batch[0]
            tags = batch[-1]
            lengths, scores = self.run_step(sess, False, batch)
            batch_paths = self.decode(scores, lengths, trans)
            for i in range(len(strings)):
                result = []
                string = strings[i][:lengths[i]]
                gold = [id_to_tag[int(x)] for x in tags[i][:lengths[i]]]
                pred = [id_to_tag[int(x)] for x in batch_paths[i][:lengths[i]]]
                for char, gold, pred in zip(string, gold, pred):
                    result.append(" ".join([''.join(char), gold, pred]))
                results.append(result)
        return results

    def evaluate_lines(self, sess, inputs, id_to_tag):
        trans = self.trans.eval(session=sess)
        lengths, scores = self.run_step(sess, False, inputs)
        batch_paths = self.decode(scores, lengths, trans)
        total_tags = [[id_to_tag[idx] for idx in path] for path in batch_paths]
        return [(0.0, tag) for tag in total_tags]
