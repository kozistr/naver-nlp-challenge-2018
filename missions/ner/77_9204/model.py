# -*- coding: utf-8 -*-
from functools import reduce
from operator import mul

import tensorflow as tf

from bert_model import BertModel, BertConfig


def label_smoothing(inputs, epsilon=0.1):
    dim = inputs.get_shape().as_list()[-1]  # number of channels
    return ((1 - epsilon) * inputs) + (epsilon / dim)


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


class BiRNN:
    def __init__(self, num_units, keep_prob, cell_type='lstm', scope=None):
        self.keep_prob = keep_prob
        self.cell_fw = tf.contrib.rnn.GRUCell(num_units) if cell_type == 'gru' \
            else self._build_single_cell(num_units)
        self.cell_bw = tf.contrib.rnn.GRUCell(num_units) if cell_type == 'gru' \
            else self._build_single_cell(num_units)
        self.scope = scope or "bi_rnn"

    def _build_single_cell(self, lstm_units):
        cell = tf.contrib.rnn.LayerNormBasicLSTMCell(lstm_units)
        # cell = tf.contrib.rnn.HighwayWrapper(cell)
        cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=self.keep_prob, output_keep_prob=self.keep_prob)
        # cell = tf.contrib.rnn.ResidualWrapper(cell, lambda i, o: i + o)
        # cell = tf.contrib.rnn.DropoutWrapper(cell, variational_recurrent=True, dtype=tf.float32,
        #                                      output_keep_prob=self.keep_prob)
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

    def __call__(self, inputs, seq_len, time_major=False):
        assert not time_major, "DenseConnectBiRNN class cannot support time_major currently"
        # this function does not support return_last_state method currently
        with tf.variable_scope(self.scope):
            flat_inputs = flatten(inputs, keep=2)  # reshape to [-1, max_time, dim]
            seq_len = flatten(seq_len, keep=0)  # reshape to [x] (one dimension sequence)
            cur_inputs = flat_inputs
            for i in range(self.num_layers):
                cur_outputs = self.dense_bi_rnn[i](cur_inputs, seq_len)
                if i < self.num_layers - 1:
                    cur_inputs = tf.concat([cur_inputs, cur_outputs], axis=-1)
                else:
                    cur_inputs = cur_outputs
            output = reconstruct(cur_inputs, ref=inputs, keep=2)
            return output


class Model:

    def __init__(self, parameter):
        self.parameter = parameter

        self.l2_reg = tf.contrib.layers.l2_regularizer(1e-3)
        self.he_uni = tf.contrib.layers.variance_scaling_initializer(factor=3., mode='FAN_AVG', uniform=True)

        self._embeddings = list()
        self._embedding_matrix = list()

        self.matricized_unary_scores = None
        self.label = None

        self.cost = None
        self.train_op = None

        self.seed = 1337

        tf.set_random_seed(self.seed)

        print("[*] Parameter Information :")
        for k, v in parameter.items():
            print("  [*] {} : {}".format(k, v))

    def build_model(self):
        """
        Tried Models :
        Try 1 : (bi-LSTM (Char) + Word Embeddings) + bi-LSMT + CRF
        Try 2 : (bi-LSTM (Char) + Word Embeddings) + Highway Network x 2 + bi-LSTM + CRF
        Try 3 : (CharCNN (Char) + Word Embeddings) + Highwat Network x 2 + bi-LSTM + multi-head-attention + CRF
        Try 4 : (CharCNN (Char) + Word Embeddings) + Highway Network x 2 + Densely-Connected-bi-LSTM + CRF
        Try 5 : (bi-LSTM (Char) + Word Embeddings) + Highway Network x 2 + Densely-Connected-bi-LSTM + CRF : F1 66.8577
        Try 6 : (bi-LSTM (Char) + Word Embeddings) + Highway Network x 2 + Densely-Connected-bi-LSTM + Residual
        + CRF : F1 68.4807
        Try 7 : (bi-LSTM (Char) + Word Embeddings) + Densely-Connected-bi-LSTM x 8 + Residual + CRF : F1 71.5
        Try 8 : BERT + bi-LSTM + Residual + CRF :
        """
        self._build_placeholder()

        bert_config = BertConfig(vocab_size=self.parameter["embedding"][0][1],
                                 hidden_size=self.parameter["word_embedding_size"],
                                 max_position_embeddings=self.parameter["sentence_length"],
                                 type_vocab_size=self.parameter["n_class"])
        bert_model = BertModel(config=bert_config,
                               is_training=self.parameter["mode"] == "train",
                               input_ids=self.morph,
                               input_mask=None,
                               token_type_ids=None,
                               use_one_hot_embeddings=True,
                               scope="BertModel")

        bert_output = bert_model.get_sequence_output()
        # max_seq_length = bert_output.get_shape().as_list()[1]

        print("[*] BERT output : ", bert_output.get_shape().as_list())

        # used = tf.sign(tf.abs(self.morph))
        # lengths = tf.reduce_sum(used, reduction_indices=1)

        """
        # { "morph": 0, "morph_tag": 1, "tag" : 2, "character": 3, .. }
        for item in self.parameter["embedding"]:
            self._embedding_matrix.append(self._build_embedding(item[1], item[2], name="embedding_" + item[0]))

        self._embeddings.append(tf.nn.embedding_lookup(self._embedding_matrix[0], self.morph))
        self._embeddings.append(tf.nn.embedding_lookup(self._embedding_matrix[1], self.character))

        self._embeddings[0] = tf.nn.dropout(self._embeddings[0], self.dropout_rate)

        all_data_emb = self.ne_dict
        for i in range(0, len(self._embeddings) - 1):
            all_data_emb = tf.concat([all_data_emb, self._embeddings[i]], axis=2)
        all_data_emb = tf.concat([all_data_emb, char_embs_rnn], axis=2)
        print("[*] Embeddings : ", all_data_emb.get_shape().as_list())
        """

        item = self.parameter["embedding"][1]
        self._embedding_matrix = self._build_embedding(item[1], item[2], name="embedding_char")
        char_embedding = tf.nn.embedding_lookup(self._embedding_matrix, self.character)

        character_embedding = tf.reshape(char_embedding, [-1, self.parameter["word_length"], item[2]])
        char_len = tf.reshape(self.character_len, [-1])
        char_embs_rnn = self._build_birnn_model(character_embedding, seq_len=char_len,
                                                lstm_units=self.parameter["char_lstm_units"],
                                                keep_prob=self.dropout_rate,
                                                last=True, scope="char_layer")

        char_embs_rnn_size = char_embs_rnn.get_shape().as_list()
        print("[*] Character Embedding RNN size : ", char_embs_rnn_size)

        all_data_emb = self.ne_dict
        all_data_emb = tf.concat([all_data_emb, bert_output, char_embs_rnn], axis=2)
        print("[*] Embeddings : ", all_data_emb.get_shape().as_list())

        dense_bi_lstm = DenselyConnectedBiRNN(num_layers=self.parameter["num_dc_layer"],
                                              num_units=self.parameter["lstm_units"],
                                              num_last_units=self.parameter["lstm_units"],
                                              keep_prob=self.dropout_rate)(all_data_emb, seq_len=self.sequence)
        print("[*] DC-bi-LSTM : ", dense_bi_lstm.get_shape().as_list())
        outputs = tf.reshape(dense_bi_lstm, (-1, 2 * self.parameter["lstm_units"]))
        print("[*] DC-bi-LSTM-reshape : ", dense_bi_lstm.get_shape().as_list())

        residual_output = tf.layers.dense(tf.reshape(all_data_emb, (-1, all_data_emb.get_shape().as_list()[-1])),
                                          units=2 * self.parameter["lstm_units"],
                                          kernel_initializer=self.he_uni,
                                          kernel_regularizer=self.l2_reg)
        outputs += residual_output
        outputs = tf.nn.dropout(outputs, self.dropout_rate)

        """
        outputs = self._build_birnn_model(bert_output,
                                          seq_len=max_seq_length,
                                          lstm_units=self.parameter["lstm_units"],
                                          keep_prob=self.dropout_rate,
                                          scope="bi-LSTM_layer")
        print("[*] outputs size : ", outputs.get_shape().as_list())
        """

        sentence_output = tf.layers.dense(outputs,
                                          units=self.parameter["n_class"],
                                          kernel_initializer=self.he_uni,
                                          kernel_regularizer=self.l2_reg)
        sentence_output = tf.nn.tanh(sentence_output)
        print("[*] sentence_output size : ", sentence_output.get_shape().as_list())

        crf_cost, crf_weight, crf_bias = self._build_crf_layer(sentence_output)

        costs = crf_cost

        self.train_op = self._build_output_layer(costs)
        self.cost = costs

        print("[+] Model loaded!")

    def _build_placeholder(self):
        self.morph = tf.placeholder(tf.int32, [None, None])
        self.ne_dict = tf.placeholder(tf.float32, [None, None, int(self.parameter["n_class"] / 2)])
        self.character = tf.placeholder(tf.int32, [None, None, None])
        self.dropout_rate = tf.placeholder(tf.float32)
        self.sequence = tf.placeholder(tf.int32, [None])
        self.character_len = tf.placeholder(tf.int32, [None, None])
        self.label = tf.placeholder(tf.int32, [None, None])
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

    @staticmethod
    def _char_embedding_init(dim):
        scale = tf.sqrt(3. / dim)
        return tf.random_uniform_initializer(-scale, scale)

    def _build_embedding(self, n_tokens, dimention, name="embedding"):
        embedding_weights = tf.get_variable(
            name,
            [n_tokens, dimention],
            initializer=self._char_embedding_init(dimention),
            dtype=tf.float32,
        )
        return embedding_weights

    def _build_weight(self, shape, scope="weight"):
        with tf.variable_scope(scope):
            W = tf.get_variable(name="W", shape=[shape[0], shape[1]], dtype=tf.float32,
                                initializer=self.he_uni,
                                regularizer=self.l2_reg)
            b = tf.get_variable(name="b", shape=[shape[1]], dtype=tf.float32, initializer=tf.zeros_initializer())
        return W, b

    def _build_single_cell(self, lstm_units, keep_prob):
        cell = tf.contrib.rnn.LayerNormBasicLSTMCell(lstm_units)
        # cell = tf.contrib.rnn.GRUCell(lstm_units)
        # cell = tf.contrib.rnn.NASCell(lstm_units)

        cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=keep_prob, output_keep_prob=keep_prob)
        # cell = tf.contrib.rnn.DropoutWrapper(cell, variational_recurrent=True, dtype=tf.float32,
        #                                      output_keep_prob=keep_prob)
        return cell

    def _build_multi_cell(self, lstm_units, keep_prob, depth):
        return [self._build_single_cell(lstm_units, keep_prob) for _ in range(depth)]

    def _build_multi_biLSTM_model(self, target, seq_len, lstm_units, keep_prob, use_residual=False,
                                  scope="biLSTM-Layer"):
        output = target
        with tf.variable_scope(scope + "1"):
            with tf.variable_scope("forward_1"):
                lstm_fw_cell1 = self._build_single_cell(lstm_units, keep_prob)

            with tf.variable_scope("backward_1"):
                lstm_bw_cell1 = self._build_single_cell(lstm_units, keep_prob)

            outputs, states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell1, lstm_bw_cell1,
                                                              inputs=output,
                                                              sequence_length=seq_len,
                                                              scope=scope + "-bi-RNN1",
                                                              dtype=tf.float32)
            output_1 = tf.concat(outputs, axis=2)
            output_1 = tf.nn.dropout(output_1, keep_prob)
            (output_fw_stat, output_bw_stat) = states

        with tf.variable_scope(scope + "2"):
            with tf.variable_scope("forward_2"):
                lstm_fw_cell2 = self._build_single_cell(lstm_units, keep_prob)

            with tf.variable_scope("backward_2"):
                lstm_bw_cell2 = self._build_single_cell(lstm_units, keep_prob)

            outputs, states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell2, lstm_bw_cell2,
                                                              inputs=output_1,
                                                              sequence_length=seq_len,
                                                              initial_state_fw=output_bw_stat,
                                                              initial_state_bw=output_fw_stat,
                                                              scope=scope + "-bi-RNN2",
                                                              dtype=tf.float32)
            output_2 = tf.concat(outputs, axis=2)

        if use_residual:
            output_2 = tf.add(output_1, output_2, name='biLSTM-residual')

        output_2 = tf.reshape(output_2, shape=[-1, 2 * lstm_units])
        return output_2

    def _build_birnn_model(self, target, seq_len, lstm_units, keep_prob, last=False, scope="layer"):
        with tf.variable_scope("forward_" + scope):
            lstm_fw_cell = self._build_single_cell(lstm_units, keep_prob)

        with tf.variable_scope("backward_" + scope):
            lstm_bw_cell = self._build_single_cell(lstm_units, keep_prob)

        with tf.variable_scope("birnn-lstm_" + scope):
            _output = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell,
                                                      dtype=tf.float32,
                                                      inputs=target, sequence_length=seq_len, scope="rnn_" + scope)
            if last:
                _, ((_, output_fw), (_, output_bw)) = _output
                outputs = tf.concat([output_fw, output_bw], axis=1)
                outputs = tf.reshape(outputs, shape=[-1, self.parameter["sentence_length"], 2 * lstm_units])
            else:
                (output_fw, output_bw), _ = _output
                outputs = tf.concat([output_fw, output_bw], axis=2)
                # outputs = tf.nn.dropout(outputs, keep_prob)
                outputs = tf.reshape(outputs, shape=[-1, 2 * self.parameter["lstm_units"]])
        return outputs

    def _build_crf_layer(self, target):
        with tf.variable_scope("crf_layer"):
            W, B = self._build_weight([self.parameter["n_class"], self.parameter["n_class"]], scope="weight_bias")
            matricized_unary_scores = tf.matmul(target, W) + B
            matricized_unary_scores = tf.reshape(matricized_unary_scores,
                                                 [-1, self.parameter["sentence_length"], self.parameter["n_class"]])

            self.matricized_unary_scores = matricized_unary_scores
            self.log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
                self.matricized_unary_scores, self.label, self.sequence)
            cost = tf.reduce_mean(-self.log_likelihood)

            self.viterbi_sequence, self.viterbi_score = tf.contrib.crf.crf_decode(matricized_unary_scores,
                                                                                  self.transition_params,
                                                                                  self.sequence)
        return cost, W, B

    def _build_output_layer(self, cost):
        with tf.variable_scope("output_layer"):
            learning_rate = tf.train.exponential_decay(self.parameter["learning_rate"],
                                                       self.global_step,
                                                       100,  # hard-coded
                                                       self.parameter['lr_decay'],
                                                       staircase=True)

            self.lr = tf.clip_by_value(learning_rate,
                                       clip_value_min=1e-4,
                                       clip_value_max=self.parameter["learning_rate"],
                                       name='lr-clipped')

            if self.parameter['opt'] == 'adam':
                opt = tf.train.AdamOptimizer(learning_rate=self.lr, epsilon=1e-6)
            else:
                opt = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=.9, use_nesterov=True)

            gradients, variables = zip(*opt.compute_gradients(cost))
            gradients, _ = tf.clip_by_global_norm(gradients, self.parameter['grad_clip'])
            train_op = opt.apply_gradients(zip(gradients, variables), global_step=self.global_step)

        return train_op

    def sparse_cross_entropy_loss(self, logits):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                              labels=self.label)
        mask = tf.sequence_mask(self.sequence)
        loss = tf.boolean_mask(loss, mask)
        loss = tf.reduce_mean(loss)
        return loss


if __name__ == "__main__":
    parameter = {
        "embedding": {
            "morph": [10, 10],
            "morph_tag": [10, 10],
            "tag": [10, 10],
            "ne_dict": [10, 10],
            "character": [10, 10],
        },
        "lstm_units": 32,
        "keep_prob": 0.65,
        "sequence_length": 300,
        "n_class": 100,
        "batch_size": 128,
        "learning_rate": 0.002,
    }
    model = Model(parameter)
    model.build_model()
