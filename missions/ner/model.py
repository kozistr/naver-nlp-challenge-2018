# -*- coding: utf-8 -*-
from functools import reduce
from operator import mul

import tensorflow as tf

from bert_model import BertModel, BertConfig
from bert_opt import AdamWeightDecayOptimizer, create_optimizer


def dropout(input_tensor, dropout_prob):
    """Perform dropout.

    Args:
      input_tensor: float Tensor.
      dropout_prob: Python float. The probability of dropping out a value (NOT of
        *keeping* a dimension as in `tf.nn.dropout`).

    Returns:
      A version of `input_tensor` with dropout applied.
    """
    if dropout_prob is None or dropout_prob == 0.0:
        return input_tensor

    output = tf.nn.dropout(input_tensor, 1.0 - dropout_prob)
    return output


def layer_norm(input_tensor, name=None):
    """Run layer normalization on the last dimension of the tensor."""
    return tf.contrib.layers.layer_norm(
        inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)


def layer_norm_and_dropout(input_tensor, dropout_prob, name=None):
    """Runs layer normalization followed by dropout."""
    output_tensor = layer_norm(input_tensor, name)
    output_tensor = dropout(output_tensor, dropout_prob)
    return output_tensor


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


class AttentionCell(tf.contrib.rnn.RNNCell):  # time_major based
    """Implement of https://pdfs.semanticscholar.org/8785/efdad2abc384d38e76a84fb96d19bbe788c1.pdf?_ga=2.156364859.18139
    40814.1518068648-1853451355.1518068648
    refer: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/rnn_cell_impl.py"""
    def __init__(self, num_units, memory, pmemory, keep_prob):
        super(AttentionCell, self).__init__()
        self.num_units = num_units
        self.keep_prob = keep_prob
        self.memory = memory
        self.pmemory = pmemory
        self.mem_units = memory.get_shape().as_list()[-1]
        self._cell = self._build_single_cell()

    def _build_single_cell(self):
        cell = tf.contrib.rnn.LayerNormBasicLSTMCell(self.num_units)
        cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=self.keep_prob, output_keep_prob=self.keep_prob)
        return cell

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def compute_output_shape(self, input_shape):
        pass

    def __call__(self, inputs, state, scope=None):
        c, m = state
        # (max_time, batch_size, att_unit)
        ha = tf.nn.tanh(tf.add(self.pmemory, tf.layers.dense(m, self.mem_units, use_bias=False, name="wah")))
        alphas = tf.squeeze(tf.exp(tf.layers.dense(ha, units=1, use_bias=False, name='way')), axis=[-1])
        alphas = tf.div(alphas, tf.reduce_sum(alphas, axis=0, keepdims=True))  # (max_time, batch_size)
        # (batch_size, att_units)
        w_context = tf.reduce_sum(tf.multiply(self.memory, tf.expand_dims(alphas, axis=-1)), axis=0)
        h, new_state = self._cell(inputs, state)
        lfc = tf.layers.dense(w_context, self.num_units, use_bias=False, name='wfc')
        # (batch_size, num_units)
        fw = tf.sigmoid(tf.layers.dense(lfc, self.num_units, use_bias=False, name='wff') +
                        tf.layers.dense(h, self.num_units, name='wfh'))
        hft = tf.multiply(lfc, fw) + h  # (batch_size, num_units)
        return hft, new_state


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
        cell = tf.contrib.rnn.AttentionCellWrapper(cell, attn_length=lstm_units)
        # cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=self.keep_prob, output_keep_prob=self.keep_prob)
        # cell = tf.contrib.rnn.ResidualWrapper(cell)
        cell = tf.contrib.rnn.DropoutWrapper(cell, variational_recurrent=True, dtype=tf.float32,
                                             output_keep_prob=self.keep_prob)
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
        self.he_uni = tf.contrib.layers.variance_scaling_initializer(factor=6., mode='FAN_AVG', uniform=True)

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
        Try 8 : (bi-LSTM (Char) + BERT) + Densely-Connected-bi-LSTM x 8 + Residual + CRF : 78.8
        Try 9 : (bi-LSTM (Char) + BERT + WordEmbedding) + Densely-Connected-bi-LSTM x 8 + Residual + CRF :
        """

        self._build_placeholder()

        word_bert_config = BertConfig(vocab_size=self.parameter["embedding"][0][1],
                                      hidden_size=self.parameter["word_embedding_size"],
                                      num_hidden_layers=self.parameter["num_hidden_layers"],
                                      num_attention_heads=self.parameter["num_attention_heads"],
                                      intermediate_size=self.parameter["intermediate_size"],
                                      # hidden_dropout_prob=1. - self.dropout_rate,
                                      # attention_probs_dropout_prob=1. - self.dropout_rate,
                                      max_position_embeddings=self.parameter["sentence_length"],
                                      type_vocab_size=self.parameter["n_class"])

        word_bert_model = BertModel(config=word_bert_config,
                                    is_training=self.parameter["mode"] == "train",
                                    input_ids=self.morph,
                                    input_mask=None,  # tf.sign(tf.abs(self.morph)),  # None,
                                    token_type_ids=self.label if self.parameter["mode"] == "train" else None,  # None,
                                    use_one_hot_embeddings=False,
                                    scope="WordBertModel")

        word_bert_output = word_bert_model.get_sequence_output()
        print("[*] Word BERT output : ", word_bert_output.get_shape().as_list())

        # { "morph": 0, "morph_tag": 1, "tag" : 2, "character": 3, .. }
        # for item in self.parameter["embedding"]:
        #     self._embedding_matrix.append(self._build_embedding(item[1], item[2], name="embedding_" + item[0]))

        # self._embeddings.append(tf.nn.embedding_lookup(self._embedding_matrix[0], self.morph))
        # self._embeddings.append(tf.nn.embedding_lookup(self._embedding_matrix[1], self.character))

        # self._embeddings[0] = tf.nn.dropout(self._embeddings[0], self.dropout_rate)

        char_embeddings = self._build_embedding(self.parameter["embedding"][1][1],
                                                self.parameter["embedding"][1][2],
                                                name="embedding_" + self.parameter["embedding"][1][0])
        char_embeddings = tf.nn.embedding_lookup(char_embeddings, self.character)
        character_embedding = tf.reshape(char_embeddings, [-1, self.parameter["word_length"],
                                                           self.parameter["char_embedding_size"]])
        char_len = tf.reshape(self.character_len, [-1])
        char_embs_rnn = self._build_birnn_model(character_embedding,
                                                seq_len=char_len,
                                                lstm_units=self.parameter["char_lstm_units"],
                                                keep_prob=self.dropout_rate,
                                                last=True, scope="char_layer")
        char_embs_rnn_size = char_embs_rnn.get_shape().as_list()
        print("[*] Character Embedding RNN size : ", char_embs_rnn_size)

        """
        with tf.name_scope("CharCNN"):
            ks = [2, 3, 4, 5]
            fs = 16

            pooled_out = []
            for idx, k in enumerate(ks):
                x = tf.layers.conv1d(character_embedding, filters=fs, kernel_size=k, dilation_rate=2 ** idx,
                                     kernel_initializer=self.he_uni, kernel_regularizer=self.l2_reg,
                                     padding='same',
                                     name="dilatedCNN-%d" % idx)
                x = tf.nn.relu(x)
                x = tf.reduce_max(x, axis=1)
                pooled_out.append(x)

            char_embs_cnn = tf.concat(pooled_out, axis=1)
            char_embs_cnn = tf.reshape(char_embs_cnn, (-1, self.parameter["sentence_length"], fs * len(ks)))
            char_embs_cnn_size = char_embs_cnn.get_shape().as_list()
            print("[*] Character Embedding CNN size : ", char_embs_cnn_size)
        """

        all_data_emb = tf.concat([self.ne_dict, word_bert_output, char_embs_rnn], axis=2)
        print("[*] Embeddings : ", all_data_emb.get_shape().as_list())

        dense_bi_lstm = DenselyConnectedBiRNN(num_layers=self.parameter["num_dc_layer"],
                                              num_units=self.parameter["lstm_units"],
                                              num_last_units=self.parameter["lstm_units"],
                                              keep_prob=self.dropout_rate)(all_data_emb, seq_len=self.sequence)
        print("[*] DC-bi-LSTM : ", dense_bi_lstm.get_shape().as_list())

        dense_bi_lstm = tf.reshape(dense_bi_lstm, (-1, dense_bi_lstm.get_shape().as_list()[-1]))
        print("[*] DC-bi-LSTM-reshape : ", dense_bi_lstm.get_shape().as_list())

        residual = tf.layers.dense(tf.reshape(all_data_emb, (-1, all_data_emb.get_shape().as_list()[-1])),
                                   units=2 * self.parameter["lstm_units"],
                                   kernel_initializer=self.he_uni,
                                   kernel_regularizer=self.l2_reg)

        dense_bi_lstm += residual  # tf.concat([context, p_context], axis=-1)
        outputs = tf.nn.dropout(dense_bi_lstm, self.dropout_rate)
        outputs = tf.reshape(outputs, (-1, dense_bi_lstm.get_shape().as_list()[-1]))
        # outputs = tf.reshape(outputs, (-1, self.parameter["sentence_length"], outputs.get_shape().as_list()[-1]))
        print("[*] outputs size : ", outputs.get_shape().as_list())

        """
        outputs = self._build_birnn_model(outputs,
                                          self.sequence,
                                          self.parameter["lstm_units"],
                                          self.dropout_rate,
                                          scope="bi-LSTM")
        print("[*] outputs size : ", outputs.get_shape().as_list())
        """

        """
        # outputs = tf.nn.dropout(outputs, self.dropout_rate)
        # outputs = layer_norm_and_dropout(outputs, self.dropout_rate)


        outputs = self._build_birnn_model(all_data_emb,
                                          self.sequence,
                                          self.parameter["lstm_units"],
                                          self.dropout_rate,
                                          scope="bi-LSTM")
        print("[*] outputs size : ", outputs.get_shape().as_list())
        """

        """
        with tf.variable_scope("stacked-bi-LSTM"):
            fw_cell = self._build_multi_cell(self.parameter["lstm_units"], self.dropout_rate,
                                             self.parameter["num_lstm_depth"])

            bw_cell = self._build_multi_cell(self.parameter["lstm_units"], self.dropout_rate,
                                             self.parameter["num_lstm_depth"])

            outputs = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell,
                                                      dtype=tf.float32,
                                                      inputs=all_data_emb,
                                                      sequence_length=self.sequence,
                                                      scope="birnn")
            (output_fw, output_bw), _ = outputs
            outputs = tf.concat([output_fw, output_bw], axis=2)
            outputs = tf.reshape(outputs, shape=[-1, outputs.get_shape().as_list()[-1]])
            print("[*] outputs size : ", outputs.get_shape().as_list())
        """

        sentence_output = tf.layers.dense(outputs,
                                          units=self.parameter["n_class"],
                                          kernel_initializer=self.he_uni,
                                          kernel_regularizer=self.l2_reg)
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
        self.global_step = tf.train.get_or_create_global_step()  # tf.Variable(0, trainable=False, name='global_step')

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
        return tf.nn.rnn_cell.MultiRNNCell([self._build_single_cell(lstm_units, keep_prob) for _ in range(depth)],
                                           state_is_tuple=True)

    def _build_multi_biLSTM_model(self, target, seq_len, lstm_units, keep_prob, use_residual=False,
                                  scope="biLSTM-Layer"):
        output = target
        with tf.variable_scope(scope + "1"):
            with tf.variable_scope("forward_1"):
                lstm_fw_cell = self._build_single_cell(lstm_units, keep_prob)

            with tf.variable_scope("backward_1"):
                lstm_bw_cell = self._build_single_cell(lstm_units, keep_prob)

            outputs, states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell,
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
                outputs = tf.reshape(outputs, shape=[-1, outputs.get_shape().as_list()[-1]])
        return outputs

    def _build_crf_layer(self, target):
        with tf.variable_scope("crf_layer"):
            W, B = self._build_weight([self.parameter["n_class"], self.parameter["n_class"]], scope="weight_bias")
            matricized_unary_scores = tf.matmul(target, W) + B
            matricized_unary_scores = tf.reshape(matricized_unary_scores,
                                                 [-1, self.parameter["sentence_length"], self.parameter["n_class"]])

            self.matricized_unary_scores = matricized_unary_scores
            self.log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
                self.matricized_unary_scores,
                self.label,
                self.sequence)
            cost = tf.reduce_mean(-self.log_likelihood)

            self.viterbi_sequence, self.viterbi_score = tf.contrib.crf.crf_decode(matricized_unary_scores,
                                                                                  self.transition_params,
                                                                                  self.sequence)
        return cost, W, B

    def _build_output_layer(self, cost):
        with tf.variable_scope("output_layer"):

            learning_rate = tf.train.exponential_decay(self.parameter["learning_rate"],
                                                       self.global_step,
                                                       75,  # 900 / 12 # 100 # hard-coded
                                                       self.parameter['lr_decay'],
                                                       staircase=True)

            self.lr = tf.clip_by_value(learning_rate,
                                       clip_value_min=2e-5,
                                       clip_value_max=self.parameter["learning_rate"],
                                       name='lr-clipped')

            if self.parameter['opt'] == 'adam':
                opt = tf.train.AdamOptimizer(learning_rate=self.lr, epsilon=1e-6)
            else:
                opt = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=.9, use_nesterov=True)

            """
            optimizer = AdamWeightDecayOptimizer(learning_rate=self.parameter["learning_rate"])
            tvars = tf.trainable_variables()
            grads = tf.gradients(cost, tvars)
            # grads, _ = tf.clip_by_global_norm(grads, self.parameter['grad_clip'])
            train_op = optimizer.apply_gradients(zip(grads, tvars), self.global_step)
            """
            """
            train_op = create_optimizer(loss=cost,
                                        init_lr=self.parameter["learning_rate"],
                                        num_train_steps=100,
                                        num_warmup_steps=None,
                                        use_tpu=False,
                                        clip_norm=self.parameter["grad_clip"])
            """
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
