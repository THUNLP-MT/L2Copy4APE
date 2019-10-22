# coding=utf-8
# Copyright 2017-2019 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import tensorflow as tf
import thumt.interface as interface
import thumt.layers as layers
import thumt.utils.getter as getter
from thumt.layers.nn import linear


def _layer_process(x, mode):
    if not mode or mode == "none":
        return x
    elif mode == "layer_norm":
        return layers.nn.layer_norm(x)
    else:
        raise ValueError("Unknown mode %s" % mode)


def _residual_fn(x, y, keep_prob=None):
    if keep_prob and keep_prob < 1.0:
        y = tf.nn.dropout(y, keep_prob)
    return x + y


def _ffn_layer(inputs, hidden_size, output_size, keep_prob=None,
               dtype=None, scope=None):
    with tf.variable_scope(scope, default_name="ffn_layer", values=[inputs],
                           dtype=dtype):
        with tf.variable_scope("input_layer"):
            hidden = layers.nn.linear(inputs, hidden_size, True, True)
            hidden = tf.nn.relu(hidden)

        if keep_prob and keep_prob < 1.0:
            hidden = tf.nn.dropout(hidden, keep_prob)

        with tf.variable_scope("output_layer"):
            output = layers.nn.linear(hidden, output_size, True, True)

        return output


def transformer_encoder(inputs, bias, params, dtype=None, scope=None, num_layers=None, scaled_bias=None):
    with tf.variable_scope(scope, default_name="encoder", dtype=dtype,
                           values=[inputs, bias]):
        x = inputs
        for layer in range(num_layers or params.num_encoder_layers):
            with tf.variable_scope("layer_%d" % layer):
                with tf.variable_scope("self_attention"):
                    y = layers.attention.multihead_attention(
                        _layer_process(x, params.layer_preprocess),
                        None,
                        bias,
                        params.num_heads,
                        params.attention_key_channels or params.hidden_size,
                        params.attention_value_channels or params.hidden_size,
                        params.hidden_size,
                        1.0 - params.attention_dropout,
                        scaled_bias=scaled_bias
                    )
                    y = y["outputs"]
                    x = _residual_fn(x, y, 1.0 - params.residual_dropout)
                    x = _layer_process(x, params.layer_postprocess)

                with tf.variable_scope("feed_forward"):
                    y = _ffn_layer(
                        _layer_process(x, params.layer_preprocess),
                        params.filter_size,
                        params.hidden_size,
                        1.0 - params.relu_dropout,
                    )
                    x = _residual_fn(x, y, 1.0 - params.residual_dropout)
                    x = _layer_process(x, params.layer_postprocess)

        outputs = _layer_process(x, params.layer_preprocess)

        return outputs


def transformer_decoder(inputs, memory, bias, mem_bias, mt_bias, params, state=None,
                        dtype=None, scope=None, scaled_bias=None):
    with tf.variable_scope(scope, default_name="decoder", dtype=dtype,
                           values=[inputs, memory, bias, mem_bias, mt_bias]):
        x = inputs
        next_state = {}
        for layer in range(params.num_decoder_layers):
            layer_name = "layer_%d" % layer
            with tf.variable_scope(layer_name):
                layer_state = state[layer_name] if state is not None else None

                with tf.variable_scope("self_attention"):
                    y = layers.attention.multihead_attention(
                        _layer_process(x, params.layer_preprocess),
                        None,
                        bias,
                        params.num_heads,
                        params.attention_key_channels or params.hidden_size,
                        params.attention_value_channels or params.hidden_size,
                        params.hidden_size,
                        1.0 - params.attention_dropout,
                        state=layer_state
                    )

                    if layer_state is not None:
                        next_state[layer_name] = y["state"]

                    y = y["outputs"]
                    x = _residual_fn(x, y, 1.0 - params.residual_dropout)
                    x = _layer_process(x, params.layer_postprocess)

                with tf.variable_scope("encdec_attention"):
                    y = layers.attention.multihead_attention(
                        _layer_process(x, params.layer_preprocess),
                        memory,
                        mem_bias,
                        params.num_heads,
                        params.attention_key_channels or params.hidden_size,
                        params.attention_value_channels or params.hidden_size,
                        params.hidden_size,
                        1.0 - params.attention_dropout,
                        scaled_bias=scaled_bias,
                    )
                    y = y["outputs"]
                    x = _residual_fn(x, y, 1.0 - params.residual_dropout)
                    x = _layer_process(x, params.layer_postprocess)

                with tf.variable_scope("feed_forward"):
                    y = _ffn_layer(
                        _layer_process(x, params.layer_preprocess),
                        params.filter_size,
                        params.hidden_size,
                        1.0 - params.relu_dropout,
                    )
                    x = _residual_fn(x, y, 1.0 - params.residual_dropout)
                    x = _layer_process(x, params.layer_postprocess)

        with tf.variable_scope("copy_net"):
            z = x
            with tf.variable_scope("encdec_attention"):
                y = layers.attention.multihead_attention(
                    _layer_process(z, params.layer_preprocess),
                    memory,
                    mt_bias,
                    1, #params.num_heads,
                    params.attention_key_channels or params.hidden_size,
                    params.attention_value_channels or params.hidden_size,
                    params.hidden_size,
                    1.0 - params.attention_dropout,
                    scaled_bias=scaled_bias,
                )
                att = y["weights"]  # [bs, 1, lq, lk]
                y = y["outputs"]
                z = _residual_fn(z, y, 1.0 - params.residual_dropout)
                z = _layer_process(z, params.layer_postprocess)

            with tf.variable_scope("feed_forward"):
                y = _ffn_layer(
                    _layer_process(z, params.layer_preprocess),
                    params.filter_size,
                    params.hidden_size,
                    1.0 - params.relu_dropout,
                )
                z = _residual_fn(z, y, 1.0 - params.residual_dropout)
                z = _layer_process(z, params.layer_postprocess)

        outputs = _layer_process(x, params.layer_preprocess)

        z = _layer_process(z, params.layer_preprocess)
        z = linear(z, 1, True, True, scope="copy_ratio_w")
        z = tf.sigmoid(z)  # [bs, lq, 1]

        att = tf.squeeze(att, axis=1) # [bs, lq, lk]

        if state is not None:
            return outputs, next_state, att, z

        return outputs, att, z


def encoding_graph(features, mode, mask_score, params):

    if mode != "train":
        params.residual_dropout = 0.0
        params.attention_dropout = 0.0
        params.relu_dropout = 0.0
        params.label_smoothing = 0.0

    dtype = tf.get_variable_scope().dtype
    hidden_size = params.hidden_size

    src_seq = features["source"]
    src_len = features["source_length"]
    src_mask = tf.sequence_mask(src_len,
                                maxlen=tf.shape(features["source"])[1],
                                dtype=dtype or tf.float32)
    ori_mask = tf.sequence_mask(features["origin_length"],
                                maxlen=tf.shape(features["source"])[1],
                                dtype=dtype or tf.float32)
    mt_mask = src_mask - ori_mask
    lang_tok_seq = tf.to_int32(src_mask - ori_mask) # (0,0,0,1,1,1,p,p,p)

    svocab = params.vocabulary["source"]
    src_vocab_size = len(svocab)
    initializer = tf.random_normal_initializer(0.0, params.hidden_size ** -0.5)

    if params.enable_tagger is True:
        reuse_flag = True
    else:
        reuse_flag = False

    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_flag):
        src_embedding = tf.get_variable("weights",
                                        [src_vocab_size, hidden_size],
                                        initializer=initializer)

        bias = tf.get_variable("bias", [hidden_size])

        lang_embedding = tf.get_variable("lang_embedding",
                                        [2, hidden_size],
                                        initializer=initializer)
        pos_embedding = tf.get_variable("pos_embedding",
                                        [256, hidden_size],
                                        initializer=initializer)

    inputs = tf.gather(src_embedding, src_seq) \
           + tf.gather(pos_embedding, features["source_pos"]) \
           + tf.gather(lang_embedding, lang_tok_seq)

    if params.multiply_embedding_mode == "sqrt_depth":
        inputs = inputs * (hidden_size ** 0.5)

    inputs = inputs * tf.expand_dims(src_mask, -1)

    encoder_input = tf.nn.bias_add(inputs, bias)
    # encoder_input = layers.attention.add_timing_signal(encoder_input)
    enc_attn_bias = layers.attention.attention_bias(src_mask, "masking",
                                                    dtype=dtype)

    if params.residual_dropout:
        keep_prob = 1.0 - params.residual_dropout
        encoder_input = tf.nn.dropout(encoder_input, keep_prob)

    if params.enable_tagger is True:
        scaled_bias = mask_score * mt_mask + ori_mask
        if params.where_to_apply != 'dec':
            encoder_output = transformer_encoder(encoder_input, enc_attn_bias, params, scaled_bias=scaled_bias)
        else:
            encoder_output = transformer_encoder(encoder_input, enc_attn_bias, params, scaled_bias=None)
        return encoder_output, scaled_bias
    else:
        encoder_output = transformer_encoder(encoder_input, enc_attn_bias, params, scaled_bias=None)
        return encoder_output


def tagging_graph(features, mode, params):

    if mode != "train":
        params.residual_dropout = 0.0
        params.attention_dropout = 0.0
        params.relu_dropout = 0.0
        params.label_smoothing = 0.0

    dtype = tf.get_variable_scope().dtype
    hidden_size = params.hidden_size
    
    src_seq = features["source"]
    src_len = features["source_length"]
    src_mask = tf.sequence_mask(src_len,
                                maxlen=tf.shape(features["source"])[1],
                                dtype=dtype or tf.float32)
    ori_mask = tf.sequence_mask(features["origin_length"],
                                maxlen=tf.shape(features["source"])[1],
                                dtype=dtype or tf.float32)
    mt_mask = src_mask - ori_mask
    lang_tok_seq = tf.to_int32(src_mask - ori_mask) # (0,0,0,1,1,1,p,p,p)

    svocab = params.vocabulary["source"]
    src_vocab_size = len(svocab)
    initializer = tf.random_normal_initializer(0.0, params.hidden_size ** -0.5)

    src_embedding = tf.get_variable("weights",
                                    [src_vocab_size, hidden_size],
                                    initializer=initializer)

    bias = tf.get_variable("bias", [hidden_size])

    lang_embedding = tf.get_variable("lang_embedding",
                                    [2, hidden_size],
                                    initializer=initializer)
    pos_embedding = tf.get_variable("pos_embedding",
                                    [256, hidden_size],
                                    initializer=initializer)

    inputs = tf.gather(src_embedding, src_seq) \
           + tf.gather(pos_embedding, features["source_pos"]) \
           + tf.gather(lang_embedding, lang_tok_seq)

    if params.multiply_embedding_mode == "sqrt_depth":
        inputs = inputs * (hidden_size ** 0.5)

    inputs = inputs * tf.expand_dims(src_mask, -1)

    encoder_input = tf.nn.bias_add(inputs, bias)
    # encoder_input = layers.attention.add_timing_signal(encoder_input)
    enc_attn_bias = layers.attention.attention_bias(src_mask, "masking",
                                                    dtype=dtype)

    if params.residual_dropout:
        keep_prob = 1.0 - params.residual_dropout
        encoder_input = tf.nn.dropout(encoder_input, keep_prob)

    with tf.variable_scope('tagging'):

        encoder_output = transformer_encoder(encoder_input, enc_attn_bias, params, num_layers=params.num_tagger_layers)
        r_encoder_output = tf.reshape(encoder_output, [-1, params.hidden_size])

        if params.pred_mask_loss == 'sigmoid_ce':
            tag_weights = tf.get_variable('tag_weights', [params.hidden_size, 1])
            logits = tf.matmul(r_encoder_output, tag_weights)
            logits = tf.reshape(logits, tf.shape(mt_mask))
            mask_score = tf.sigmoid(logits)
        elif params.pred_mask_loss == 'softmax_ce':
            tag_weights = tf.get_variable('tag_weights', [params.hidden_size, 2])
            logits = tf.matmul(r_encoder_output, tag_weights)
            prob = tf.nn.softmax(logits, axis=-1)
            mask_score = prob[:,1]
            mask_score = tf.reshape(mask_score, tf.shape(mt_mask))

        if mode == 'train':
            tag_seq = features["tagging"]
            tag_seq = tf.to_float(tag_seq)
            if params.pred_mask_loss == 'sigmoid_ce':
                ce = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tag_seq)
            elif params.pred_mask_loss == 'softmax_ce':
                ce = layers.nn.smoothed_softmax_cross_entropy_with_logits(
                    logits=logits,
                    labels=tf.to_int32(tag_seq),
                    smoothing=params.label_smoothing,
                    normalize=True
                )
                ce = tf.reshape(ce, tf.shape(mt_mask))
            loss = tf.reduce_sum(ce * mt_mask) / tf.reduce_sum(mt_mask)
            return mask_score, loss
        elif mode == 'eval':
            tag_seq = features["tagging"]

            correct = tf.to_float(tf.equal(
                tf.to_int32(tf.rint(mask_score)),
                tag_seq
            ))

            return tf.reduce_sum(correct * mt_mask), tf.reduce_sum(mt_mask)
        elif mode == 'infer':
            return mask_score
        else:
            raise ValueError("Unknown mode {}".format(mode))


def decoding_graph(features, state, mode, params):
    if mode != "train":
        params.residual_dropout = 0.0
        params.attention_dropout = 0.0
        params.relu_dropout = 0.0
        params.label_smoothing = 0.0

    dtype = tf.get_variable_scope().dtype
    src_seq = features["source"]
    tgt_seq = features["target"]
    src_len = features["source_length"]
    tgt_len = features["target_length"]
    src_mask = tf.sequence_mask(src_len,
                                maxlen=tf.shape(features["source"])[1],
                                dtype=dtype or tf.float32)
    tgt_mask = tf.sequence_mask(tgt_len,
                                maxlen=tf.shape(features["target"])[1],
                                dtype=dtype or tf.float32)
    ori_mask = tf.sequence_mask(features["origin_length"],
                                maxlen=tf.shape(features["source"])[1],
                                dtype=dtype or tf.float32)
    mt_mask = src_mask - ori_mask
    
    hidden_size = params.hidden_size
    tvocab = params.vocabulary["target"]
    tgt_vocab_size = len(tvocab)

    svocab = params.vocabulary["source"]
    src_vocab_size = len(svocab)
    src_seq_one_hot = tf.one_hot(src_seq, src_vocab_size) # [bs, lk, vocab]

    initializer = tf.random_normal_initializer(0.0, params.hidden_size ** -0.5)

    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
        tgt_embedding = tf.get_variable("weights",
                                        [tgt_vocab_size, hidden_size],
                                        initializer=initializer)
        lang_embedding = tf.get_variable("lang_embedding",
                                        [2, hidden_size],
                                        initializer=initializer)
        pos_embedding = tf.get_variable("pos_embedding",
                                        [256, hidden_size],
                                        initializer=initializer)

    if params.shared_embedding_and_softmax_weights:
        weights = tgt_embedding
    else:
        weights = tf.get_variable("softmax", [tgt_vocab_size, hidden_size],
                                  initializer=initializer)

    targets = tf.gather(tgt_embedding, tgt_seq)

    if params.multiply_embedding_mode == "sqrt_depth":
        targets = targets * (hidden_size ** 0.5)

    targets = targets * tf.expand_dims(tgt_mask, -1)

    enc_attn_bias = layers.attention.attention_bias(src_mask, "masking",
                                                    dtype=dtype)
    enc_attn_bias_mt = layers.attention.attention_bias(mt_mask, "masking",
                                                    dtype=dtype)
    dec_attn_bias = layers.attention.attention_bias(tf.shape(targets)[1],
                                                    "causal", dtype=dtype)
    # Shift left
    decoder_input = tf.pad(targets, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
    decoder_input = layers.attention.add_timing_signal(decoder_input)

    if params.residual_dropout:
        keep_prob = 1.0 - params.residual_dropout
        decoder_input = tf.nn.dropout(decoder_input, keep_prob)

    encoder_output = state["encoder"]
    
    if params.enable_tagger is True and params.where_to_apply != 'enc':
        scaled_bias = state["scaled_bias"]
    else:
        scaled_bias = None

    if mode != "infer":
        decoder_output, copy_prob, copy_alpha = transformer_decoder(decoder_input, encoder_output,
                                             dec_attn_bias, enc_attn_bias, enc_attn_bias_mt,
                                             params, scaled_bias=scaled_bias)
    else:
        decoder_input = decoder_input[:, -1:, :]
        dec_attn_bias = dec_attn_bias[:, :, -1:, :]
        decoder_outputs = transformer_decoder(decoder_input, encoder_output,
                                              dec_attn_bias, enc_attn_bias, enc_attn_bias_mt,
                                              params, state=state["decoder"], scaled_bias=scaled_bias)

        decoder_output, decoder_state, copy_prob, copy_alpha = decoder_outputs 
        decoder_output = decoder_output[:, -1, :]
        logits = tf.matmul(decoder_output, weights, False, True)
        prob_gen = tf.nn.softmax(logits) # [bs, vocab]

        prob_copy = tf.matmul(copy_prob, src_seq_one_hot)

        prob_copy = tf.squeeze(prob_copy, axis=1)
        copy_alpha = tf.squeeze(copy_alpha, axis=1)

        prob = prob_gen * (1 - copy_alpha) + prob_copy * copy_alpha
        log_prob = tf.log(prob)

        if params.enable_tagger is True:
            return log_prob, {"encoder": encoder_output, "scaled_bias": state["scaled_bias"], "decoder": decoder_state}
        else:
            return log_prob, {"encoder": encoder_output, "decoder": decoder_state}

    decoder_output = tf.reshape(decoder_output, [-1, hidden_size])
    logits = tf.matmul(decoder_output, weights, False, True)
    prob_gen = tf.nn.softmax(logits) # [bs, lq, vocab]

    prob_copy = tf.matmul(copy_prob, src_seq_one_hot) # [bs, lq, vocab]
    prob_copy = tf.reshape(prob_copy, [-1, tgt_vocab_size])
    _copy_alpha = tf.reshape(copy_alpha, [-1, 1])

    prob = prob_gen * (1 - _copy_alpha) + prob_copy * _copy_alpha
    log_prob = tf.log(prob)

    labels = features["target"]

    # label smoothing
    ce = layers.nn.smoothed_softmax_cross_entropy_with_logits(
        logits=log_prob,
        labels=labels,
        smoothing=params.label_smoothing,
        normalize=True
    )
    tgt_mask = tf.cast(tgt_mask, ce.dtype)

    ce = tf.reshape(ce, tf.shape(tgt_seq))

    L_ape = tf.reduce_sum(ce * tgt_mask) / tf.reduce_sum(tgt_mask)

    if params.enable_tagger is False:
        return L_ape

    # L_copy
    copy_prob *= copy_alpha
    copy_prob *= tf.expand_dims(tgt_mask, axis=-1)
    copy_cover = tf.reduce_sum(copy_prob, axis=1) # [bs, lk]
    tag_seq = features["tagging"] # [bs, lk]
    label = tf.to_float(tag_seq)
    square_error = (label - copy_cover) * (label - copy_cover)
    L_copy = tf.reduce_sum(square_error * mt_mask) / tf.reduce_sum(mt_mask)

    loss = L_ape + params.copy_lambda * L_copy

    return loss


def model_graph(features, mode, params):
    if params.enable_tagger is False:
        encoder_output = encoding_graph(features, mode, None, params)
        state = {
            "encoder": encoder_output
        }
        output = decoding_graph(features, state, mode, params)
        return output
    elif params.enable_tagger is True:
        if mode == 'train':
            mask_score, loss1 = tagging_graph(features, 'train', params)
            encoder_output, scaled_bias = encoding_graph(features, mode, mask_score, params)
            state = {
                "encoder": encoder_output,
                "scaled_bias": scaled_bias
            }
            loss2 = decoding_graph(features, state, mode, params)
            loss = params.multi_task_alpha*loss1+(1-params.multi_task_alpha)*loss2
            return loss
        elif mode == 'eval':
            valid_res = tagging_graph(features, 'eval', params)
            return valid_res
        else:
            raise ValueError("Unknown mode {}".format(mode))


class Transformer(interface.NMTModel):

    def __init__(self, params, scope="transformer"):
        super(Transformer, self).__init__(params=params, scope=scope)

    def get_training_func(self, initializer, regularizer=None, dtype=None):
        def training_fn(features, params=None, reuse=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)

            if dtype != tf.float32:
                custom_getter = getter.fp32_variable_getter
            else:
                custom_getter = None

            with tf.variable_scope(self._scope, initializer=initializer,
                                   regularizer=regularizer, reuse=reuse,
                                   custom_getter=custom_getter, dtype=dtype):
                loss = model_graph(features, "train", params)
                return loss

        return training_fn

    def get_evaluation_func(self):
        def evaluation_fn(features, params=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)

            with tf.variable_scope(self._scope):
                score = model_graph(features, "eval", params)

            return score

        return evaluation_fn

    def get_inference_func(self):
        def encoding_fn(features, params=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)

            with tf.variable_scope(self._scope):
                if params.enable_tagger is True:
                    mask_score = tagging_graph(features, "infer", params)
                    encoder_output, scaled_bias = encoding_graph(features, "infer", mask_score, params)
                    batch = tf.shape(encoder_output)[0]
                    state = {
                        "encoder": encoder_output,
                        "scaled_bias": scaled_bias,
                        "decoder": {
                            "layer_%d" % i: {
                                "key": tf.zeros([batch, 0, params.hidden_size]),
                                "value": tf.zeros([batch, 0, params.hidden_size])
                            }
                            for i in range(params.num_decoder_layers)
                        }
                    }
                else:
                    encoder_output = encoding_graph(features, "infer", None, params)
                    batch = tf.shape(encoder_output)[0]
                    state = {
                        "encoder": encoder_output,
                        "decoder": {
                            "layer_%d" % i: {
                                "key": tf.zeros([batch, 0, params.hidden_size]),
                                "value": tf.zeros([batch, 0, params.hidden_size])
                            }
                            for i in range(params.num_decoder_layers)
                        }
                    }
            return state

        def decoding_fn(features, state, params=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)

            with tf.variable_scope(self._scope):
                log_prob, new_state = decoding_graph(features, state, "infer",
                                                     params)

            return log_prob, new_state

        return encoding_fn, decoding_fn

    @staticmethod
    def get_name():
        return "transformer"

    @staticmethod
    def get_parameters():
        params = tf.contrib.training.HParams(
            pad="<pad>",
            bos="<eos>",
            eos="<eos>",
            unk="<unk>",
            append_eos=False,
            hidden_size=512,
            filter_size=2048,
            num_heads=8,
            num_encoder_layers=6,
            num_tagger_layers=3,
            num_decoder_layers=6,
            attention_dropout=0.0,
            residual_dropout=0.1,
            relu_dropout=0.0,
            label_smoothing=0.1,
            attention_key_channels=0,
            attention_value_channels=0,
            layer_preprocess="none",
            layer_postprocess="layer_norm",
            multiply_embedding_mode="sqrt_depth",
            shared_embedding_and_softmax_weights=False,
            shared_source_target_embedding=False,
            # Override default parameters
            learning_rate_decay="linear_warmup_rsqrt_decay",
            initializer="uniform_unit_scaling",
            initializer_gain=1.0,
            learning_rate=1.0,
            batch_size=4096,
            constant_batch_size=False,
            adam_beta1=0.9,
            adam_beta2=0.98,
            adam_epsilon=1e-9,
            clip_grad_norm=0.0,
            #
            enable_tagger=False, # True, False
            pred_mask_loss='sigmoid_ce', # 'softmax_ce'
            where_to_apply='both', # 'enc', 'dec', 'both'
            multi_task_alpha=0.1,
            copy_lambda=0.1,
        )

        return params
