from numpy import logical_not
import tensorflow as tf

import model


def top_k_logits(logits, k):
    if k == 0:
        return logits

    def _top_k():
        values, _ = tf.nn.top_k(logits, k=k)
        min_values = values[:, -1, tf.newaxis]
        return tf.where(
            logits < min_values,
            tf.ones_like(logits, dtype=logits.dtype) * -1e10,
            logits,
        )

    return tf.cond(tf.equal(k, 0), lambda: logits, lambda: _top_k())


def top_p_logits(logits, p):
    batch, _ = logits.shape.as_list()
    sorted_logits = tf.sort(logits, direction="DESCENDING", axis=-1)
    cumulative_probs = tf.cumsum(tf.nn.softmax(sorted_logits, axis=-1), axis=-1)
    indices = tf.stack(
        [
            tf.range(0, batch),
            tf.maximum(
                tf.reduce_sum(tf.cast(cumulative_probs <= p, tf.int32), axis=-1) - 1, 0
            ),
        ]
    )
    min_values = tf.gather_nd(sorted_logits, indices)
    return tf.where(logits < min_values, tf.ones_like(logits) * -1e10, logits)


def sample_sequence(
    *,
    hparams,
    length,
    start_token=None,
    batch_size=None,
    context=None,
    temperature=1,
    top_k=0,
    top_p=1
):
    if start_token is None:
        assert context is not None, "Specify exactly one of start_token and context!"
    else:
        assert context is None, "Specify exactly one of start_token and context!"
        context = tf.fill([batch_size, 1], start_token)

    def step(hparams, tokens, past=None):
        lm_output = model.model(
            hparams=hparams, X=tokens, past=past, reuse=tf.AUTO_REUSE
        )
        logits = lm_output["logits"][:, :, : hparams.n_vocab]




