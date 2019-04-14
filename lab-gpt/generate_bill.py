"""
Most of this code was adapted from `src/interactive_conditional_samples.py`
by OpenAI @ https://github.com/openai/gpt-2.

Requires model.py, sample.py, and encoder.py in PYTHONPATH.
"""

import fire
import json
import os
import numpy as np
import tensorflow as tf
import textwrap

import model, sample, encoder  # <- from https://github.com/openai/gpt-2


TEMPLATE = """A BILL



DESC_HERE

    Be it enacted by the Senate and House of Representatives of the
United States of America in Congress assembled,

SECTION 1. """


def run_model(
    text_prompt,
    model_name,
    seed=None,
    nsamples=1,
    batch_size=1,
    length=None,
    temperature=0.8,
    top_k=40,
):
    """
    Interactively run the model
    :model_name=117M : String, which model to use
    :seed=None : Integer seed for random number generators, fix seed to reproduce
     results
    :nsamples=1 : Number of samples to return total
    :batch_size=1 : Number of batches (only affects speed/memory).  Must divide nsamples.
    :length=None : Number of tokens in generated text, if None (default), is
     determined by model hyperparameters
    :temperature=1 : Float value controlling randomness in boltzmann
     distribution. Lower temperature results in less random completions. As the
     temperature approaches zero, the model will become deterministic and
     repetitive. Higher temperature results in more random completions.
    :top_k=0 : Integer value controlling diversity. 1 means only 1 word is
     considered for each step (token), resulting in deterministic completions,
     while 40 means 40 words are considered at each step. 0 (default) is a
     special setting meaning no restrictions. 40 generally is a good value.
    """
    if batch_size is None:
        batch_size = 1
    assert nsamples % batch_size == 0

    enc = encoder.get_encoder(model_name)
    hparams = model.default_hparams()
    with open(os.path.join('models', model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx // 2
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    with tf.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.set_random_seed(seed)
        output = sample.sample_sequence(
            hparams=hparams, length=length,
            context=context,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k
        )

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join('models', model_name))
        saver.restore(sess, ckpt)

        context_tokens = enc.encode(text_prompt)
        generated = 0
        print('='*72)
        print(text_prompt, end='')
        gen_text = ''
        for _ in range(nsamples // batch_size):
            out = sess.run(output, feed_dict={
                context: [context_tokens for _ in range(batch_size)]
            })[:, len(context_tokens):]
            for i in range(batch_size):
                generated += 1
                text = enc.decode(out[i])
                gen_text += text
        try:
            idx = gen_text.index('<|endoftext|>')
        except:
            idx = -1
        if idx > 0:
            gen_text = gen_text[:idx]
        print(gen_text)
        print('='*72)


if __name__ == '__main__':
    prompt = input('Bill prompt ("To require/reward/provide..."): ')
    prompt_formatted = [line.center(72).rstrip() for line in textwrap.wrap(prompt, 72)]
    model_input = TEMPLATE.replace('DESC_HERE', '\n'.join(prompt_formatted))
    run_model(model_input, 'gov')
