# beamspy

## Motivation

Imagine the following scenario:

1. For some reason, you need to adjust logits during text (or other sequence) generation.
2. Further, the function you want depends on the previous generated text/sequence.

The transformers package offers you the option to customize generation with a list of "LogitsProcessors", which encapsulate functions which have the previous tokens in a beam and the current logits as arguments. However, if you want to consider the prefix every time, this can become inefficient: if your function has complexity K on a sequence of n tokens, then this approach has a complexity of approx. K*n^2. For very long texts, this will hurt!

So you will probably make your function such that it is "incremental": it can reuse previous work, so that updating it when adding a new token to a sequence is faster than rerunning on the whole previous sequence.

But, the LogitsProcessor API gives you a beam's tokens, it does not give you a beam's "parent" -- i.e., that this beam is obtained by adding a particular token to a particular beam you have seen on the previous step. There probably is a way to get this in the transformers library already, but I'm a noob and didn't see it -- so I made one!

This is that package.

## What does it do

Track beam ancestry. Every time this LogitsProcessor is called, it checks which beam from the previous step k-1 is the parent of which beam in the current step k. A beam at k-1 is a parent for a beam at k if the beam at k can be written as the beam at k-1 plus some new token.

This LogitsProcessor does not do anything else. If you want to implement your custom, fancy, prefix dependent function you can however use this as a superclass.

## How does it do it

Naively, to check which beam is the parent of which beam needs checking the corresponding sequences of tokens. But this is exactly the kind of O(n^2) operation we would rather avoid. Instead, it is enough to check a subset of positions -- those at which a beam diverges and becomes the parent of several beams. This list of positions to check must be updated throughout the generation: new positions to check must be added as beams diverge, positions that no longer discriminate between currently active beams can be discarded. The end result is that we only keep a very small set of positions to check. Worst case, this is log(m), if m is the number of beams.

## How to use it

Write something in the _doWork function to encode whatever it is you need doing to the logits.

Currently, this does nothing. It can print them though if you select the print_beams flag, together with the beam recipes. Much nice, very zen, to see the texts as they evolve (but don't do this except for debugging, reprinting sequences as they are generated token by token is a slow operation ...)

