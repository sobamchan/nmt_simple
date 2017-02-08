import numpy as np
from chainer import Variable

demb = 100
def mk_ct(gh, ht, emb_dimention):
    s = 0.0
    for i in range(len(gh)):
        s += np.exp(ht.dot(gh[i]))
    ct = np.zeros(demb)
    for i in range(len(gh)):
        alpi = np.exp(ht.dot(gh[i])) / s
        ct += alpi * gh[i]
    ct = Variable(np.array([ct]).astype(np.float32))

    return ct
