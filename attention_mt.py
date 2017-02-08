import numpy as np
import chainer
from chainer import Variable
import chainer.functions as F
import chainer.links as L

from utils import mk_ct

import sobamchan_chainer

class AttentionMT(sobamchan_chainer.Model):

    def __init__(self, i_vocab_num, t_vocab_num, k):
        '''
        i_vocab_num: input language vocabulary number
        t_vocab_num: expected language vocabulary number
        k: embedding dimention size, embedding output
        '''
        super(AttentionMT, self).__init__(
            embed_x = L.EmbedID(i_vocab_num, k),
            embed_y = L.EmbedID(t_vocab_num, k),
            H = L.LSTM(k, k),
            Wc1 = L.Linear(k, k),
            Wc2 = L.Linear(k, k),
            W = L.Linear(k, t_vocab_num),
        )

    def __call__(self, i_line, t_line):
        gh = []
        self.H.reset_state()
        for i in range(len(i_line)):
            word_id = i_vocab[i_line[i]]
            x_k = self.embed_x(self.prepare_input([word_id], dtype=np.int32))
            h = self.H(x_k)
            gh.append(np.copy(h.data[0])) #TODO this doesn't allow batch

        x_k = self.embed_x(self.prepare_input([i_vocab['<eos>']], dtype=np.int32))
        tx = self.prepare_input([t_vocab[t_line[0]]], dtype=np.int32)
        h = self.H(x_k)
        ct = mk_ct(gh, h.data[0], h.data.shape[1])
        h2 = F.tanh(self.Wc1(ct) + self.Wc2(h))
        loss_sum = F.softmax_cross_entropy(self.W(h2), tx)

        for i in range(len(t_line)):
            word_id = t_vocab[t_line[i]]
            x_k = self.embed_y(self.prepare_input([word_id], dtype=np.int32))
            next_word_id = t_vocab['<eos>'] if (len(t_line) == i + 1) else t_vocab[t_line[i+1]]
            tx = self.prepare_input([next_word_id], dtype=np.int32)
            h = self.H(x_k)
            ct = mk_ct(gh, h.data[0], h.data.shape[1]) #TODO this doesn't allow batch
            h2 = F.tanh(self.Wc1(ct) + self.Wc2(h))
            loss = F.softmax_cross_entropy(self.W(h2), tx)
            loss_sum += loss

        return loss_sum
