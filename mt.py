import numpy as np
import chainer
from chainer import Variable
import chainer.functions as F
import chainer.links as L

import sobamchan_chainer

class MT(sobamchan_chainer.Model):

    def __init__(self, i_vocab_num, t_vocab_num, k):
        '''
        i_vocab_num: input language vocabulary number
        t_vocab_num: expected language vocabulary number
        k: embedding dimention size, embedding output
        '''
        super(MT, self).__init__(
            embed_x = L.EmbedID(i_vocab_num, k),
            embed_y = L.EmbedID(t_vocab_num, k),
            H = L.LSTM(k, k),
            W = L.Linear(k, t_vocab_num),
        )

    def __call__(self, i_line, t_line):
        '''
        i_line: input language line
        t_line: expected language line
        '''
        # pass though input in LSTM
        self.H.reset_state()
        for i in range(len(i_line)):
            word_id = i_vocab[i_line[i]]
            x_k = self.embed_x(self.prepare_input(word_id, dtype=np.int32))
            h = self.H(x_k)
        x_k = self.embed_x(self.prepare_input(i_line['<eos>'], dtype=np.int32))
        h = self.H(x_k)

        tx = self.prepare_input(t_vocab[t_line[0]], dtype=np.int32)
        loss_sum = F.softmax_cross_entropy(self.W(h), tx)

        for i in range(len(t_line)):
            word_id = t_vocab[t_line[i]]
            x_k = self.embed_y(self.prepare_input(word_id), dtype=np.int32)
            next_word_id = t_vocab['<eos>'] if (i == len(t_line) - 1) else t_vocab[t_line[i+1]]
            tx = self.prepare_input([next_word_id], dtype=np.int32)
            h = self.H(x_k)
            loss = F.softmax_cross_entropy(self.W(h), tx)
            loss_sum += loss

        return loss_sum
