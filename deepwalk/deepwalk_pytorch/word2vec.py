from __future__ import division
import os
import pickle
import random
import argparse
import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from .model import Word2Vec, SGNS,SGNS_dp

import pdb
from .preprocess import Preprocess
import pdb
import math
import pandas as pd
import copy

# from deepwalk.deepwalk import discriminator

class PermutedSubsampledCorpus(Dataset):
    def __init__(self, data, ws=None):
        #data = pickle.load(open(datapath, 'rb'))
        if ws is not None:
            self.data = []
            for iword, owords in data:
                if random.random() > ws[iword]:
                    self.data.append((iword, owords))
        else:
            self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        iword, owords = self.data[idx]
        return iword, np.array(owords)


class ModWord2Vec():
    def __init__(self,sentences=None,
            corpus_file=None, 
            size=100, 
            alpha=0.025, 
            window=5, 
            min_count=5, 
            max_vocab_size=None, 
            sample=0.001, 
            seed=1, 
            workers=3, 
            min_alpha=0.0001, 
            sg=0, 
            hs=0, 
            negative=5, 
            ns_exponent=0.75, 
            cbow_mean=1, 
            hashfxn=None, 
            iter=5,
            null_word='<UNK>', 
            trim_rule=None, 
            sorted_vocab=1, 
            batch_words=10000, 
            compute_loss=False, 
            callbacks=(), 
            max_final_vocab=None):
        self.data = sentences
        self.e_dim = size
        self.alpha = alpha
        self.window = window
        self.min_count = min_count
        self.max_vocab_size = len(sentences)
        self.ss_t = sample
        self.seed = 1
        self.workers = 3
        self.min_alpha = min_alpha
        self.n_negs = negative
        self.ns_exponent = ns_exponent
        self.sg = sg
        self.hs = hs
        self.cbow_mean = cbow_mean
        self.hashfxn = hashfxn
        self.epoch = iter
        self.null_word = null_word
        self.trim_rule = trim_rule
        self.sorted_vocab = sorted_vocab
        self.compute_loss = compute_loss
        self.batch_words = batch_words
        self.callbacks = callbacks
        self.max_final_vocab = None

        self.data = self.preprocess(sentences)
        self.idx2vec = self.train()

    def preprocess(self, sentences):
        pre = Preprocess(self.data, window = self.window, unk = self.null_word, max_vocab = self.max_vocab_size)
        self.idx2word, self.word2idx, self.vocab, self.wc = pre.build()
        return pre.convert()

    def train(self,cuda=False, weights=False):
        wf = np.array([self.wc[word] for word in self.idx2word])
        wf = wf / wf.sum()
        ws = 1 - np.sqrt(self.ss_t / wf)
        ws = np.clip(ws, 0, 1)
        # print('999999',wf)
        vocab_size = len(self.idx2word)
        weights = wf if weights else None
        model = Word2Vec(vocab_size=vocab_size, embedding_size=self.e_dim)
        sgns = SGNS(embedding=model, vocab_size=vocab_size, n_negs=self.n_negs, weights=weights)
        if cuda:
            sgns = sgns.cuda()
        optim = Adam(sgns.parameters())
        for self.epoch in range(1, self.epoch + 1):
            dataset = PermutedSubsampledCorpus(self.data)
            dataloader = DataLoader(dataset, batch_size=self.batch_words, shuffle=True)
            total_batches = int(np.ceil(len(dataset) / self.batch_words))
            # print(len(dataset))
            # print(self.batch_words)
            # exit()
            pbar = tqdm(dataloader)
            pbar.set_description("[Epoch {}]".format(self.epoch))
            for iword, owords in pbar:
                # print(iword)
                # print(owords)
                # print(iword.size())
                # print(owords.size())
                # exit()
                loss = sgns(iword, owords)
                print(loss)
                optim.zero_grad()
                loss.backward()
                optim.step()
                pbar.set_postfix(loss=loss.item())
        idx2vec = model.ivectors.weight.data.cpu().numpy()
        return idx2vec

    def save_emb(self, savepath, num_nodes):
        perm=[]
        for word in range(num_nodes):
            p=self.word2idx[str(word)]
            perm.append(p)
            print(p)

        perm=np.array(perm)

        # perm = np.array([self.word2idx[str(word)] for word in range(num_nodes)])
        emb = self.idx2vec[perm]
        np.save(savepath , emb)

        return emb



class ModWord2Vec_dp():
    def __init__(self,sentences=None,
            corpus_file=None,
            size=100,
            alpha=0.025,
            window=5,
            min_count=5,
            max_vocab_size=None,
            sample=0.001,
            seed=1,
            workers=3,
            min_alpha=0.0001,
            sg=0,
            hs=0,
            negative=5,
            ns_exponent=0.75,
            cbow_mean=1,
            hashfxn=None,
            iter=5,
            null_word='<UNK>',
            trim_rule=None,
            sorted_vocab=1,
            batch_words=10000,
            compute_loss=False,
            callbacks=(),
            max_final_vocab=None,
            budget=400):
        self.data = sentences
        self.e_dim = size
        self.alpha = alpha
        self.window = window
        self.min_count = min_count
        self.max_vocab_size = len(sentences)
        self.ss_t = sample
        self.seed = 1
        self.workers = 3
        self.min_alpha = min_alpha
        self.n_negs = negative
        self.ns_exponent = ns_exponent
        self.sg = sg
        self.hs = hs
        self.cbow_mean = cbow_mean
        self.hashfxn = hashfxn
        self.epoch = iter
        self.null_word = null_word
        self.trim_rule = trim_rule
        self.sorted_vocab = sorted_vocab
        self.compute_loss = compute_loss
        self.batch_words = batch_words
        self.callbacks = callbacks
        self.max_final_vocab = None
        self.sigma = budget

        self.data = self.preprocess(sentences)
        self.idx2vec = self.train()


        # self.use_cuda = torch.cuda.is_available()
        #
        # self.device = torch.device("cuda" if self.use_cuda else "cpu")

    def preprocess(self, sentences):
        pre = Preprocess(self.data, window = self.window, unk = self.null_word, max_vocab = self.max_vocab_size)
        self.idx2word, self.word2idx, self.vocab, self.wc = pre.build()
        return pre.convert()

    def train(self,cuda=False, weights=False):
        wf = np.array([self.wc[word] for word in self.idx2word])
        wf = wf / wf.sum()
        ws = 1 - np.sqrt(self.ss_t / wf)
        ws = np.clip(ws, 0, 1)
        vocab_size = len(self.idx2word)
        weights = wf if weights else None
        model = Word2Vec(vocab_size=vocab_size, embedding_size=self.e_dim)
        sgns = SGNS_dp(embedding=model, vocab_size=vocab_size, n_negs=self.n_negs, weights=weights)
        self.device = torch.device("cuda" if cuda else "cpu")
        if cuda:
            sgns = sgns.cuda()
        optim = Adam(sgns.parameters())
        loss_list = {}
        for self.epoch in range(1, self.epoch + 1):
            loss_s = []
            C=1
            sigma=self.sigma
            print(sigma)
            dataset = PermutedSubsampledCorpus(self.data)
            dataloader = DataLoader(dataset, batch_size=self.batch_words, shuffle=True)
            total_batches = int(np.ceil(len(dataset) / self.batch_words))
            pbar = tqdm(dataloader)
            pbar.set_description("[Epoch {}]".format(self.epoch))
            for iword, owords in pbar:
                loss = sgns(iword, owords)
                optim.zero_grad()

                grads = [torch.zeros(p.shape).to(self.device) for p in sgns.parameters()]

                igrad = torch.autograd.grad(loss, sgns.parameters(), retain_graph=True)
                # print(igradgrads=[torch.zeros(p.shape).to(self.device) for p in sgns.parameters()])


                l2_norm = torch.tensor(0.0).to(self.device)
                for g in igrad:
                    l2_norm += g.norm(2) ** 2
                    # l2_norm += g.sum().square().tolist()
                # print('time12:', int(time.time() / 1000))
                l2_norm = l2_norm.sqrt()
                divisor = max(torch.tensor(1.0).to(self.device), l2_norm / C)
                for i in range(len(igrad)):
                    grads[i] += igrad[i] / divisor

                for i in range(len(grads)):
                    print(grads[i])
                    grads[i] += sigma * C * (torch.randn_like(grads[i]).to(self.device))
                    print(grads[i])
                    grads[i] /= np.shape(iword)[0]
                    grads[i].detach_()

                # exit()

                p_list = [p for p in sgns.parameters()]
                for i in range(len(p_list)):
                    p_list[i].grad = grads[i]
                    print(p_list[i].grad)
                    p_list[i].grad.detach_()

                    print(p_list[i].grad)


                for p in sgns.parameters():
                    print('*******')
                    print(p.grad)

                loss.backward()
                optim.step()
                pbar.set_postfix(loss=loss.item())

                loss_s.append(loss.item())

            loss_list[self.epoch] = loss_s
        idx2vec = model.ivectors.weight.data.cpu().numpy()

        file_ = open( './'+'loss_item'+str(sigma), 'w')
        for los in loss_list:
            line = str()
            for lo in loss_list[los]:
                line += str(lo) + ' '
            line += '\n'
            file_.write(line)
        file_.close()
        return idx2vec

    def save_emb(self, savepath, num_nodes):
        perm = np.array([self.word2idx[str(word)] for word in range(num_nodes)])
        emb = self.idx2vec[perm]
        np.save(savepath , emb)

        return emb

class ModWord2Vec_adv_103():
    def __init__(self,sentences=None,
            corpus_file=None,
            size=100,
            alpha=0.025,
            window=5,
            min_count=5,
            max_vocab_size=None,
            sample=0.001,
            seed=1,
            workers=3,
            min_alpha=0.0001,
            sg=0,
            hs=0,
            negative=5,
            ns_exponent=0.75,
            cbow_mean=1,
            hashfxn=None,
            iter=5,
            null_word='<UNK>',
            trim_rule=None,
            sorted_vocab=1,
            batch_words=10000,
            compute_loss=False,
            callbacks=(),
            max_final_vocab=None,
            budget=400):
        self.data = sentences
        self.e_dim = size
        self.alpha = alpha
        self.window = window
        self.min_count = min_count
        self.max_vocab_size = len(sentences)
        self.ss_t = sample
        self.seed = 1
        self.workers = 3
        self.min_alpha = min_alpha
        self.n_negs = negative
        self.ns_exponent = ns_exponent
        self.sg = sg
        self.hs = hs
        self.cbow_mean = cbow_mean
        self.hashfxn = hashfxn
        self.epoch = iter
        self.null_word = null_word
        self.trim_rule = trim_rule
        self.sorted_vocab = sorted_vocab
        self.compute_loss = compute_loss
        self.batch_words = batch_words
        self.callbacks = callbacks
        self.max_final_vocab = None
        # self.sigma = budget

        self.data = self.preprocess(sentences)
        self.idx2vec = self.train()


        # self.use_cuda = torch.cuda.is_available()
        #
        # self.device = torch.device("cuda" if self.use_cuda else "cpu")

    def preprocess(self, sentences):
        pre = Preprocess(self.data, window = self.window, unk = self.null_word, max_vocab = self.max_vocab_size)
        self.idx2word, self.word2idx, self.vocab, self.wc = pre.build()
        return pre.convert()

    def train(self,cuda=False, weights=False):
        wf = np.array([self.wc[word] for word in self.idx2word])
        wf = wf / wf.sum()
        ws = 1 - np.sqrt(self.ss_t / wf)
        ws = np.clip(ws, 0, 1)
        vocab_size = len(self.idx2word)
        weights = wf if weights else None
        model = Word2Vec(vocab_size=vocab_size, embedding_size=self.e_dim)
        sgns = SGNS_dp(embedding=model, vocab_size=vocab_size, n_negs=self.n_negs, weights=weights)
        self.device = torch.device("cuda" if cuda else "cpu")
        if cuda:
            sgns = sgns.cuda()
        optim = Adam(sgns.parameters())
        loss_list = {}
        for self.epoch in range(1, self.epoch + 1):
            loss_s = []
            C=1
            sigma=self.sigma
            print(sigma)
            dataset = PermutedSubsampledCorpus(self.data)
            dataloader = DataLoader(dataset, batch_size=self.batch_words, shuffle=True)
            total_batches = int(np.ceil(len(dataset) / self.batch_words))
            pbar = tqdm(dataloader)
            pbar.set_description("[Epoch {}]".format(self.epoch))
            for iword, owords in pbar:
                loss = sgns(iword, owords)
                optim.zero_grad()

                grads = [torch.zeros(p.shape).to(self.device) for p in sgns.parameters()]

                igrad = torch.autograd.grad(loss, sgns.parameters(), retain_graph=True)
                # print(igradgrads=[torch.zeros(p.shape).to(self.device) for p in sgns.parameters()])


                l2_norm = torch.tensor(0.0).to(self.device)
                for g in igrad:
                    l2_norm += g.norm(2) ** 2
                    # l2_norm += g.sum().square().tolist()
                # print('time12:', int(time.time() / 1000))
                l2_norm = l2_norm.sqrt()
                divisor = max(torch.tensor(1.0).to(self.device), l2_norm / C)
                for i in range(len(igrad)):
                    grads[i] += igrad[i] / divisor

                for i in range(len(grads)):
                    print(grads[i])
                    grads[i] += sigma * C * (torch.randn_like(grads[i]).to(self.device))
                    print(grads[i])
                    grads[i] /= np.shape(iword)[0]
                    grads[i].detach_()

                # exit()

                p_list = [p for p in sgns.parameters()]
                for i in range(len(p_list)):
                    p_list[i].grad = grads[i]
                    print(p_list[i].grad)
                    p_list[i].grad.detach_()

                    print(p_list[i].grad)


                for p in sgns.parameters():
                    print('*******')
                    print(p.grad)

                loss.backward()
                optim.step()
                pbar.set_postfix(loss=loss.item())

                loss_s.append(loss.item())

            loss_list[self.epoch] = loss_s
        idx2vec = model.ivectors.weight.data.cpu().numpy()

        file_ = open( './'+'loss_item'+str(sigma), 'w')
        for los in loss_list:
            line = str()
            for lo in loss_list[los]:
                line += str(lo) + ' '
            line += '\n'
            file_.write(line)
        file_.close()
        return idx2vec

    def save_emb(self, savepath, num_nodes):
        perm = np.array([self.word2idx[str(word)] for word in range(num_nodes)])
        emb = self.idx2vec[perm]
        np.save(savepath , emb)

        return emb






class ModWord2Vec_defense():
    def __init__(self,train_edges,test_edges,num_nodes,out, sentences=None,
            corpus_file=None,
            size=100,
            alpha=0.025,
            window=5,
            min_count=5,
            max_vocab_size=None,
            sample=0.001,
            seed=1,
            workers=3,
            min_alpha=0.0001,
            sg=0,
            hs=0,
            negative=5,
            ns_exponent=0.75,
            cbow_mean=1,
            hashfxn=None,
            iter=100,
            null_word='<UNK>',
            trim_rule=None,
            sorted_vocab=1,
            batch_words=10000,
            compute_loss=False,
            callbacks=(),
            max_final_vocab=None):
        self.data = sentences
        self.e_dim = size
        self.alpha = alpha
        self.window = window
        self.min_count = min_count
        self.max_vocab_size = len(sentences)
        self.ss_t = sample
        self.seed = 1
        self.workers = 3
        self.min_alpha = min_alpha
        self.n_negs = negative
        self.ns_exponent = ns_exponent
        self.sg = sg
        self.hs = hs
        self.cbow_mean = cbow_mean
        self.hashfxn = hashfxn
        self.epoch = iter
        self.null_word = null_word
        self.trim_rule = trim_rule
        self.sorted_vocab = sorted_vocab
        self.compute_loss = compute_loss
        self.batch_words = batch_words
        self.callbacks = callbacks
        self.max_final_vocab = None
        self.train_edges=train_edges
        self.test_edges=test_edges
        self.num_nodes=num_nodes
        self.out=out

        self.data = self.preprocess(sentences)
        self.idx2vec = self.train()



    def preprocess(self, sentences):
        pre = Preprocess(self.data, window = self.window, unk = self.null_word, max_vocab = self.max_vocab_size)
        self.idx2word, self.word2idx, self.vocab, self.wc = pre.build()
        return pre.convert()

    def train(self,cuda=False, weights=False):
        wf = np.array([self.wc[word] for word in self.idx2word])
        wf = wf / wf.sum()
        ws = 1 - np.sqrt(self.ss_t / wf)
        ws = np.clip(ws, 0, 1)
        vocab_size = len(self.idx2word)
        weights = wf if weights else None
        model = Word2Vec(vocab_size=vocab_size, embedding_size=self.e_dim)
        sgns = SGNS(embedding=model, vocab_size=vocab_size, n_negs=self.n_negs, weights=weights)
        if cuda:
            sgns = sgns.cuda()
        optim = Adam(sgns.parameters())
        for self.epoch in range(1, self.epoch + 1):
            dataset = PermutedSubsampledCorpus(self.data)
            dataloader = DataLoader(dataset, batch_size=self.batch_words, shuffle=True)
            total_batches = int(np.ceil(len(dataset) / self.batch_words))
            print(len(dataset))
            print(self.batch_words)
            # exit()
            pbar = tqdm(dataloader)
            pbar.set_description("[Epoch {}]".format(self.epoch))
            embedding = model
            # print(embedding.ivectors.weight.data.cpu().numpy())
            for iword, owords in pbar:
                loss = sgns(iword, owords)
                print(loss)
                embedding = model
                # print(embedding.ivectors.weight.data.cpu().numpy())
                loss_dis=discriminator_loss(self.train_edges,self.test_edges,self.num_nodes,self.word2idx,embedding=model)
                # print(loss_dis)
                # print(type(loss_dis))
                # print(torch.tensor(loss_dis))
                loss=loss-100*torch.tensor(loss_dis)
                print(loss)
                optim.zero_grad()
                loss.backward(retain_graph=True)
                optim.step()
                pbar.set_postfix(loss=loss.item())
                print(loss)
                embedding = model
                # print(embedding.ivectors.weight.data.cpu().numpy())
        idx2vec = model.ivectors.weight.data.cpu().numpy()
        return idx2vec

    def save_emb(self, savepath, num_nodes):
        perm = np.array([self.word2idx[str(word)] for word in range(num_nodes)])
        emb = self.idx2vec[perm]
        np.save(savepath , emb)

        return emb

class ModWord2Vec_defense2():
    def __init__(self,train_edges,test_edges,num_nodes,out, sentences=None,
            corpus_file=None,
            size=100,
            alpha=0.025,
            window=5,
            min_count=5,
            max_vocab_size=None,
            sample=0.001,
            seed=1,
            workers=3,
            min_alpha=0.0001,
            sg=0,
            hs=0,
            negative=5,
            ns_exponent=0.75,
            cbow_mean=1,
            hashfxn=None,
            iter=5,
            null_word='<UNK>',
            trim_rule=None,
            sorted_vocab=1,
            batch_words=10000,
            compute_loss=False,
            callbacks=(),
            max_final_vocab=None):
        self.data = sentences
        self.e_dim = size
        self.alpha = alpha
        self.window = window
        self.min_count = min_count
        self.max_vocab_size = len(sentences)
        self.ss_t = sample
        self.seed = 1
        self.workers = 3
        self.min_alpha = min_alpha
        self.n_negs = negative
        self.ns_exponent = ns_exponent
        self.sg = sg
        self.hs = hs
        self.cbow_mean = cbow_mean
        self.hashfxn = hashfxn
        self.epoch = iter
        self.null_word = null_word
        self.trim_rule = trim_rule
        self.sorted_vocab = sorted_vocab
        self.compute_loss = compute_loss
        self.batch_words = batch_words
        self.callbacks = callbacks
        self.max_final_vocab = None
        self.train_edges=train_edges
        self.test_edges=test_edges
        self.num_nodes=num_nodes
        self.out=out

        self.data = self.preprocess(sentences)
        self.idx2vec = self.train()



    def preprocess(self, sentences):
        pre = Preprocess(self.data, window = self.window, unk = self.null_word, max_vocab = self.max_vocab_size)
        self.idx2word, self.word2idx, self.vocab, self.wc = pre.build()
        return pre.convert()

    def train(self,cuda=False, weights=False):
        wf = np.array([self.wc[word] for word in self.idx2word])
        wf = wf / wf.sum()
        ws = 1 - np.sqrt(self.ss_t / wf)
        ws = np.clip(ws, 0, 1)
        vocab_size = len(self.idx2word)
        weights = wf if weights else None
        model = Word2Vec(vocab_size=vocab_size, embedding_size=self.e_dim)
        sgns = SGNS(embedding=model, vocab_size=vocab_size, n_negs=self.n_negs, weights=weights)
        if cuda:
            sgns = sgns.cuda()
        optim = Adam(sgns.parameters())
        for self.epoch in range(1, self.epoch + 1):
            dataset = PermutedSubsampledCorpus(self.data)
            dataloader = DataLoader(dataset, batch_size=self.batch_words, shuffle=True)
            total_batches = int(np.ceil(len(dataset) / self.batch_words))
            print(len(dataset))
            print(self.batch_words)
            # exit()
            pbar = tqdm(dataloader)
            pbar.set_description("[Epoch {}]".format(self.epoch))
            embedding = model
            # print(embedding)
            loss_dis,mlp = discriminator_loss2(self.train_edges, self.test_edges, self.num_nodes, self.word2idx,
                                          embedding=model)
            for iword, owords in pbar:
                loss = sgns(iword, owords)
                print(loss)

                # print(loss_dis)
                # print(type(loss_dis))
                # print(torch.tensor(loss_dis))
                loss_dis = discriminator_loss_adv(self.train_edges, self.test_edges, self.num_nodes, self.word2idx,mlp,
                                              embedding=model)
                loss=loss-100*loss_dis
                print(loss)
                optim.zero_grad()
                loss.backward(retain_graph=True)
                optim.step()
                pbar.set_postfix(loss=loss.item())
                print(loss)

                embedding = model
                # print(embedding)
                # exit()
        idx2vec = model.ivectors.weight.data.cpu().numpy()
        return idx2vec

    def save_emb(self, savepath, num_nodes):
        perm = np.array([self.word2idx[str(word)] for word in range(num_nodes)])
        emb = self.idx2vec[perm]
        np.save(savepath , emb)

        return emb


class ModWord2Vec_defense3():
    def __init__(self,F, res_dir,train_edges,test_edges,num_nodes,out, sentences=None,
            corpus_file=None,
            size=100,
            alpha=0.025,
            window=5,
            min_count=5,
            max_vocab_size=None,
            sample=0.001,
            seed=1,
            workers=3,
            min_alpha=0.0001,
            sg=0,
            hs=0,
            negative=5,
            ns_exponent=0.75,
            cbow_mean=1,
            hashfxn=None,
            iter=50,
            null_word='<UNK>',
            trim_rule=None,
            sorted_vocab=1,
            batch_words=10000,
            compute_loss=False,
            callbacks=(),
            max_final_vocab=None):
        self.data = sentences
        self.e_dim = size
        self.alpha = alpha
        self.window = window
        self.min_count = min_count
        self.max_vocab_size = len(sentences)
        self.ss_t = sample
        self.seed = 1
        self.workers = 3
        self.min_alpha = min_alpha
        self.n_negs = negative
        self.ns_exponent = ns_exponent
        self.sg = sg
        self.hs = hs
        self.cbow_mean = cbow_mean
        self.hashfxn = hashfxn
        self.epoch = iter
        self.null_word = null_word
        self.trim_rule = trim_rule
        self.sorted_vocab = sorted_vocab
        self.compute_loss = compute_loss
        self.batch_words = batch_words
        self.callbacks = callbacks
        self.max_final_vocab = None
        self.train_edges=train_edges
        self.test_edges=test_edges
        self.num_nodes=num_nodes
        self.out=out
        self.F=F
        self.res_dir=res_dir

        self.data = self.preprocess(sentences)
        self.idx2vec = self.train()



    def preprocess(self, sentences):
        pre = Preprocess(self.data, window = self.window, unk = self.null_word, max_vocab = self.max_vocab_size)
        self.idx2word, self.word2idx, self.vocab, self.wc = pre.build()
        return pre.convert()

    def train(self,cuda=False, weights=False):
        wf = np.array([self.wc[word] for word in self.idx2word])
        wf = wf / wf.sum()
        ws = 1 - np.sqrt(self.ss_t / wf)
        ws = np.clip(ws, 0, 1)
        # print('99999',wf)
        vocab_size = len(self.idx2word)
        weights = wf if weights else None
        print('wwww',weights)
        model = Word2Vec(vocab_size=vocab_size, embedding_size=self.e_dim)
        sgns = SGNS(embedding=model, vocab_size=vocab_size, n_negs=self.n_negs, weights=weights)
        # print('sgn',sgns)
        # print(sgns.type())
        if cuda:
            sgns = sgns.cuda()
        optim = Adam(sgns.parameters())
        min=10000

        cnt_it=0

        edgeall = list([list(edge_tuple) for edge_tuple in self.train_edges])

        # train_edges_sampled = random.sample(edgeall, np.shape(test_edges)[0])
        train_edges_sampled = random.sample(edgeall, np.shape(self.test_edges)[0])

        for self.epoch in range(1, self.epoch + 1):
            dataset = PermutedSubsampledCorpus(self.data)
            dataloader = DataLoader(dataset, batch_size=self.batch_words, shuffle=True)
            total_batches = int(np.ceil(len(dataset) / self.batch_words))
            # print(len(dataset))
            # print(self.batch_words)
            # exit()
            pbar = tqdm(dataloader)
            pbar.set_description("[Epoch {}]".format(self.epoch))
            embedding = model
            # print('ppp',embedding)
            loss_dis,mlp = discriminator_loss2(train_edges_sampled, self.test_edges, self.num_nodes, self.word2idx,
                                          embedding=model)
            for iword, owords in pbar:
                print(iword)
                loss = sgns(iword, owords)
                print('embed_loss')
                print(loss)

                # print(loss_dis)
                # print(type(loss_dis))
                # print(torch.tensor(loss_dis))
                if cnt_it==0:
                    gain_dis,acc_kmeans_sim, acc_mlp_sim, acc_rf_sim, acc_svm_sim = discriminator_gain(train_edges_sampled, self.test_edges, self.num_nodes, self.word2idx,mlp,cnt_it,self.F, self.res_dir,
                                              embedding=model)

                else:
                    gain_dis= discriminator_gain(
                        train_edges_sampled, self.test_edges, self.num_nodes, self.word2idx, mlp, cnt_it, self.F,
                        self.res_dir,
                        embedding=model)
                print('dis')
                print(gain_dis)
                print('before')
                print((loss))

                loss=(loss+torch.tensor(0.01*gain_dis))
                print(type(loss))
                optim.zero_grad()
                loss.backward()
                # loss2=torch.tensor(gain_dis)
                # loss2.backward()
                optim.step()
                print('777:',loss.item())
                pbar.set_postfix(loss=loss.item())
                print('after')
                print(loss)

                if min>loss:
                    print('~~~~~~')
                    idx2vec = model.ivectors.weight.data.cpu().numpy()
                    idx2vec=copy.deepcopy(idx2vec)
                    # print('min')
                    # print(idx2vec[9])
                    min=loss
                    cnt_it =0

                if min<loss:
                    print('@@@@@@@@')
                    print(cnt_it)
                    cnt_it+=1
                    if cnt_it>=5:
                        if self.epoch>1:
                            break

            if self.epoch>1 and cnt_it>=5:
                break

            # print(idx2vec[9])
        # discriminator_gain2(train_edges_sampled, self.test_edges, self.num_nodes, self.word2idx, mlp,
        #                     self.F, self.res_dir,
        #                     idx2vec)

        with open('./data/' + 'embeds-' + self.F,'w') as f:
            perm = np.array([self.word2idx[str(word)] for word in range(self.num_nodes)])
            emb = idx2vec[perm]
            f.write('%d %d\n' % (self.num_nodes, self.e_dim))
            for i in range(self.num_nodes):
                e = ' '.join(map(lambda x: str(x), emb[i]))
                f.write('%s %s\n' % (str(i), e))

        print(acc_kmeans_sim, acc_mlp_sim, acc_rf_sim, acc_svm_sim)





        # print(embedding)
        # exit()

        return idx2vec

    def save_emb(self, savepath, num_nodes):
        perm = np.array([self.word2idx[str(word)] for word in range(num_nodes)])
        emb = self.idx2vec[perm]
        np.save(savepath , emb)

        return emb


class ModWord2Vec5():
    def __init__(self,sentences=None,
            corpus_file=None,
            size=100,
            alpha=0.025,
            window=5,
            min_count=5,
            max_vocab_size=None,
            sample=0.001,
            seed=1,
            workers=3,
            min_alpha=0.0001,
            sg=0,
            hs=0,
            negative=5,
            ns_exponent=0.75,
            cbow_mean=1,
            hashfxn=None,
            iter=51,
            null_word='<UNK>',
            trim_rule=None,
            sorted_vocab=1,
            batch_words=10000,
            compute_loss=False,
            callbacks=(),
            max_final_vocab=None):
        self.data = sentences
        self.e_dim = size
        self.alpha = alpha
        self.window = window
        self.min_count = min_count
        self.max_vocab_size = len(sentences)
        self.ss_t = sample
        self.seed = 1
        self.workers = 3
        self.min_alpha = min_alpha
        self.n_negs = negative
        self.ns_exponent = ns_exponent
        self.sg = sg
        self.hs = hs
        self.cbow_mean = cbow_mean
        self.hashfxn = hashfxn
        self.epoch = iter
        self.null_word = null_word
        self.trim_rule = trim_rule
        self.sorted_vocab = sorted_vocab
        self.compute_loss = compute_loss
        self.batch_words = batch_words
        self.callbacks = callbacks
        self.max_final_vocab = None

        self.data = self.preprocess(sentences)
        self.idx2vec = self.train()

    def preprocess(self, sentences):
        pre = Preprocess(self.data, window = self.window, unk = self.null_word, max_vocab = self.max_vocab_size)
        self.idx2word, self.word2idx, self.vocab, self.wc = pre.build()
        return pre.convert()

    def train(self,cuda=False, weights=False):
        wf = np.array([self.wc[word] for word in self.idx2word])
        wf = wf / wf.sum()
        ws = 1 - np.sqrt(self.ss_t / wf)
        ws = np.clip(ws, 0, 1)
        vocab_size = len(self.idx2word)
        weights = wf if weights else None
        model = Word2Vec(vocab_size=vocab_size, embedding_size=self.e_dim)
        sgns = SGNS(embedding=model, vocab_size=vocab_size, n_negs=self.n_negs, weights=weights)
        if cuda:
            sgns = sgns.cuda()
        optim = Adam(sgns.parameters())
        for self.epoch in range(1, self.epoch + 1):
            dataset = PermutedSubsampledCorpus(self.data)
            dataloader = DataLoader(dataset, batch_size=self.batch_words, shuffle=True)
            total_batches = int(np.ceil(len(dataset) / self.batch_words))
            # print(len(dataset))
            # print(self.batch_words)
            # exit()
            pbar = tqdm(dataloader)
            pbar.set_description("[Epoch {}]".format(self.epoch))
            for iword, owords in pbar:
                # print(iword)
                # print(owords)
                # print(iword.size())
                # print(owords.size())
                # exit()
                loss = sgns(iword, owords)
                print(loss)
                optim.zero_grad()
                loss.backward()
                optim.step()
                pbar.set_postfix(loss=loss.item())
        idx2vec = model.ivectors.weight.data.cpu().numpy()
        return idx2vec

    def save_emb(self, savepath, num_nodes):
        perm = np.array([self.word2idx[str(word)] for word in range(num_nodes)])
        emb = self.idx2vec[perm]
        np.save(savepath , emb)

        return emb





def discriminator(train_edges,test_edges,num_nodes,word2idx,embedding):
    idx2vec = embedding.ivectors.weight.data.cpu().numpy()
    # print(num_nodes)
    # print(word2idx)

    perm = np.array([word2idx[str(word)] for word in range(num_nodes)])
    emb = idx2vec[perm]

    # start_time = time.time()
    # Generate bootstrapped edge embeddings (as is done in node2vec paper)
    # Edge embedding for (v1, v2) = hadamard product of node embeddings for v1, v2
    emb_matrix=emb
    def get_edge_embeddings(edge_list):
        embs = []
        sim_matrix=[]
        embs_1=[]
        embs_2 = []
        for edge in edge_list:
            node1 = edge[0]
            node2 = edge[1]
            emb1 = emb_matrix[node1]
            #print(np.shape(emb1))
            emb2 = emb_matrix[node2]
            edge_emb = np.multiply(emb1, emb2)
            sim = np.dot(emb1, emb2)/(np.linalg.norm(emb1)*np.linalg.norm(emb2))
            #edge_emb = np.array(emb1) + np.array(emb2)
            # print(np.shape(edge_emb))
            embs.append(edge_emb)
            embs_1.append(emb1)
            embs_2.append(emb2)
            sim_matrix.append(sim)
        embs = np.array(embs)
        sim_matrix = np.array(sim_matrix)
        embs_1=np.array(embs_1)
        embs_2 =np.array(embs_2)

        # #with open('/Users/xiulingwang/Downloads/line-master/data/embds/' + str(ego_user) + flag + '-' + Flag, 'w') as f:
        # with open('E:/python/banlance/code/' + DATASET + '/' + METHOD + '/embeds/' + Flag + '-' + str(ego_user) + flag,'w') as f:
        # #with open('/Users/xiulingwang/Downloads/'+DATASET+'/line/3-split/' + str(ego_user)+ '-' + flag + '-' + Flag+'-' +'embds','w') as f:
        #     f.write('%d %d\n' % (edge_list.shape[0], args.representation_size))
        #     for i in range(edge_list.shape[0]):
        #         e = ' '.join(map(lambda x: str(x), embs[i]))
        #         f.write('%s %s %s\n' % (str(edge_list[i][0]), str(edge_list[i][1]), e))

        return embs,sim_matrix,embs_1,embs_2

    edgeall = list([list(edge_tuple) for edge_tuple in train_edges])

    # train_edges_sampled = random.sample(edgeall, np.shape(test_edges)[0])
    train_edges_sampled = random.sample(edgeall, np.shape(test_edges)[0])


    # Train-set edge embeddings
    pos_train_edge_embs ,pos_train_sim_matrix,pos_embs_1_train,pos_embs_2_train= get_edge_embeddings(train_edges_sampled)
    # neg_train_edge_embs,neg_train_sim_matrix,neg_embs_1_train,neg_embs_2_train = get_edge_embeddings(train_edges_false, ego_user,DATASET,Flag, flag='neg-train')
    train_edge_embs = pos_train_edge_embs
    train_sim_matrix= pos_train_sim_matrix
    train_embs_1 = pos_embs_1_train
    train_embs_2 = pos_embs_2_train

    # Create train-set edge labels: 1 = real edge, 0 = false edge
    train_edge_labels = np.ones(len(train_edges_sampled))


    # Test-set edge embeddings, labels
    pos_test_edge_embs,pos_test_sim_matrix,pos_embs_1_test,pos_embs_2_test = get_edge_embeddings(test_edges)
    # neg_test_edge_embs ,neg_test_sim_matrix,neg_embs_1_test,neg_embs_2_test= get_edge_embeddings(test_edges_false,ego_user,DATASET,Flag, flag='neg-test')
    test_edge_embs = pos_test_edge_embs
    test_sim_matrix = pos_test_sim_matrix
    test_embs_1 = pos_embs_1_test
    test_embs_2 = pos_embs_2_test

    # Create val-set edge labels: 1 = real edge, 0 = false edge
    test_edge_labels = np.ones(len(test_edges))


    ###########sim_svm

    train_edges_list = np.array(train_edges_sampled)
    # print(train_edges_list)
    test_edges_list = test_edges
    # print(test_edges_list)

    edges_list = np.concatenate((train_edges_list, test_edges_list), axis=0)

    # print(type(train_edges_list))
    # print(type(test_edges_list))
    # print(type(edges_list))

    # print(np.shape(train_edges_list))
    # print(np.shape(test_edges_list))
    # print(np.shape(edges_list))

    ylabel = [1] * train_sim_matrix.shape[0] + [0] * test_sim_matrix.shape[0]

    # print(train_sim_matrix)
    # print(test_sim_matrix)

    sim_matrix = np.concatenate((train_sim_matrix, test_sim_matrix), axis=0)
    # print(sim_matrix)
    # print(np.shape(train_sim_matrix))
    # print(np.shape(test_sim_matrix))
    sim_matrix = sim_matrix.reshape(-1, 1)
    # print(sim_matrix)
    # print(np.shape(sim_matrix))
    # exit()

    sim_matrix_train = train_sim_matrix
    sim_matrix_test = test_sim_matrix

    sim_matrix_train = sim_matrix_train.reshape(-1, 1)
    sim_matrix_test = sim_matrix_test.reshape(-1, 1)

    # print(np.shape(sim_matrix_train))
    # print(np.shape(sim_matrix_test))

    from sklearn.model_selection import train_test_split

    ylabel1 = ylabel
    ylable1 = np.reshape(len(ylabel1), 1)

    # print((edges_list))
    # print((ylabel1))
    # print(np.shape(ylabel1))
    # print(np.shape(edges_list))
    y_label = np.zeros((np.shape(edges_list)[0], 3))
    for i in range(np.shape(edges_list)[0]):
        y_label[i][0] = edges_list[i][0]
        y_label[i][1] = edges_list[i][1]
        y_label[i][2] = ylabel[i]
    # print(np.shape(y_label))

    y_label_train = np.zeros((np.shape(train_edges_list)[0], 3))
    for i in range(np.shape(train_edges_list)[0]):
        y_label_train[i][0] = train_edges_list[i][0]
        y_label_train[i][1] = train_edges_list[i][1]
        y_label_train[i][2] = 1
    # print(np.shape(y_label_train))

    y_label_test = np.zeros((np.shape(test_edges_list)[0], 3))
    for i in range(np.shape(test_edges_list)[0]):
        y_label_test[i][0] = test_edges_list[i][0]
        y_label_test[i][1] = test_edges_list[i][1]
        y_label_test[i][2] = 0
    # print(np.shape(y_label_test))

    X_train_train, X_train_test, y_train_train, y_train_test = train_test_split(sim_matrix_train, y_label_train,
                                                                                test_size=0.1, random_state=42)

    X_test_train, X_test_test, y_test_train, y_test_test = train_test_split(sim_matrix_test, y_label_test,
                                                                            test_size=0.1, random_state=42)

    X_train = np.concatenate((X_train_train, X_test_train), axis=0)
    X_test = np.concatenate((X_train_test, X_test_test), axis=0)
    y_train = np.concatenate((y_train_train, y_test_train), axis=0)
    y_test = np.concatenate((y_train_test, y_test_test), axis=0)

    from sklearn import metrics
    from sklearn.neural_network import MLPClassifier

    mlp = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(64, 32, 16, 18), random_state=1,
                        max_iter=500)

    mlp.fit(X_train, y_train[:, 2])

    loss=mlp.loss_

    # print("Training set score: %f" % mlp.score(X_train, y_train[:, 2]))
    # print("Test set score: %f" % mlp.score(X_test, y_test[:, 2]))
    #
    # y_score = mlp.predict(X_test)
    # print(metrics.f1_score(y_test[:, 2], y_score, average='micro'))
    # print(metrics.classification_report(y_test[:, 2], y_score, labels=range(3)))

    return loss


def discriminator_loss(train_edges,test_edges,num_nodes,word2idx,embedding):
    idx2vec = embedding.ivectors.weight.data.cpu().numpy()
    # print(num_nodes)
    # print(word2idx)

    perm = np.array([word2idx[str(word)] for word in range(num_nodes)])
    emb = idx2vec[perm]

    # start_time = time.time()
    # Generate bootstrapped edge embeddings (as is done in node2vec paper)
    # Edge embedding for (v1, v2) = hadamard product of node embeddings for v1, v2
    emb_matrix=emb
    def get_edge_embeddings(edge_list):
        embs = []
        sim_matrix=[]
        embs_1=[]
        embs_2 = []
        for edge in edge_list:
            node1 = edge[0]
            node2 = edge[1]
            emb1 = emb_matrix[node1]
            #print(np.shape(emb1))
            emb2 = emb_matrix[node2]
            edge_emb = np.multiply(emb1, emb2)
            sim = np.dot(emb1, emb2)/(np.linalg.norm(emb1)*np.linalg.norm(emb2))
            #edge_emb = np.array(emb1) + np.array(emb2)
            # print(np.shape(edge_emb))
            embs.append(edge_emb)
            embs_1.append(emb1)
            embs_2.append(emb2)
            sim_matrix.append(sim)
        embs = np.array(embs)
        sim_matrix = np.array(sim_matrix)
        embs_1=np.array(embs_1)
        embs_2 =np.array(embs_2)

        # #with open('/Users/xiulingwang/Downloads/line-master/data/embds/' + str(ego_user) + flag + '-' + Flag, 'w') as f:
        # with open('E:/python/banlance/code/' + DATASET + '/' + METHOD + '/embeds/' + Flag + '-' + str(ego_user) + flag,'w') as f:
        # #with open('/Users/xiulingwang/Downloads/'+DATASET+'/line/3-split/' + str(ego_user)+ '-' + flag + '-' + Flag+'-' +'embds','w') as f:
        #     f.write('%d %d\n' % (edge_list.shape[0], args.representation_size))
        #     for i in range(edge_list.shape[0]):
        #         e = ' '.join(map(lambda x: str(x), embs[i]))
        #         f.write('%s %s %s\n' % (str(edge_list[i][0]), str(edge_list[i][1]), e))

        return embs,sim_matrix,embs_1,embs_2

    edgeall = list([list(edge_tuple) for edge_tuple in train_edges])
    np.random.seed(0)
    # train_edges_sampled = random.sample(edgeall, np.shape(test_edges)[0])
    train_edges_sampled = random.sample(edgeall, np.shape(test_edges)[0])


    # Train-set edge embeddings
    pos_train_edge_embs ,pos_train_sim_matrix,pos_embs_1_train,pos_embs_2_train= get_edge_embeddings(train_edges_sampled)
    # neg_train_edge_embs,neg_train_sim_matrix,neg_embs_1_train,neg_embs_2_train = get_edge_embeddings(train_edges_false, ego_user,DATASET,Flag, flag='neg-train')
    train_edge_embs = pos_train_edge_embs
    train_sim_matrix= pos_train_sim_matrix
    train_embs_1 = pos_embs_1_train
    train_embs_2 = pos_embs_2_train

    # Create train-set edge labels: 1 = real edge, 0 = false edge
    train_edge_labels = np.ones(len(train_edges_sampled))


    # Test-set edge embeddings, labels
    pos_test_edge_embs,pos_test_sim_matrix,pos_embs_1_test,pos_embs_2_test = get_edge_embeddings(test_edges)
    # neg_test_edge_embs ,neg_test_sim_matrix,neg_embs_1_test,neg_embs_2_test= get_edge_embeddings(test_edges_false,ego_user,DATASET,Flag, flag='neg-test')
    test_edge_embs = pos_test_edge_embs
    test_sim_matrix = pos_test_sim_matrix
    test_embs_1 = pos_embs_1_test
    test_embs_2 = pos_embs_2_test

    # Create val-set edge labels: 1 = real edge, 0 = false edge
    test_edge_labels = np.ones(len(test_edges))


    ###########sim_svm

    train_edges_list = np.array(train_edges_sampled)
    # print(train_edges_list)
    test_edges_list = test_edges
    # print(test_edges_list)

    edges_list = np.concatenate((train_edges_list, test_edges_list), axis=0)

    # print(type(train_edges_list))
    # print(type(test_edges_list))
    # print(type(edges_list))

    # print(np.shape(train_edges_list))
    # print(np.shape(test_edges_list))
    # print(np.shape(edges_list))

    ylabel = [1] * train_sim_matrix.shape[0] + [0] * test_sim_matrix.shape[0]

    # print(train_sim_matrix)
    # print(test_sim_matrix)

    loss_train = []
    for i in range(len(train_embs_1)):
        los_denominator = 0
        for j in range(np.shape(emb_matrix)[0]):
            if i != j:
                los_denominator += math.exp(np.dot(train_embs_1[i], emb_matrix[j]))
        los = math.exp(np.dot(train_embs_1[i], train_embs_2[i]) / los_denominator - train_edge_labels[i]) ** 2
        loss_train.append(los)
        # print(train_embs_1[i], train_embs_2[i],train_edge_labels[i])

    loss_test = []
    for i in range(len(test_embs_1)):
        los_denominator = 0
        for j in range(np.shape(emb_matrix)[0]):
            if i != j:
                los_denominator += math.exp(np.dot(test_embs_1[i], emb_matrix[j]))
        los = math.exp(np.dot(test_embs_1[i], test_embs_2[i]) / los_denominator - test_edge_labels[i]) ** 2

        loss_test.append(los)

    loss_matrix = np.concatenate((loss_train, loss_test), axis=0)
    # print(loss_matrix)

    loss_matrix = loss_matrix.reshape(-1, 1)
    # print(loss_matrix)

    loss_matrix_train = np.array(loss_train)
    # print(loss_matrix_train)

    loss_matrix_test = np.array(loss_test)
    # print(loss_matrix_test)

    loss_matrix_train = loss_matrix_train.reshape(-1, 1)
    # print(loss_matrix_train)

    loss_matrix_test = loss_matrix_test.reshape(-1, 1)
    # print(loss_matrix_test)
    from sklearn.model_selection import train_test_split

    ylabel1 = ylabel
    ylable1 = np.reshape(len(ylabel1), 1)

    # print((edges_list))
    # print((ylabel1))
    # print(np.shape(ylabel1))
    # print(np.shape(edges_list))
    y_label = np.zeros((np.shape(edges_list)[0], 3))
    for i in range(np.shape(edges_list)[0]):
        y_label[i][0] = edges_list[i][0]
        y_label[i][1] = edges_list[i][1]
        y_label[i][2] = ylabel[i]
    # print(np.shape(y_label))

    y_label_train = np.zeros((np.shape(train_edges_list)[0], 3))
    for i in range(np.shape(train_edges_list)[0]):
        y_label_train[i][0] = train_edges_list[i][0]
        y_label_train[i][1] = train_edges_list[i][1]
        y_label_train[i][2] = 1
    # print(np.shape(y_label_train))

    y_label_test = np.zeros((np.shape(test_edges_list)[0], 3))
    for i in range(np.shape(test_edges_list)[0]):
        y_label_test[i][0] = test_edges_list[i][0]
        y_label_test[i][1] = test_edges_list[i][1]
        y_label_test[i][2] = 0
    # print(np.shape(y_label_test))

    X_train_train, X_train_test, y_train_train, y_train_test = train_test_split(loss_matrix_train, y_label_train,
                                                                                test_size=0.3, random_state=42)

    X_test_train, X_test_test, y_test_train, y_test_test = train_test_split(loss_matrix_test, y_label_test,
                                                                            test_size=0.3, random_state=42)

    X_train = np.concatenate((X_train_train, X_test_train), axis=0)
    X_test = np.concatenate((X_train_test, X_test_test), axis=0)
    y_train = np.concatenate((y_train_train, y_test_train), axis=0)
    y_test = np.concatenate((y_train_test, y_test_test), axis=0)


    from sklearn import metrics
    from sklearn.neural_network import MLPClassifier

    mlp = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(64, 32, 16, 18), random_state=1,
                        max_iter=500)

    mlp.fit(X_train, y_train[:, 2])

    loss=mlp.loss_

    # print("Training set score: %f" % mlp.score(X_train, y_train[:, 2]))
    # print("Test set score: %f" % mlp.score(X_test, y_test[:, 2]))
    #
    # y_score = mlp.predict(X_test)
    # loss=nn.BCELoss(y_score,y_test[:, 2])

    # print(metrics.f1_score(y_test[:, 2], y_score, average='micro'))
    # print(metrics.classification_report(y_test[:, 2], y_score, labels=range(3)))

    return loss

def discriminator_loss2(train_edges_sampled,test_edges,num_nodes,word2idx,embedding):
    idx2vec = embedding.ivectors.weight.data.cpu().numpy()
    # print(num_nodes)
    # print(word2idx)

    perm = np.array([word2idx[str(word)] for word in range(num_nodes)])
    emb = idx2vec[perm]

    # start_time = time.time()
    # Generate bootstrapped edge embeddings (as is done in node2vec paper)
    # Edge embedding for (v1, v2) = hadamard product of node embeddings for v1, v2
    emb_matrix=emb
    def get_edge_embeddings(edge_list):
        embs = []
        sim_matrix=[]
        embs_1=[]
        embs_2 = []
        for edge in edge_list:
            node1 = edge[0]
            node2 = edge[1]
            emb1 = emb_matrix[node1]
            #print(np.shape(emb1))
            emb2 = emb_matrix[node2]
            edge_emb = np.multiply(emb1, emb2)
            sim = np.dot(emb1, emb2)/(np.linalg.norm(emb1)*np.linalg.norm(emb2))
            #edge_emb = np.array(emb1) + np.array(emb2)
            # print(np.shape(edge_emb))
            embs.append(edge_emb)
            embs_1.append(emb1)
            embs_2.append(emb2)
            sim_matrix.append(sim)
        embs = np.array(embs)
        sim_matrix = np.array(sim_matrix)
        embs_1=np.array(embs_1)
        embs_2 =np.array(embs_2)

        # #with open('/Users/xiulingwang/Downloads/line-master/data/embds/' + str(ego_user) + flag + '-' + Flag, 'w') as f:
        # with open('E:/python/banlance/code/' + DATASET + '/' + METHOD + '/embeds/' + Flag + '-' + str(ego_user) + flag,'w') as f:
        # #with open('/Users/xiulingwang/Downloads/'+DATASET+'/line/3-split/' + str(ego_user)+ '-' + flag + '-' + Flag+'-' +'embds','w') as f:
        #     f.write('%d %d\n' % (edge_list.shape[0], args.representation_size))
        #     for i in range(edge_list.shape[0]):
        #         e = ' '.join(map(lambda x: str(x), embs[i]))
        #         f.write('%s %s %s\n' % (str(edge_list[i][0]), str(edge_list[i][1]), e))

        return embs,sim_matrix,embs_1,embs_2

    # edgeall = list([list(edge_tuple) for edge_tuple in train_edges])
    # np.random.seed(0)
    # # train_edges_sampled = random.sample(edgeall, np.shape(test_edges)[0])
    # train_edges_sampled = random.sample(edgeall, np.shape(test_edges)[0])


    # Train-set edge embeddings
    pos_train_edge_embs ,pos_train_sim_matrix,pos_embs_1_train,pos_embs_2_train= get_edge_embeddings(train_edges_sampled)
    # neg_train_edge_embs,neg_train_sim_matrix,neg_embs_1_train,neg_embs_2_train = get_edge_embeddings(train_edges_false, ego_user,DATASET,Flag, flag='neg-train')
    train_edge_embs = pos_train_edge_embs
    train_sim_matrix= pos_train_sim_matrix
    train_embs_1 = pos_embs_1_train
    train_embs_2 = pos_embs_2_train

    # Create train-set edge labels: 1 = real edge, 0 = false edge
    train_edge_labels = np.ones(len(train_edges_sampled))


    # Test-set edge embeddings, labels
    pos_test_edge_embs,pos_test_sim_matrix,pos_embs_1_test,pos_embs_2_test = get_edge_embeddings(test_edges)
    # neg_test_edge_embs ,neg_test_sim_matrix,neg_embs_1_test,neg_embs_2_test= get_edge_embeddings(test_edges_false,ego_user,DATASET,Flag, flag='neg-test')
    test_edge_embs = pos_test_edge_embs
    test_sim_matrix = pos_test_sim_matrix
    test_embs_1 = pos_embs_1_test
    test_embs_2 = pos_embs_2_test

    # Create val-set edge labels: 1 = real edge, 0 = false edge
    test_edge_labels = np.ones(len(test_edges))


    ###########sim_svm

    train_edges_list = np.array(train_edges_sampled)
    # print(train_edges_list)
    test_edges_list = test_edges
    # print(test_edges_list)

    edges_list = np.concatenate((train_edges_list, test_edges_list), axis=0)

    # print(type(train_edges_list))
    # print(type(test_edges_list))
    # print(type(edges_list))

    # print(np.shape(train_edges_list))
    # print(np.shape(test_edges_list))
    # print(np.shape(edges_list))

    ylabel = [1] * train_sim_matrix.shape[0] + [0] * test_sim_matrix.shape[0]

    # print(train_sim_matrix)
    # print(test_sim_matrix)

    sim_matrix = np.concatenate((train_sim_matrix, test_sim_matrix), axis=0)
    # print(sim_matrix)
    # print(np.shape(train_sim_matrix))
    # print(np.shape(test_sim_matrix))
    sim_matrix = sim_matrix.reshape(-1, 1)
    # print(sim_matrix)
    # print(np.shape(sim_matrix))
    # exit()

    sim_matrix_train = train_sim_matrix
    sim_matrix_test = test_sim_matrix

    sim_matrix_train = sim_matrix_train.reshape(-1, 1)
    sim_matrix_test = sim_matrix_test.reshape(-1, 1)

    # print(np.shape(sim_matrix_train))
    # print(np.shape(sim_matrix_test))    # print(loss_matrix_test)
    from sklearn.model_selection import train_test_split

    ylabel1 = ylabel
    ylable1 = np.reshape(len(ylabel1), 1)

    # print((edges_list))
    # print((ylabel1))
    # print(np.shape(ylabel1))
    # print(np.shape(edges_list))
    y_label = np.zeros((np.shape(edges_list)[0], 3))
    for i in range(np.shape(edges_list)[0]):
        y_label[i][0] = edges_list[i][0]
        y_label[i][1] = edges_list[i][1]
        y_label[i][2] = ylabel[i]
    # print(np.shape(y_label))

    y_label_train = np.zeros((np.shape(train_edges_list)[0], 3))
    for i in range(np.shape(train_edges_list)[0]):
        y_label_train[i][0] = train_edges_list[i][0]
        y_label_train[i][1] = train_edges_list[i][1]
        y_label_train[i][2] = 1
    # print(np.shape(y_label_train))

    y_label_test = np.zeros((np.shape(test_edges_list)[0], 3))
    for i in range(np.shape(test_edges_list)[0]):
        y_label_test[i][0] = test_edges_list[i][0]
        y_label_test[i][1] = test_edges_list[i][1]
        y_label_test[i][2] = 0
    # print(np.shape(y_label_test))

    X_train_train, X_train_test, y_train_train, y_train_test = train_test_split(sim_matrix_train, y_label_train,
                                                                                test_size=0.3, random_state=42)

    X_test_train, X_test_test, y_test_train, y_test_test = train_test_split(sim_matrix_test, y_label_test,
                                                                            test_size=0.3, random_state=42)

    X_train = np.concatenate((X_train_train, X_test_train), axis=0)
    X_test = np.concatenate((X_train_test, X_test_test), axis=0)
    y_train = np.concatenate((y_train_train, y_test_train), axis=0)
    y_test = np.concatenate((y_train_test, y_test_test), axis=0)

    from sklearn import metrics
    from sklearn.neural_network import MLPClassifier

    mlp = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(64, 32, 16, 18), random_state=1,
                        max_iter=500)

    mlp.fit(X_train, y_train[:, 2])

    # loss=mlp.loss_

    # print("Training set score: %f" % mlp.score(X_train, y_train[:, 2]))
    # print("Test set score: %f" % mlp.score(X_test, y_test[:, 2]))
    from sklearn import metrics
    y_score = mlp.predict(X_test)
    # acc_mlp_sim = accuracy_score(y_score, y_test[:, 2])
    # print(y_score)
    # print(np.shape(y_score))
    # print(y_test[:, 2])
    # print(np.shape(y_test[:, 2]))
    ls = 0
    for i in range(len(y_score)):
        if y_score[i] != y_test[i][2]:
            if y_score[i] == 1:
                ls += y_score[i]
            else:
                ls += 1-y_score[i]
    loss = ls / len(y_score)



    # print(metrics.f1_score(y_test[:, 2], y_score, average='micro'))
    # print(metrics.classification_report(y_test[:, 2], y_score, labels=range(3)))

    return loss,mlp



def discriminator_loss_adv(train_edges,test_edges,num_nodes,word2idx,mlp,embedding):
    idx2vec = embedding.ivectors.weight.data.cpu().numpy()
    # print(num_nodes)
    # print(word2idx)

    perm = np.array([word2idx[str(word)] for word in range(num_nodes)])
    emb = idx2vec[perm]

    # start_time = time.time()
    # Generate bootstrapped edge embeddings (as is done in node2vec paper)
    # Edge embedding for (v1, v2) = hadamard product of node embeddings for v1, v2
    emb_matrix=emb
    def get_edge_embeddings(edge_list):
        embs = []
        sim_matrix=[]
        embs_1=[]
        embs_2 = []
        for edge in edge_list:
            node1 = edge[0]
            node2 = edge[1]
            emb1 = emb_matrix[node1]
            #print(np.shape(emb1))
            emb2 = emb_matrix[node2]
            edge_emb = np.multiply(emb1, emb2)
            sim = np.dot(emb1, emb2)/(np.linalg.norm(emb1)*np.linalg.norm(emb2))
            #edge_emb = np.array(emb1) + np.array(emb2)
            # print(np.shape(edge_emb))
            embs.append(edge_emb)
            embs_1.append(emb1)
            embs_2.append(emb2)
            sim_matrix.append(sim)
        embs = np.array(embs)
        sim_matrix = np.array(sim_matrix)
        embs_1=np.array(embs_1)
        embs_2 =np.array(embs_2)

        # #with open('/Users/xiulingwang/Downloads/line-master/data/embds/' + str(ego_user) + flag + '-' + Flag, 'w') as f:
        # with open('E:/python/banlance/code/' + DATASET + '/' + METHOD + '/embeds/' + Flag + '-' + str(ego_user) + flag,'w') as f:
        # #with open('/Users/xiulingwang/Downloads/'+DATASET+'/line/3-split/' + str(ego_user)+ '-' + flag + '-' + Flag+'-' +'embds','w') as f:
        #     f.write('%d %d\n' % (edge_list.shape[0], args.representation_size))
        #     for i in range(edge_list.shape[0]):
        #         e = ' '.join(map(lambda x: str(x), embs[i]))
        #         f.write('%s %s %s\n' % (str(edge_list[i][0]), str(edge_list[i][1]), e))

        return embs,sim_matrix,embs_1,embs_2

    edgeall = list([list(edge_tuple) for edge_tuple in train_edges])

    # train_edges_sampled = random.sample(edgeall, np.shape(test_edges)[0])
    train_edges_sampled = random.sample(edgeall, np.shape(test_edges)[0])


    # Train-set edge embeddings
    pos_train_edge_embs ,pos_train_sim_matrix,pos_embs_1_train,pos_embs_2_train= get_edge_embeddings(train_edges_sampled)
    # neg_train_edge_embs,neg_train_sim_matrix,neg_embs_1_train,neg_embs_2_train = get_edge_embeddings(train_edges_false, ego_user,DATASET,Flag, flag='neg-train')
    train_edge_embs = pos_train_edge_embs
    train_sim_matrix= pos_train_sim_matrix
    train_embs_1 = pos_embs_1_train
    train_embs_2 = pos_embs_2_train

    # Create train-set edge labels: 1 = real edge, 0 = false edge
    train_edge_labels = np.ones(len(train_edges_sampled))


    # Test-set edge embeddings, labels
    pos_test_edge_embs,pos_test_sim_matrix,pos_embs_1_test,pos_embs_2_test = get_edge_embeddings(test_edges)
    # neg_test_edge_embs ,neg_test_sim_matrix,neg_embs_1_test,neg_embs_2_test= get_edge_embeddings(test_edges_false,ego_user,DATASET,Flag, flag='neg-test')
    test_edge_embs = pos_test_edge_embs
    test_sim_matrix = pos_test_sim_matrix
    test_embs_1 = pos_embs_1_test
    test_embs_2 = pos_embs_2_test

    # Create val-set edge labels: 1 = real edge, 0 = false edge
    test_edge_labels = np.ones(len(test_edges))


    ###########sim_svm

    train_edges_list = np.array(train_edges_sampled)
    # print(train_edges_list)
    test_edges_list = test_edges
    # print(test_edges_list)

    edges_list = np.concatenate((train_edges_list, test_edges_list), axis=0)

    # print(type(train_edges_list))
    # print(type(test_edges_list))
    # print(type(edges_list))

    # print(np.shape(train_edges_list))
    # print(np.shape(test_edges_list))
    # print(np.shape(edges_list))

    ylabel = [1] * train_sim_matrix.shape[0] + [0] * test_sim_matrix.shape[0]

    # print(train_sim_matrix)
    # print(test_sim_matrix)

    sim_matrix = np.concatenate((train_sim_matrix, test_sim_matrix), axis=0)
    # print(sim_matrix)
    print(np.shape(train_sim_matrix))
    print(np.shape(test_sim_matrix))
    sim_matrix = sim_matrix.reshape(-1, 1)
    # print(sim_matrix)
    print(np.shape(sim_matrix))
    # exit()

    sim_matrix_train = train_sim_matrix
    sim_matrix_test = test_sim_matrix

    sim_matrix_train = sim_matrix_train.reshape(-1, 1)
    sim_matrix_test = sim_matrix_test.reshape(-1, 1)

    print(np.shape(sim_matrix_train))
    print(np.shape(sim_matrix_test))    # print(loss_matrix_test)
    from sklearn.model_selection import train_test_split

    ylabel1 = ylabel
    ylable1 = np.reshape(len(ylabel1), 1)

    # print((edges_list))
    # print((ylabel1))
    # print(np.shape(ylabel1))
    # print(np.shape(edges_list))
    y_label = np.zeros((np.shape(edges_list)[0], 3))
    for i in range(np.shape(edges_list)[0]):
        y_label[i][0] = edges_list[i][0]
        y_label[i][1] = edges_list[i][1]
        y_label[i][2] = ylabel[i]
    print(np.shape(y_label))

    y_label_train = np.zeros((np.shape(train_edges_list)[0], 3))
    for i in range(np.shape(train_edges_list)[0]):
        y_label_train[i][0] = train_edges_list[i][0]
        y_label_train[i][1] = train_edges_list[i][1]
        y_label_train[i][2] = 1
    print(np.shape(y_label_train))

    y_label_test = np.zeros((np.shape(test_edges_list)[0], 3))
    for i in range(np.shape(test_edges_list)[0]):
        y_label_test[i][0] = test_edges_list[i][0]
        y_label_test[i][1] = test_edges_list[i][1]
        y_label_test[i][2] = 0
    print(np.shape(y_label_test))

    X_train_train, X_train_test, y_train_train, y_train_test = train_test_split(sim_matrix_train, y_label_train,
                                                                                test_size=0.3, random_state=42)

    X_test_train, X_test_test, y_test_train, y_test_test = train_test_split(sim_matrix_test, y_label_test,
                                                                            test_size=0.3, random_state=42)

    X_train = np.concatenate((X_train_train, X_test_train), axis=0)
    X_test = np.concatenate((X_train_test, X_test_test), axis=0)
    y_train = np.concatenate((y_train_train, y_test_train), axis=0)
    y_test = np.concatenate((y_train_test, y_test_test), axis=0)

    from sklearn import metrics
    from sklearn.neural_network import MLPClassifier

    mlp = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(64, 32, 16, 18), random_state=1,
                        max_iter=500)

    mlp.fit(X_train, y_train[:, 2])

    loss=mlp.loss_
    print(loss)

    # y_score = mlp.predict(X_test)
    # ls = 0
    # for i in range(len(y_score)):
    #     if y_score[i] != y_test[i][2]:
    #         if y_score[i] == 1:
    #             ls += y_score[i]
    #         else:
    #             ls += 1-y_score[i]
    # loss = ls / len(y_score)
    # print(loss)

    # exit()

    # print("Training set score: %f" % mlp.score(X_train, y_train[:, 2]))
    # print("Test set score: %f" % mlp.score(X_test, y_test[:, 2]))
    #
    # y_score = mlp.predict(X_test)
    # print(metrics.f1_score(y_test[:, 2], y_score, average='micro'))
    # print(metrics.classification_report(y_test[:, 2], y_score, labels=range(3)))

    return loss

def discriminator_gain(train_edges_sampled,test_edges,num_nodes,word2idx,mlp,cnt_it,F, res_dir,embedding):
    idx2vec = embedding.ivectors.weight.data.cpu().numpy()
    # print(num_nodes)
    # print(word2idx)

    perm = np.array([word2idx[str(word)] for word in range(num_nodes)])
    emb = idx2vec[perm]

    # start_time = time.time()
    # Generate bootstrapped edge embeddings (as is done in node2vec paper)
    # Edge embedding for (v1, v2) = hadamard product of node embeddings for v1, v2
    emb_matrix=emb
    def get_edge_embeddings(edge_list):
        embs = []
        sim_matrix=[]
        embs_1=[]
        embs_2 = []
        for edge in edge_list:
            node1 = edge[0]
            node2 = edge[1]
            emb1 = emb_matrix[node1]
            #print(np.shape(emb1))
            emb2 = emb_matrix[node2]
            edge_emb = np.multiply(emb1, emb2)
            sim = np.dot(emb1, emb2)/(np.linalg.norm(emb1)*np.linalg.norm(emb2))
            #edge_emb = np.array(emb1) + np.array(emb2)
            # print(np.shape(edge_emb))
            embs.append(edge_emb)
            embs_1.append(emb1)
            embs_2.append(emb2)
            sim_matrix.append(sim)
        embs = np.array(embs)
        sim_matrix = np.array(sim_matrix)
        embs_1=np.array(embs_1)
        embs_2 =np.array(embs_2)

        # #with open('/Users/xiulingwang/Downloads/line-master/data/embds/' + str(ego_user) + flag + '-' + Flag, 'w') as f:
        # with open('E:/python/banlance/code/' + DATASET + '/' + METHOD + '/embeds/' + Flag + '-' + str(ego_user) + flag,'w') as f:
        # #with open('/Users/xiulingwang/Downloads/'+DATASET+'/line/3-split/' + str(ego_user)+ '-' + flag + '-' + Flag+'-' +'embds','w') as f:
        #     f.write('%d %d\n' % (edge_list.shape[0], args.representation_size))
        #     for i in range(edge_list.shape[0]):
        #         e = ' '.join(map(lambda x: str(x), embs[i]))
        #         f.write('%s %s %s\n' % (str(edge_list[i][0]), str(edge_list[i][1]), e))

        return embs,sim_matrix,embs_1,embs_2

    # edgeall = list([list(edge_tuple) for edge_tuple in train_edges])

    # train_edges_sampled = random.sample(edgeall, np.shape(test_edges)[0])
    # train_edges_sampled = random.sample(edgeall, np.shape(test_edges)[0])


    # Train-set edge embeddings
    pos_train_edge_embs ,pos_train_sim_matrix,pos_embs_1_train,pos_embs_2_train= get_edge_embeddings(train_edges_sampled)
    # neg_train_edge_embs,neg_train_sim_matrix,neg_embs_1_train,neg_embs_2_train = get_edge_embeddings(train_edges_false, ego_user,DATASET,Flag, flag='neg-train')
    train_edge_embs = pos_train_edge_embs
    train_sim_matrix= pos_train_sim_matrix
    train_embs_1 = pos_embs_1_train
    train_embs_2 = pos_embs_2_train

    # Create train-set edge labels: 1 = real edge, 0 = false edge
    train_edge_labels = np.ones(len(train_edges_sampled))


    # Test-set edge embeddings, labels
    pos_test_edge_embs,pos_test_sim_matrix,pos_embs_1_test,pos_embs_2_test = get_edge_embeddings(test_edges)
    # neg_test_edge_embs ,neg_test_sim_matrix,neg_embs_1_test,neg_embs_2_test= get_edge_embeddings(test_edges_false,ego_user,DATASET,Flag, flag='neg-test')
    test_edge_embs = pos_test_edge_embs
    test_sim_matrix = pos_test_sim_matrix
    test_embs_1 = pos_embs_1_test
    test_embs_2 = pos_embs_2_test

    # Create val-set edge labels: 1 = real edge, 0 = false edge
    test_edge_labels = np.ones(len(test_edges))


    ###########sim_svm

    train_edges_list = np.array(train_edges_sampled)
    # print(train_edges_list)
    test_edges_list = test_edges
    # print(test_edges_list)

    edges_list = np.concatenate((train_edges_list, test_edges_list), axis=0)

    # print(type(train_edges_list))
    # print(type(test_edges_list))
    # print(type(edges_list))

    # print(np.shape(train_edges_list))
    # print(np.shape(test_edges_list))
    # print(np.shape(edges_list))

    ylabel = [1] * train_sim_matrix.shape[0] + [0] * test_sim_matrix.shape[0]

    # print(train_sim_matrix)
    # print(test_sim_matrix)

    sim_matrix = np.concatenate((train_sim_matrix, test_sim_matrix), axis=0)
    # print(sim_matrix)
    # print(np.shape(train_sim_matrix))
    # print(np.shape(test_sim_matrix))
    sim_matrix = sim_matrix.reshape(-1, 1)
    # print(sim_matrix)
    # print(np.shape(sim_matrix))
    # exit()

    sim_matrix_train = train_sim_matrix
    sim_matrix_test = test_sim_matrix

    sim_matrix_train = sim_matrix_train.reshape(-1, 1)
    sim_matrix_test = sim_matrix_test.reshape(-1, 1)

    # print(np.shape(sim_matrix_train))
    # print(np.shape(sim_matrix_test))    # print(loss_matrix_test)
    from sklearn.model_selection import train_test_split

    ylabel1 = ylabel
    ylable1 = np.reshape(len(ylabel1), 1)

    # print((edges_list))
    # print((ylabel1))
    # print(np.shape(ylabel1))
    # print(np.shape(edges_list))
    y_label = np.zeros((np.shape(edges_list)[0], 3))
    for i in range(np.shape(edges_list)[0]):
        y_label[i][0] = edges_list[i][0]
        y_label[i][1] = edges_list[i][1]
        y_label[i][2] = ylabel[i]
    # print(np.shape(y_label))

    y_label_train = np.zeros((np.shape(train_edges_list)[0], 3))
    for i in range(np.shape(train_edges_list)[0]):
        y_label_train[i][0] = train_edges_list[i][0]
        y_label_train[i][1] = train_edges_list[i][1]
        y_label_train[i][2] = 1
    # print(np.shape(y_label_train))

    y_label_test = np.zeros((np.shape(test_edges_list)[0], 3))
    for i in range(np.shape(test_edges_list)[0]):
        y_label_test[i][0] = test_edges_list[i][0]
        y_label_test[i][1] = test_edges_list[i][1]
        y_label_test[i][2] = 0
    # print(np.shape(y_label_test))

    X_train_train, X_train_test, y_train_train, y_train_test = train_test_split(sim_matrix_train, y_label_train,
                                                                                test_size=0.3, random_state=42)

    X_test_train, X_test_test, y_test_train, y_test_test = train_test_split(sim_matrix_test, y_label_test,
                                                                            test_size=0.3, random_state=42)

    X_train = np.concatenate((X_train_train, X_test_train), axis=0)
    X_test = np.concatenate((X_train_test, X_test_test), axis=0)
    y_train = np.concatenate((y_train_train, y_test_train), axis=0)
    y_test = np.concatenate((y_train_test, y_test_test), axis=0)

    from sklearn.metrics import accuracy_score
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.svm import SVC

    # mlp = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(64, 32, 16, 18), random_state=1,
    #                     max_iter=500)
    #
    # mlp.fit(X_train, y_train[:, 2])

    svm = OneVsRestClassifier(SVC())
    svm.fit(X_train, y_train[:, 2])
    gain=0
    cnt_true=0

    pree = svm.predict(X_test)
    # print('prob')
    # print(prob)
    # print('pree')
    # print(pree)
    for i in range(len(pree)):
        if i <np.shape(X_train_test)[0]:
            if pree[i]==1:
                cnt_true+=1
            # gain+=np.log(prob[i][1])

        else:
            if pree[i] == 0:
                cnt_true += 1
            # gain += np.log(prob[i][0])
    # gain=gain/np.shape(X_train)[0]


    acc=cnt_true/np.shape(X_test)[0]
    gain =acc
    print('acc',acc)

    acc_sim = accuracy_score(pree, y_test[:, 2])
    print(acc_sim)
    # exit()
    # print('gain')
    # print(gain)

    # y_score = mlp.predict(X_train)
    # print(y_score)
    #
    # exit()
    if cnt_it==0:
        acc_kmeans_sim, acc_mlp_sim, acc_rf_sim, acc_svm_sim=discriminator_gain2(train_edges_sampled, test_edges, num_nodes, word2idx, mlp, F, res_dir, idx2vec)
        return gain, acc_kmeans_sim, acc_mlp_sim, acc_rf_sim, acc_svm_sim
    else:
        return gain



    # y_score = mlp.predict(X_test)
    # ls = 0
    # for i in range(len(y_score)):
    #     if y_score[i] != y_test[i][2]:
    #         if y_score[i] == 1:
    #             ls += y_score[i]
    #         else:
    #             ls += 1-y_score[i]
    # loss = ls / len(y_score)
    # print(loss)

    # exit()

    # print("Training set score: %f" % mlp.score(X_train, y_train[:, 2]))
    # print("Test set score: %f" % mlp.score(X_test, y_test[:, 2]))
    #
    # y_score = mlp.predict(X_test)
    # print(metrics.f1_score(y_test[:, 2], y_score, average='micro'))
    # print(metrics.classification_report(y_test[:, 2], y_score, labels=range(3)))



def discriminator_gain2(train_edges_sampled,test_edges,num_nodes,word2idx,mlp,F, res_dir,idx2vec):
    # idx2vec = embedding.ivectors.weight.data.cpu().numpy()
    # print(num_nodes)
    # print(word2idx)

    perm = np.array([word2idx[str(word)] for word in range(num_nodes)])
    emb = idx2vec[perm]

    # start_time = time.time()
    # Generate bootstrapped edge embeddings (as is done in node2vec paper)
    # Edge embedding for (v1, v2) = hadamard product of node embeddings for v1, v2
    emb_matrix=emb
    def get_edge_embeddings(edge_list):
        embs = []
        sim_matrix=[]
        embs_1=[]
        embs_2 = []
        for edge in edge_list:
            node1 = edge[0]
            node2 = edge[1]
            emb1 = emb_matrix[node1]
            #print(np.shape(emb1))
            emb2 = emb_matrix[node2]
            edge_emb = np.multiply(emb1, emb2)
            sim = np.dot(emb1, emb2)/(np.linalg.norm(emb1)*np.linalg.norm(emb2))
            #edge_emb = np.array(emb1) + np.array(emb2)
            # print(np.shape(edge_emb))
            embs.append(edge_emb)
            embs_1.append(emb1)
            embs_2.append(emb2)
            sim_matrix.append(sim)
        embs = np.array(embs)
        sim_matrix = np.array(sim_matrix)
        embs_1=np.array(embs_1)
        embs_2 =np.array(embs_2)

        # #with open('/Users/xiulingwang/Downloads/line-master/data/embds/' + str(ego_user) + flag + '-' + Flag, 'w') as f:
        # with open('E:/python/banlance/code/' + DATASET + '/' + METHOD + '/embeds/' + Flag + '-' + str(ego_user) + flag,'w') as f:
        # #with open('/Users/xiulingwang/Downloads/'+DATASET+'/line/3-split/' + str(ego_user)+ '-' + flag + '-' + Flag+'-' +'embds','w') as f:
        #     f.write('%d %d\n' % (edge_list.shape[0], args.representation_size))
        #     for i in range(edge_list.shape[0]):
        #         e = ' '.join(map(lambda x: str(x), embs[i]))
        #         f.write('%s %s %s\n' % (str(edge_list[i][0]), str(edge_list[i][1]), e))

        return embs,sim_matrix,embs_1,embs_2

    # edgeall = list([list(edge_tuple) for edge_tuple in train_edges])
    #
    # # train_edges_sampled = random.sample(edgeall, np.shape(test_edges)[0])
    # train_edges_sampled = random.sample(edgeall, np.shape(test_edges)[0])


    # Train-set edge embeddings
    pos_train_edge_embs ,pos_train_sim_matrix,pos_embs_1_train,pos_embs_2_train= get_edge_embeddings(train_edges_sampled)
    # neg_train_edge_embs,neg_train_sim_matrix,neg_embs_1_train,neg_embs_2_train = get_edge_embeddings(train_edges_false, ego_user,DATASET,Flag, flag='neg-train')
    train_edge_embs = pos_train_edge_embs
    train_sim_matrix= pos_train_sim_matrix
    train_embs_1 = pos_embs_1_train
    train_embs_2 = pos_embs_2_train

    # Create train-set edge labels: 1 = real edge, 0 = false edge
    train_edge_labels = np.ones(len(train_edges_sampled))


    # Test-set edge embeddings, labels
    pos_test_edge_embs,pos_test_sim_matrix,pos_embs_1_test,pos_embs_2_test = get_edge_embeddings(test_edges)
    # neg_test_edge_embs ,neg_test_sim_matrix,neg_embs_1_test,neg_embs_2_test= get_edge_embeddings(test_edges_false,ego_user,DATASET,Flag, flag='neg-test')
    test_edge_embs = pos_test_edge_embs
    test_sim_matrix = pos_test_sim_matrix
    test_embs_1 = pos_embs_1_test
    test_embs_2 = pos_embs_2_test

    # Create val-set edge labels: 1 = real edge, 0 = false edge
    test_edge_labels = np.ones(len(test_edges))


    ###########sim_svm

    train_edges_list = np.array(train_edges_sampled)
    # print(train_edges_list)
    test_edges_list = test_edges
    # print(test_edges_list)

    edges_list = np.concatenate((train_edges_list, test_edges_list), axis=0)

    # print(type(train_edges_list))
    # print(type(test_edges_list))
    # print(type(edges_list))

    # print(np.shape(train_edges_list))
    # print(np.shape(test_edges_list))
    # print(np.shape(edges_list))

    ylabel = [1] * train_sim_matrix.shape[0] + [0] * test_sim_matrix.shape[0]

    # print(train_sim_matrix)
    # print(test_sim_matrix)

    sim_matrix = np.concatenate((train_sim_matrix, test_sim_matrix), axis=0)
    # print(sim_matrix)
    # print(np.shape(train_sim_matrix))
    # print(np.shape(test_sim_matrix))
    sim_matrix = sim_matrix.reshape(-1, 1)
    # print(sim_matrix)
    # print(np.shape(sim_matrix))
    # exit()

    sim_matrix_train = train_sim_matrix
    sim_matrix_test = test_sim_matrix

    sim_matrix_train = sim_matrix_train.reshape(-1, 1)
    sim_matrix_test = sim_matrix_test.reshape(-1, 1)

    # print(np.shape(sim_matrix_train))
    # print(np.shape(sim_matrix_test))    # print(loss_matrix_test)
    from sklearn.model_selection import train_test_split

    ylabel1 = ylabel
    ylable1 = np.reshape(len(ylabel1), 1)

    # print((edges_list))
    # print((ylabel1))
    # print(np.shape(ylabel1))
    # print(np.shape(edges_list))
    y_label = np.zeros((np.shape(edges_list)[0], 3))
    for i in range(np.shape(edges_list)[0]):
        y_label[i][0] = edges_list[i][0]
        y_label[i][1] = edges_list[i][1]
        y_label[i][2] = ylabel[i]
    # print(np.shape(y_label))

    y_label_train = np.zeros((np.shape(train_edges_list)[0], 3))
    for i in range(np.shape(train_edges_list)[0]):
        y_label_train[i][0] = train_edges_list[i][0]
        y_label_train[i][1] = train_edges_list[i][1]
        y_label_train[i][2] = 1
    # print(np.shape(y_label_train))

    y_label_test = np.zeros((np.shape(test_edges_list)[0], 3))
    for i in range(np.shape(test_edges_list)[0]):
        y_label_test[i][0] = test_edges_list[i][0]
        y_label_test[i][1] = test_edges_list[i][1]
        y_label_test[i][2] = 0
    # print(np.shape(y_label_test))

    X_train_train, X_train_test, y_train_train, y_train_test = train_test_split(sim_matrix_train, y_label_train,
                                                                                test_size=0.3, random_state=42)

    X_test_train, X_test_test, y_test_train, y_test_test = train_test_split(sim_matrix_test, y_label_test,
                                                                            test_size=0.3, random_state=42)

    X_train = np.concatenate((X_train_train, X_test_train), axis=0)
    X_test = np.concatenate((X_train_test, X_test_test), axis=0)
    y_train = np.concatenate((y_train_train, y_test_train), axis=0)
    y_test = np.concatenate((y_train_test, y_test_test), axis=0)

    from sklearn.cluster import KMeans
    from sklearn.metrics import accuracy_score

    accuracy = []
    for i in range(500):
        kmeans = KMeans(n_clusters=2, random_state=i).fit(sim_matrix)
        # kmeans = KMeans(n_clusters=2, random_state=i).fit(X)
        ylabel = [1] * train_sim_matrix.shape[0] + [0] * test_sim_matrix.shape[0]
        acc = accuracy_score(kmeans.labels_, ylabel)
        accuracy.append(acc)

    acc_kmeans_sim = max(accuracy)

    tsts=[]
    print(len(kmeans.labels_))
    for i in range(len(kmeans.labels_)):
        node1=edges_list[i][0]
        node2=edges_list[i][1]
        # dgr1=g.degree(node1)
        # dgr2 = g.degree(node2)
        # gender1 = g.nodes[node1]['gender']
        # gender2 = g.nodes[node2]['gender']

        sim0 = sim_matrix[i]
        # print(sim0)
        # exit()

        # if (node1, node2) in g.edges():
        #     edge_betw = edge_between[(node1, node2)]
        # else:
        #     edge_betw = 0

        tst = [kmeans.labels_[i], ylabel[i], node1, node2]
        tsts.append(tst)
    name = ['y_score', 'y_test_grd', 'node1', 'node2']
    result = pd.DataFrame(columns=name, data=tsts)
    result.to_csv("{}{}-kmeans_sim.csv".format(res_dir, F))





    from sklearn import metrics
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import accuracy_score
    mlp = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(64, 32, 16, 18), random_state=1,
                        max_iter=500)

    mlp.fit(X_train, y_train[:, 2])
    pree = mlp.predict(X_train)
    # print('prob')
    # print(prob)
    # print('pree')
    # print(pree)
    print("Training set score: %f" % mlp.score(X_train, y_train[:, 2]))
    print("Test set score: %f" % mlp.score(X_test, y_test[:, 2]))

    y_score = mlp.predict(X_test)
    print(metrics.f1_score(y_test[:, 2], y_score, average='micro'))
    print(metrics.classification_report(y_test[:, 2], y_score, labels=range(3)))

    acc_mlp_sim = accuracy_score(y_score, y_test[:, 2])

    tsts = []
    for i in range(len(y_score)):
        node1 = y_test[i][0]
        node2 = y_test[i][1]
        # dgr1 = g.degree(node1)
        # dgr2 = g.degree(node2)
        #
        # gender1 = g.nodes[node1]['gender']
        # gender2 = g.nodes[node2]['gender']
        #
        # if (node1, node2) in g.edges():
        #     edge_betw = edge_between[(node1, node2)]
        # else:
        #     edge_betw = 0

        tst = [y_score[i], y_test[i][2], y_test[i][0], y_test[i][1]]
        tsts.append(tst)
    name = ['y_score', 'y_test_grd', 'node1', 'node2']
    result = pd.DataFrame(columns=name, data=tsts)
    result.to_csv("{}{}-mlp_sim.csv".format(res_dir, F))

    from sklearn.ensemble import RandomForestClassifier

    rf = RandomForestClassifier(max_depth=150, random_state=0)
    rf.fit(X_train, y_train[:, 2])

    print("Training set score: %f" % rf.score(X_train, y_train[:, 2]))
    print("Test set score: %f" % rf.score(X_test, y_test[:, 2]))

    y_score = rf.predict(X_test)
    print(metrics.f1_score(y_test[:, 2], y_score, average='micro'))
    print(metrics.classification_report(y_test[:, 2], y_score, labels=range(3)))

    acc_rf_sim = accuracy_score(y_score, y_test[:, 2])

    tsts = []
    for i in range(len(y_score)):
        node1 = y_test[i][0]
        node2 = y_test[i][1]
        # dgr1 = g.degree(node1)
        # dgr2 = g.degree(node2)
        #
        # gender1 = g.nodes[node1]['gender']
        # gender2 = g.nodes[node2]['gender']

        # if (node1, node2) in g.edges():
        #     edge_betw = edge_between[(node1, node2)]
        # else:
        #     edge_betw = 0

        tst = [y_score[i], y_test[i][2], y_test[i][0], y_test[i][1]]
        tsts.append(tst)
    name = ['y_score', 'y_test_grd', 'node1', 'node2']

    result = pd.DataFrame(columns=name, data=tsts)
    result.to_csv("{}{}-rf_sim.csv".format(res_dir, F))

    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.svm import SVC

    svm = OneVsRestClassifier(SVC())
    svm.fit(X_train, y_train[:, 2])

    print("Training set score: %f" % svm.score(X_train, y_train[:, 2]))
    print("Test set score: %f" % svm.score(X_test, y_test[:, 2]))

    y_score = svm.predict(X_test)
    print(metrics.f1_score(y_test[:, 2], y_score, average='micro'))
    print(metrics.classification_report(y_test[:, 2], y_score, labels=range(3)))

    acc_svm_sim = accuracy_score(y_score, y_test[:, 2])

    tsts = []
    for i in range(len(y_score)):
        node1 = y_test[i][0]
        node2 = y_test[i][1]
        # dgr1 = g.degree(node1)
        # dgr2 = g.degree(node2)
        # gender1 = g.nodes[node1]['gender']
        # gender2 = g.nodes[node2]['gender']
        #
        # if (node1, node2) in g.edges():
        #     edge_betw = edge_between[(node1, node2)]
        # else:
        #     edge_betw = 0

        tst = [y_score[i], y_test[i][2], y_test[i][0], y_test[i][1]]
        tsts.append(tst)
    name = ['y_score', 'y_test_grd', 'node1', 'node2']
    result = pd.DataFrame(columns=name, data=tsts)
    result.to_csv("{}{}-svm_sim.csv".format(res_dir, F))

    print(acc_kmeans_sim, acc_mlp_sim, acc_rf_sim, acc_svm_sim)

    return (acc_kmeans_sim, acc_mlp_sim, acc_rf_sim, acc_svm_sim)




if __name__ == "__main__":
        data = np.array(np.random.randint(0,13210, size=(13210, 80)),str)
        w2v = ModWord2Vec(data)
        w2v.save_emb("embedding.npy",13210)
