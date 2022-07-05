import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from word2vec.data_reader import DataReader, Word2vecDataset
from word2vec.model import SkipGramModel


class Word2VecTrainer:
    def __init__(self, input_file, output_file, emb_dimension=100, batch_size=32, window_size=5, iterations=5,
                 initial_lr=0.001, min_count=12):

        self.data = DataReader(input_file, min_count)
        dataset = Word2vecDataset(self.data, window_size)
        self.dataloader = DataLoader(dataset, batch_size=batch_size,
                                     shuffle=False, num_workers=0, collate_fn=dataset.collate)

        self.output_file_name = output_file
        self.emb_size = len(self.data.word2id)
        self.emb_dimension = emb_dimension
        self.batch_size = batch_size
        self.iterations = iterations
        self.initial_lr = initial_lr
        self.skip_gram_model = SkipGramModel(self.emb_size, self.emb_dimension)

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        if self.use_cuda:
            self.skip_gram_model.cuda()

    def train(self):

        for iteration in range(self.iterations):

            print("\n\n\nIteration: " + str(iteration + 1))
            optimizer = optim.SparseAdam(self.skip_gram_model.parameters(), lr=self.initial_lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(self.dataloader))

            running_loss = 0.0
            for i, sample_batched in enumerate(tqdm(self.dataloader)):
                #print('!!!!!!!!')
                #print(sample_batched)

                if len(sample_batched[0]) > 1:
                    pos_u = sample_batched[0].to(self.device)
                    pos_v = sample_batched[1].to(self.device)
                    neg_v = sample_batched[2].to(self.device)
                    #print('*****')
                    print((pos_u))
                    #print(np.shape(pos_v)[0])
                    print((neg_v))
                    print(np.shape(pos_u)[0])
                    print(np.shape(pos_v)[0])
                    print(np.shape(neg_v)[0])
                    exit()

                    scheduler.step()
                    optimizer.zero_grad()
                    loss,emb_mappings = self.skip_gram_model.forward(pos_u, pos_v, neg_v)
                    loss.backward()
                    optimizer.step()

                    running_loss = running_loss * 0.9 + loss.item() * 0.1
                    if i > 0 and i % 500 == 0:
                        print(" Loss: " + str(running_loss))

            self.skip_gram_model.save_embedding(self.data.id2word, self.output_file_name)
            return emb_mappings



    def train_dp(self):

        for iteration in range(self.iterations):

            print("\n\n\nIteration: " + str(iteration + 1))
            optimizer = optim.Adam(self.skip_gram_model.parameters(), lr=self.initial_lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(self.dataloader))
            print(len(self.dataloader))

            C = 1
            sigma = 10467#lastfm

            running_loss = 0.0
            for i, sample_batched in enumerate(tqdm(self.dataloader)):
                #print('!!!!!!!!')

                if len(sample_batched[0]) > 1:
                    pos_u = sample_batched[0].to(self.device)
                    pos_v = sample_batched[1].to(self.device)
                    neg_v = sample_batched[2].to(self.device)
                    #print('*****')
                    # print(pos_u)
                    # print(pos_v)
                    # print(neg_v)
                    # print(np.shape(pos_u))
                    # print(np.shape(pos_v))
                    # print(np.shape(neg_v))
                    # exit()


                    scheduler.step()
                    optimizer.zero_grad()
                    loss,emb_mappings = self.skip_gram_model.forward(pos_u, pos_v, neg_v)

                    grads = [torch.zeros(p.shape).to(self.device) for p in self.skip_gram_model.parameters()]

                    igrad = torch.autograd.grad(loss, self.skip_gram_model.parameters(), retain_graph=True)

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
                        grads[i] += sigma * C * (torch.randn_like(grads[i]).to(self.device))
                        grads[i] /= np.shape(pos_u)[0]+np.shape(neg_v)[0]

                    p_list = [p for p in self.skip_gram_model.parameters()]
                    for i in range(len(p_list)):

                        p_list[i].grad = grads[i]
                        #p_list[i].grad.detach_()

                    print(p_list)

                    loss.backward()
                    optimizer.step()

                    running_loss = running_loss * 0.9 + loss.item() * 0.1
                    if i > 0 and i % 500 == 0:
                        print(" Loss: " + str(running_loss))

            self.skip_gram_model.save_embedding(self.data.id2word, self.output_file_name)
            return emb_mappings





if __name__ == '__main__':
    w2v = Word2VecTrainer(input_file="input.txt", output_file="out.vec")
    w2v.train()
