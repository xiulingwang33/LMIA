# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import tensorflow as tf
import numpy as np
import argparse
from model import LINEModel
from utils import DBLPDataLoader,DBLPDataLoader1
import pickle
import time
import random
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
import pandas as pd
import copy

def LINE(g_train, train_test_split,graph_file,DATASET,METHOD,ego_user, F):
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_dim', default=128)
    parser.add_argument('--batch_size', default=128)
    parser.add_argument('--K', default=5)
    parser.add_argument('--proximity', default='second-order', help='first-order or second-order')
    parser.add_argument('--learning_rate', default=0.025)
    parser.add_argument('--mode', default='train')
    parser.add_argument('--num_batches', default=2000)
    parser.add_argument('--total_graph', default=True)
    parser.add_argument('--graph_file', default='/Users/xiulingwang/Downloads/line-master/data/0-adj-feat.pkl')
    parser.add_argument('--edge_score_mode', default='edge-emb')
    parser.add_argument('--uid', default='0')
    parser.add_argument('--flag', default='weighted')
    args = parser.parse_args()
    #args.proximity='first-order'
    args.graph_file=graph_file
    args.uid = str(ego_user)
    args.flag=str(F)
    print(args.graph_file)
    if args.mode == 'train':
        normalized_embedding=train(args)
        data_loader = DBLPDataLoader(graph_file=args.graph_file)
        emb_list = []
        print(np.shape(g_train)[0])
        for node_index in range(0, np.shape(g_train)[0]):
            node_str = str(node_index)
            node_emb = normalized_embedding[node_index]
            emb_list.append(node_emb)
        emb_matrix = np.vstack(emb_list)
        print(emb_list)
        print(np.shape(emb_list))

        with open('/Users/xiulingwang/Downloads/' + DATASET + '/' + METHOD + '/embeds/' + F + '-' + str(ego_user),
                  'w') as f:
            f.write('%d %d\n' % (np.shape(g_train)[0], args.embedding_dim))
            for i in range(np.shape(g_train)[0]):
                e = ' '.join(map(lambda x: str(x), emb_list[i]))
                f.write('%s %s\n' % (str(i), e))

        # with open('/Users/xiulingwang/Downloads/' + DATASET + '/' + METHOD + '/embeds/' + F + '-' + str(ego_user),'w') as f:
        #     pickle.dump(data_loader.embedding_mapping(normalized_embedding), f)
        # print(args.graph_file)

        n2v_scores, val_edge_labels, val_preds, test_edge_labels, test_preds=linkpre_scores(args, emb_matrix, train_test_split, ego_user,DATASET,METHOD, F)
        return n2v_scores, val_edge_labels, val_preds, test_edge_labels, test_preds



    elif args.mode == 'test':
        test(args)




def train_adj_defense(args,sigma):
    data_loader = DBLPDataLoader1(graph_file=args.graph_file,b=sigma)
    suffix = args.proximity
    args.num_of_nodes = data_loader.num_of_nodes
    model = LINEModel(args)
    with tf.Session() as sess:
        print(args)
        print('batches\tloss\tsampling time\ttraining_time\tdatetime')
        tf.global_variables_initializer().run()
        initial_embedding = sess.run(model.embedding)
        learning_rate = args.learning_rate
        sampling_time, training_time = 0, 0
        for b in range(args.num_batches):
            t1 = time.time()
            u_i, u_j, label = data_loader.fetch_batch(batch_size=args.batch_size, K=args.K)
            feed_dict = {model.u_i: u_i, model.u_j: u_j, model.label: label, model.learning_rate: learning_rate}
            t2 = time.time()
            sampling_time += t2 - t1
            if b % 100 != 0:
                sess.run(model.train_op, feed_dict=feed_dict)
                training_time += time.time() - t2
                if learning_rate > args.learning_rate * 0.0001:
                    learning_rate = args.learning_rate * (1 - b / args.num_batches)
                else:
                    learning_rate = args.learning_rate * 0.0001
            else:
                loss = sess.run(model.loss, feed_dict=feed_dict)
                print('%d\t%f\t%0.2f\t%0.2f\t%s' % (b, loss, sampling_time, training_time,
                                                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
                sampling_time, training_time = 0, 0
            if b % 1000 == 0 or b == (args.num_batches - 1):
                embedding = sess.run(model.embedding)

                normalized_embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
                pickle.dump(data_loader.embedding_mapping(normalized_embedding),
                            open('data/embedding_%s_%s.pkl' % (args.uid,args.flag), 'wb'))
    return (embedding)



def train(args):
    data_loader = DBLPDataLoader(graph_file=args.graph_file)
    suffix = args.proximity
    args.num_of_nodes = data_loader.num_of_nodes
    model = LINEModel(args)
    with tf.Session() as sess:
        print(args)
        print('batches\tloss\tsampling time\ttraining_time\tdatetime')
        tf.global_variables_initializer().run()
        initial_embedding = sess.run(model.embedding)
        learning_rate = args.learning_rate
        sampling_time, training_time = 0, 0
        for b in range(args.num_batches):
            t1 = time.time()
            u_i, u_j, label = data_loader.fetch_batch(batch_size=args.batch_size, K=args.K)
            feed_dict = {model.u_i: u_i, model.u_j: u_j, model.label: label, model.learning_rate: learning_rate}
            t2 = time.time()
            sampling_time += t2 - t1
            if b % 100 != 0:
                sess.run(model.train_op, feed_dict=feed_dict)
                training_time += time.time() - t2
                if learning_rate > args.learning_rate * 0.0001:
                    learning_rate = args.learning_rate * (1 - b / args.num_batches)
                else:
                    learning_rate = args.learning_rate * 0.0001
            else:
                loss = sess.run(model.loss, feed_dict=feed_dict)
                print('%d\t%f\t%0.2f\t%0.2f\t%s' % (b, loss, sampling_time, training_time,
                                                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
                sampling_time, training_time = 0, 0
            if b % 1000 == 0 or b == (args.num_batches - 1):
                embedding = sess.run(model.embedding)

                normalized_embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
                pickle.dump(data_loader.embedding_mapping(normalized_embedding),
                            open('data/embedding_%s_%s.pkl' % (args.uid,args.flag), 'wb'))
    return (embedding)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def linkpre_scores(args, emb_matrix, train_test_split, ego_user,DATASET,METHOD, Flag):
    adj_train, train_edges, train_edges_false, val_edges, val_edges_false, \
    test_edges, test_edges_false = train_test_split
    start_time = time.time()
    # Generate bootstrapped edge embeddings (as is done in node2vec paper)
    # Edge embedding for (v1, v2) = hadamard product of node embeddings for v1, v2
    if args.edge_score_mode == "edge-emb":

        def get_edge_embeddings(edge_list,ego_user,DATASET, Flag, flag):
            embs = []
            for edge in edge_list:
                node1 = edge[0]
                node2 = edge[1]
                emb1 = emb_matrix[node1]
                emb2 = emb_matrix[node2]
                edge_emb = np.multiply(emb1, emb2)
                #edge_emb = np.array(emb1) + np.array(emb2)
                embs.append(list(edge_emb))
            embs = np.array(embs)

            #with open('/Users/xiulingwang/Downloads/line-master/data/embds/' + str(ego_user) + flag + '-' + Flag, 'w') as f:
            #with open('/Users/xiulingwang/Downloads/'+DATASET+'/line/3-split/' + str(ego_user)+ '-' + flag + '-' + Flag+'-' +'embds','w') as f:
            #with open('/Users/xiulingwang/Downloads/line-master/data/embds/' + str(ego_user) + flag + '-' + Flag, 'w') as f:
            with open('./' + DATASET + '/' + METHOD + '/embeds/' + Flag + '-' + str(ego_user) + flag, 'w') as f:
                f.write('%d %d\n' % (edge_list.shape[0], args.embedding_dim))
                for i in range(edge_list.shape[0]):
                    e = ' '.join(map(lambda x: str(x), embs[i]))
                    f.write('%s %s %s\n' % (str(edge_list[i][0]), str(edge_list[i][1]), e))

            return embs

        # Train-set edge embeddings
        pos_train_edge_embs = get_edge_embeddings(train_edges,ego_user,DATASET, Flag, flag='pos-train')
        neg_train_edge_embs = get_edge_embeddings(train_edges_false, ego_user,DATASET,Flag, flag='neg-train')
        train_edge_embs = np.concatenate([pos_train_edge_embs, neg_train_edge_embs])

        # Create train-set edge labels: 1 = real edge, 0 = false edge
        train_edge_labels = np.concatenate([np.ones(len(train_edges)), np.zeros(len(train_edges_false))])

        # Val-set edge embeddings, labels
        if len(val_edges) > 0 and len(val_edges_false) > 0:
            pos_val_edge_embs = get_edge_embeddings(val_edges,ego_user,DATASET,Flag, flag='pos-val')
            neg_val_edge_embs = get_edge_embeddings(val_edges_false,ego_user,DATASET,Flag, flag='neg-val')
            val_edge_embs = np.concatenate([pos_val_edge_embs, neg_val_edge_embs])
            val_edge_labels = np.concatenate([np.ones(len(val_edges)), np.zeros(len(val_edges_false))])

        # Test-set edge embeddings, labels
        pos_test_edge_embs = get_edge_embeddings(test_edges,ego_user,DATASET,Flag, flag='pos-test')
        neg_test_edge_embs = get_edge_embeddings(test_edges_false,ego_user,DATASET,Flag, flag='neg-test')
        test_edge_embs = np.concatenate([pos_test_edge_embs, neg_test_edge_embs])

        # Create val-set edge labels: 1 = real edge, 0 = false edge
        test_edge_labels = np.concatenate([np.ones(len(test_edges)), np.zeros(len(test_edges_false))])

        # Train logistic regression classifier on train-set edge embeddings
        edge_classifier = LogisticRegression(random_state=0)
        edge_classifier.fit(train_edge_embs, train_edge_labels)

        # Predicted edge scores: probability of being of class "1" (real edge)
        if len(val_edges) > 0 and len(val_edges_false) > 0:
            val_preds = edge_classifier.predict_proba(val_edge_embs)[:, 1]
        test_preds = edge_classifier.predict_proba(test_edge_embs)[:, 1]
        print(test_preds)
        print(np.shape(test_preds))

        runtime = time.time() - start_time

        # Calculate scores
        if len(val_edges) > 0 and len(val_edges_false) > 0:
            n2v_val_roc = roc_auc_score(val_edge_labels, val_preds)
            # n2v_val_roc_curve = roc_curve(val_edge_labels, val_preds)
            n2v_val_ap = average_precision_score(val_edge_labels, val_preds)
        else:
            n2v_val_roc = None
            n2v_val_roc_curve = None
            n2v_val_ap = None

        n2v_test_roc = roc_auc_score(test_edge_labels, test_preds)
        # n2v_test_roc_curve = roc_curve(test_edge_labels, test_preds)
        n2v_test_ap = average_precision_score(test_edge_labels, test_preds)


    # Generate edge scores using simple dot product of node embeddings (like in GAE paper)
    elif args.edge_score_mode == "dot-product":
        score_matrix = np.dot(emb_matrix, emb_matrix.T)
        runtime = time.time() - start_time

        # Val set scores
        if len(val_edges) > 0:
            n2v_val_roc, n2v_val_ap = get_roc_score(val_edges, val_edges_false, score_matrix, apply_sigmoid=True)
        else:
            n2v_val_roc = None
            n2v_val_roc_curve = None
            n2v_val_ap = None

        # Test set scores
        n2v_test_roc, n2v_test_ap = get_roc_score(test_edges, test_edges_false, score_matrix, apply_sigmoid=True)

    else:
        print
        "Invalid edge_score_mode! Either use edge-emb or dot-product."

    # Record scores
    n2v_scores = {}

    n2v_scores['test_roc'] = n2v_test_roc
    # n2v_scores['test_roc_curve'] = n2v_test_roc_curve
    n2v_scores['test_ap'] = n2v_test_ap

    n2v_scores['val_roc'] = n2v_val_roc
    # n2v_scores['val_roc_curve'] = n2v_val_roc_curve
    n2v_scores['val_ap'] = n2v_val_ap

    n2v_scores['runtime'] = runtime

    return n2v_scores, val_edge_labels, val_preds, test_edge_labels, test_preds


# Input: positive test/val edges, negative test/val edges, edge score matrix
# Output: ROC AUC score, ROC Curve (FPR, TPR, Thresholds), AP score
def get_roc_score(edges_pos, edges_neg, score_matrix, apply_sigmoid=False):
    # Edge case
    if len(edges_pos) == 0 or len(edges_neg) == 0:
        return (None, None, None)

    # Store positive edge predictions, actual values
    preds_pos = []
    pos = []
    for edge in edges_pos:
        if apply_sigmoid == True:
            preds_pos.append(sigmoid(score_matrix[edge[0], edge[1]]))
        else:
            preds_pos.append(score_matrix[edge[0], edge[1]])
        pos.append(1)  # actual value (1 for positive)

    # Store negative edge predictions, actual values
    preds_neg = []
    neg = []
    for edge in edges_neg:
        if apply_sigmoid == True:
            preds_neg.append(sigmoid(score_matrix[edge[0], edge[1]]))
        else:
            preds_neg.append(score_matrix[edge[0], edge[1]])
        neg.append(0)  # actual value (0 for negative)

    # Calculate scores
    preds_all = np.hstack([preds_pos, preds_neg])
    labels_all = np.hstack([np.ones(len(preds_pos)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    # roc_curve_tuple = roc_curve(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    # return roc_score, roc_curve_tuple, ap_score
    return roc_score, ap_score


# Return a list of tuples (node1, node2) for networkx link prediction evaluation
def get_ebunch(train_test_split):
    adj_train, train_edges, train_edges_false, val_edges, val_edges_false, \
    test_edges, test_edges_false = train_test_split

    test_edges_list = test_edges.tolist()  # convert to nested list
    test_edges_list = [tuple(node_pair) for node_pair in test_edges_list]  # convert node-pairs to tuples
    test_edges_false_list = test_edges_false.tolist()
    test_edges_false_list = [tuple(node_pair) for node_pair in test_edges_false_list]
    return (test_edges_list + test_edges_false_list)


def test(args):
    pass

#if __name__ == '__main__':
    #main()


def LINE1(g_train, train_test_split,graph_file,DATASET,METHOD,ego_user, F):
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_dim', default=128)
    parser.add_argument('--batch_size', default=128)
    parser.add_argument('--K', default=5)
    parser.add_argument('--proximity', default='second-order', help='first-order or second-order')
    parser.add_argument('--learning_rate', default=0.025)
    parser.add_argument('--mode', default='train')
    parser.add_argument('--num_batches', default=8000)
    parser.add_argument('--total_graph', default=True)
    parser.add_argument('--graph_file', default='/Users/xiulingwang/Downloads/line-master/data/0-adj-feat.pkl')
    parser.add_argument('--edge_score_mode', default='edge-emb')
    parser.add_argument('--uid', default='0')
    parser.add_argument('--flag', default='weighted')
    args = parser.parse_args()
    #args.proximity='first-order'
    args.graph_file=graph_file
    args.uid = str(ego_user)
    args.flag=str(F)
    print(args.graph_file)
    if args.mode == 'train':
        normalized_embedding=train(args)
        data_loader = DBLPDataLoader(graph_file=args.graph_file)
        emb_list = []
        print(np.shape(g_train)[0])
        for node_index in range(0, np.shape(g_train)[0]):
            node_str = str(node_index)
            node_emb = normalized_embedding[node_index]
            emb_list.append(node_emb)
        emb_matrix = np.vstack(emb_list)
        print(emb_list)
        print(np.shape(emb_list))

        with open('E:\\python\\banlance\\code\\'+DATASET+'\\'+METHOD+'-embeds-'+F+'-'+str(ego_user),
                  'w') as f:
            f.write('%d %d\n' % (np.shape(g_train)[0], args.embedding_dim))
            for i in range(np.shape(g_train)[0]):
                e = ' '.join(map(lambda x: str(x), emb_list[i]))
                f.write('%s %s\n' % (str(i), e))

        # with open('/Users/xiulingwang/Downloads/' + DATASET + '/' + METHOD + '/embeds/' + F + '-' + str(ego_user),'w') as f:
        #     pickle.dump(data_loader.embedding_mapping(normalized_embedding), f)
        # print(args.graph_file)

                n2v_scores, train_edge_labels, test_edge_labels, test_preds, train_sim_matrix, test_sim_matrix, train_edge_embs, test_edge_embs, train_embs_1, train_embs_2, test_embs_1, test_embs_2=linkpre_scores1(args, emb_matrix, train_test_split, ego_user,DATASET,METHOD, F)
        return n2v_scores, train_edge_labels,test_edge_labels, test_preds,emb_matrix,train_sim_matrix,test_sim_matrix,train_edge_embs,test_edge_embs,train_embs_1,train_embs_2,test_embs_1,test_embs_2




    elif args.mode == 'test':
        test(args)


def linkpre_scores1(args, emb_matrix, train_test_split, ego_user,DATASET,METHOD, Flag):
    adj_train, train_edges, train_edges_false, val_edges, val_edges_false, \
    test_edges, test_edges_false = train_test_split
    start_time = time.time()
    # Generate bootstrapped edge embeddings (as is done in node2vec paper)
    # Edge embedding for (v1, v2) = hadamard product of node embeddings for v1, v2
    if args.edge_score_mode == "edge-emb":

        def get_edge_embeddings(edge_list,ego_user,DATASET, Flag, flag):
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
                print(np.shape(edge_emb))
                embs.append(edge_emb)
                embs_1.append(emb1)
                embs_2.append(emb2)
                sim_matrix.append(sim)
            embs = np.array(embs)
            sim_matrix = np.array(sim_matrix)
            embs_1=np.array(embs_1)
            embs_2 =np.array(embs_2)

            #with open('/Users/xiulingwang/Downloads/line-master/data/embds/' + str(ego_user) + flag + '-' + Flag, 'w') as f:
            #with open('/Users/xiulingwang/Downloads/'+DATASET+'/line/3-split/' + str(ego_user)+ '-' + flag + '-' + Flag+'-' +'embds','w') as f:
            #with open('/Users/xiulingwang/Downloads/line-master/data/embds/' + str(ego_user) + flag + '-' + Flag, 'w') as f:
            # with open('/Users/xiulingwang/Downloads/' + DATASET + '/' + METHOD + '/embeds/' + Flag + '-' + str(ego_user) + flag, 'w') as f:
            #     f.write('%d %d\n' % (edge_list.shape[0], args.embedding_dim))
            #     for i in range(edge_list.shape[0]):
            #         e = ' '.join(map(lambda x: str(x), embs[i]))
            #         f.write('%s %s %s\n' % (str(edge_list[i][0]), str(edge_list[i][1]), e))

            return embs,sim_matrix,embs_1,embs_2

        # Train-set edge embeddings
        pos_train_edge_embs ,pos_train_sim_matrix,pos_embs_1_train,pos_embs_2_train= get_edge_embeddings(train_edges,ego_user,DATASET, Flag, flag='pos-train')
        neg_train_edge_embs,neg_train_sim_matrix,neg_embs_1_train,neg_embs_2_train = get_edge_embeddings(train_edges_false, ego_user,DATASET,Flag, flag='neg-train')
        train_edge_embs = np.concatenate((pos_train_edge_embs, neg_train_edge_embs), axis=0)
        train_sim_matrix= np.concatenate((pos_train_sim_matrix, neg_train_sim_matrix), axis=0)
        train_embs_1 = np.concatenate((pos_embs_1_train, neg_embs_1_train), axis=0)
        train_embs_2 = np.concatenate((pos_embs_2_train, neg_embs_2_train), axis=0)

        # Create train-set edge labels: 1 = real edge, 0 = false edge
        train_edge_labels = np.concatenate((np.ones(len(train_edges)), np.zeros(len(train_edges_false))), axis=0)

        # Val-set edge embeddings, labels
        if len(val_edges) > 0 and len(val_edges_false) > 0:
            pos_val_edge_embs,pos_val_sim_matrix,pos_embs_1_val,pos_embs_2_val = get_edge_embeddings(val_edges,ego_user,DATASET,Flag, flag='pos-val')
            neg_val_edge_embs,neg_val_sim_matrix,neg_embs_1_val,neg_embs_2_val = get_edge_embeddings(val_edges_false,ego_user,DATASET,Flag, flag='neg-val')
            val_edge_embs = np.concatenate([pos_val_edge_embs, neg_val_edge_embs])
            val_edge_labels = np.concatenate((np.ones(len(val_edges)), np.zeros(len(val_edges_false))), axis=0)
            val_sim_matrix = np.concatenate((pos_val_sim_matrix, neg_val_sim_matrix), axis=0)
            val_embs_1 = np.concatenate((pos_embs_1_val, neg_embs_1_val), axis=0)
            val_embs_2 = np.concatenate((pos_embs_2_val, neg_embs_2_val), axis=0)

        # Test-set edge embeddings, labels
        pos_test_edge_embs,pos_test_sim_matrix,pos_embs_1_test,pos_embs_2_test = get_edge_embeddings(test_edges,ego_user,DATASET,Flag, flag='pos-test')
        neg_test_edge_embs ,neg_test_sim_matrix,neg_embs_1_test,neg_embs_2_test= get_edge_embeddings(test_edges_false,ego_user,DATASET,Flag, flag='neg-test')
        test_edge_embs = np.concatenate((pos_test_edge_embs, neg_test_edge_embs), axis=0)
        test_sim_matrix = np.concatenate((pos_test_sim_matrix, neg_test_sim_matrix), axis=0)
        test_embs_1 = np.concatenate((pos_embs_1_test, neg_embs_1_test), axis=0)
        test_embs_2 = np.concatenate((pos_embs_2_test, neg_embs_2_test), axis=0)

        # Create val-set edge labels: 1 = real edge, 0 = false edge
        test_edge_labels = np.concatenate((np.ones(len(test_edges)), np.zeros(len(test_edges_false))), axis=0)


        # Train logistic regression classifier on train-set edge embeddings
        edge_classifier = LogisticRegression(random_state=0)
        edge_classifier.fit(train_edge_embs, train_edge_labels)

        # Predicted edge scores: probability of being of class "1" (real edge)
        if len(val_edges) > 0 and len(val_edges_false) > 0:
            val_preds = edge_classifier.predict_proba(val_edge_embs)[:, 1]
        test_preds = edge_classifier.predict_proba(test_edge_embs)[:, 1]
        print(test_preds)
        print(np.shape(test_preds))

        runtime = time.time() - start_time

        # Calculate scores
        if len(val_edges) > 0 and len(val_edges_false) > 0:
            n2v_val_roc = roc_auc_score(val_edge_labels, val_preds)
            # n2v_val_roc_curve = roc_curve(val_edge_labels, val_preds)
            n2v_val_ap = average_precision_score(val_edge_labels, val_preds)
        else:
            n2v_val_roc = None
            n2v_val_roc_curve = None
            n2v_val_ap = None

        n2v_test_roc = roc_auc_score(test_edge_labels, test_preds)
        # n2v_test_roc_curve = roc_curve(test_edge_labels, test_preds)
        n2v_test_ap = average_precision_score(test_edge_labels, test_preds)


    # Generate edge scores using simple dot product of node embeddings (like in GAE paper)
    elif args.edge_score_mode == "dot-product":
        score_matrix = np.dot(emb_matrix, emb_matrix.T)
        runtime = time.time() - start_time

        # Val set scores
        if len(val_edges) > 0:
            n2v_val_roc, n2v_val_ap = get_roc_score(val_edges, val_edges_false, score_matrix, apply_sigmoid=True)
        else:
            n2v_val_roc = None
            n2v_val_roc_curve = None
            n2v_val_ap = None

        # Test set scores
        n2v_test_roc, n2v_test_ap = get_roc_score(test_edges, test_edges_false, score_matrix, apply_sigmoid=True)

    else:
        print
        "Invalid edge_score_mode! Either use edge-emb or dot-product."

    # Record scores
    n2v_scores = {}

    n2v_scores['test_roc'] = n2v_test_roc
    # n2v_scores['test_roc_curve'] = n2v_test_roc_curve
    n2v_scores['test_ap'] = n2v_test_ap

    n2v_scores['val_roc'] = n2v_val_roc
    # n2v_scores['val_roc_curve'] = n2v_val_roc_curve
    n2v_scores['val_ap'] = n2v_val_ap

    n2v_scores['runtime'] = runtime

    return n2v_scores, train_edge_labels,test_edge_labels, test_preds,train_sim_matrix,test_sim_matrix,train_edge_embs,test_edge_embs,train_embs_1,train_embs_2,test_embs_1,test_embs_2




def LINE2(g_train, train_test_split,graph_file,DATASET,METHOD,ego_user, F,dp,res,sigma,ord):
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_dim', default=128)
    parser.add_argument('--batch_size', default=1000)
    parser.add_argument('--K', default=5)
    parser.add_argument('--proximity', default='second-order', help='first-order or second-order')
    parser.add_argument('--learning_rate', default=0.025)
    parser.add_argument('--mode', default='train')
    parser.add_argument('--num_batches', default=2000)
    parser.add_argument('--total_graph', default=True)
    parser.add_argument('--graph_file', default='/Users/xiulingwang/Downloads/line-master/data/0-adj-feat.pkl')
    parser.add_argument('--edge_score_mode', default='edge-emb')
    parser.add_argument('--uid', default='0')
    parser.add_argument('--flag', default='weighted')
    args = parser.parse_args()
    #args.proximity='first-order'
    args.graph_file=graph_file
    args.uid = str(ego_user)
    args.flag=str(F)
    if ord=='s':
        args.proximity='second-order'
    if ord == 'f':
        args.proximity = 'first-order'
    print(args.graph_file)

    adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = train_test_split
    if args.mode == 'train':
        if dp==1:
            normalized_embedding=train_dp1(args,sigma)
        elif dp==0:
            normalized_embedding = train(args)

        elif dp==2:
            normalized_embedding = train_defense(args,train_edges,test_edges,g_train )

        elif dp==3:
            normalized_embedding = train_defense2(args,train_edges,test_edges,g_train )
        elif dp==4:
            normalized_embedding = train_defense3(args,train_edges,test_edges,g_train,F, res)

        elif dp==5:
            normalized_embedding = train(args)
        # data_loader = DBLPDataLoader(graph_file=args.graph_file)

        elif dp==6:
            normalized_embedding = train_adj_defense(args,sigma)

        emb_list = []
        # print(np.shape(g_train)[0])
        for node_index in range(0, np.shape(g_train)[0]):
            node_str = str(node_index)
            node_emb = normalized_embedding[node_index]
            emb_list.append(node_emb)
        emb_matrix = np.vstack(emb_list)
        # print(emb_list)
        # print(np.shape(emb_list))

        # with open(res+F+'-embeds-'+str(ego_user),
        #           'w') as f:
        #     f.write('%d %d\n' % (np.shape(g_train)[0], args.embedding_dim))
        #     for i in range(np.shape(g_train)[0]):
        #         e = ' '.join(map(lambda x: str(x), emb_list[i]))
        #         f.write('%s %s\n' % (str(i), e))

        # with open('/Users/xiulingwang/Downloads/' + DATASET + '/' + METHOD + '/embeds/' + F + '-' + str(ego_user),'w') as f:
        #     pickle.dump(data_loader.embedding_mapping(normalized_embedding), f)
        # print(args.graph_file)

        train_edge_labels, test_edge_labels, train_sim_matrix, test_sim_matrix, train_edge_embs, test_edge_embs, train_embs_1, train_embs_2, test_embs_1, test_embs_2, train_edges_sampled=linkpre_scores2(args, emb_matrix, train_test_split, ego_user,DATASET,METHOD, F)
        return train_edge_labels,test_edge_labels, emb_matrix,train_sim_matrix,test_sim_matrix,train_edge_embs,test_edge_embs,train_embs_1,train_embs_2,test_embs_1,test_embs_2,train_edges_sampled




    elif args.mode == 'test':
        test(args)




def train_dp(args,sigma):
    data_loader = DBLPDataLoader(graph_file=args.graph_file)
    suffix = args.proximity
    args.num_of_nodes = data_loader.num_of_nodes
    model = LINEModel(args)

    C=1

    with tf.Session() as sess:
        print(args)
        print('batches\tloss\tsampling time\ttraining_time\tdatetime')
        tf.global_variables_initializer().run()
        initial_embedding = sess.run(model.embedding)
        learning_rate = args.learning_rate
        sampling_time, training_time = 0, 0
        loss_dp=[]
        for b in range(args.num_batches):
            t1 = time.time()
            u_i, u_j, label = data_loader.fetch_batch(batch_size=args.batch_size, K=args.K)
            feed_dict = {model.u_i: u_i, model.u_j: u_j, model.label: label, model.learning_rate: learning_rate}
            # tf.Print(model.u_i)
            # print(sess.run(model.u_i))

            t2 = time.time()
            sampling_time += t2 - t1
            if b % 100 != 0:
                sess.run(model.train_op, feed_dict=feed_dict)
                training_time += time.time() - t2
                if learning_rate > args.learning_rate * 0.0001:
                    learning_rate = args.learning_rate * (1 - b / args.num_batches)
                else:
                    learning_rate = args.learning_rate * 0.0001


                gradients_list,loss = sess.run([model.gradients_list,model.loss], feed_dict=feed_dict)

                print(loss)

            else:
                gradients_list,loss = sess.run([model.gradients_list,model.loss], feed_dict=feed_dict)
                print('%d\t%f\t%0.2f\t%0.2f\t%s' % (b, loss, sampling_time, training_time,
                                                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
                sampling_time, training_time = 0, 0

                variables_list=tf.trainable_variables()
                # print(variables_list)
                # print(variables_list[0][0][0])
                # print(variables_list[1])


                print(loss)
                print('****')
            # print(gradients_list)
            # print(np.shape(gradients_list))
            # print(type(gradients_list))


                grads = [tf.zeros(p.shape) for p in tf.trainable_variables()]

                l2_norm=tf.constant(0.000)


                for gd_list in gradients_list:
                    # print('*****')
                    # print(gd_list)
                    # print(type(gd_list))
                    # print((gd_list * gd_list))
                    #
                    # print(tf.reduce_sum(gd_list * gd_list))

                    l2_norm=tf.add(l2_norm,tf.rsqrt(tf.reduce_sum(gd_list * gd_list) + 0.000001))
                    # print(l2_norm.eval(session=sess))
                    # print(l2_norm / C)
                    # exit()

                # divisor = max(tf.convert_to_tensor(1.00000001), l2_norm / C)
                divisor = tf.maximum(tf.convert_to_tensor(1.00000001), l2_norm/ C)
                for gd in gd_list:
                    # print(gd)
                    gd += gd / divisor
                    # print(gd)


                grads_noisy=(gd_list +tf.random_normal(tf.shape(grads),stddev=sigma))/args.batch_size

                # print(gradients_list)

                # print(grads_noisy)
                # # print(variables_list)
                # print('******')

                grads_noisy = grads_noisy.eval(session=sess)
                #
                # print(grads_noisy)
                # print(type(grads_noisy))
                # print(type(variables_list))
                #
                # print(np.shape(grads_noisy))
                # # print((variables_list).size())
                # print(np.shape(variables_list))
                # print(np.shape(variables_list[0]))

                # print(variables_list)



                model.optimizer.apply_gradients(zip(list(grads_noisy), variables_list))

                model.gradients_list=tf.convert_to_tensor(grads_noisy)

                # print(model.gradients_list)
                # print(model.gradients_list.eval(session=sess))

                # exit()
                gradients_list, loss = sess.run([model.gradients_list, model.loss], feed_dict=feed_dict)

                # _, loss = sess.run([list(grads_noisy), model.loss], feed_dict=feed_dict)
                print(loss)


            # exit()




            if b % 1000 == 0 or b == (args.num_batches - 1):
                embedding = sess.run(model.embedding)

                normalized_embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
                pickle.dump(data_loader.embedding_mapping(normalized_embedding),
                            open('data/embedding_%s_%s.pkl' % (args.uid,args.flag), 'wb'))
    return (embedding)


def linkpre_scores2(args, emb_matrix, train_test_split, ego_user,DATASET,METHOD, Flag):
    adj_train, train_edges, train_edges_false, val_edges, val_edges_false, \
    test_edges, test_edges_false = train_test_split
    start_time = time.time()
    # Generate bootstrapped edge embeddings (as is done in node2vec paper)
    # Edge embedding for (v1, v2) = hadamard product of node embeddings for v1, v2
    if args.edge_score_mode == "edge-emb":

        def get_edge_embeddings(edge_list,ego_user,DATASET, Flag, flag):
            tsts=[]
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
                sim2 = np.dot(emb1, emb2)
                # sim3 = np.sqrt(np.sum(np.sqrt(np.array(emb1)-np.array(emb2))))
                # print(sim3)
                sim3 = np.linalg.norm(np.array(emb1) - np.array(emb2))
                # print(sim3)
                sim4 = 1 / (1 + sim3)
                embs.append(edge_emb)
                embs_1.append(emb1)
                embs_2.append(emb2)
                sim_matrix.append([sim, sim2, sim3, sim4])

                tst = [node1, node2, sim, sim2, sim3, sim4]
                tsts.append(tst)

            embs = np.array(embs)
            sim_matrix = np.array(sim_matrix)
            embs_1 = np.array(embs_1)
            embs_2 = np.array(embs_2)

            name = ['node1', 'node2', 'sim1', 'sim2', 'sim3', 'sim4']
            result = pd.DataFrame(columns=name, data=tsts)
            result.to_csv("{}{}-similarity.csv".format(Flag, flag))



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
        pos_train_edge_embs ,pos_train_sim_matrix,pos_embs_1_train,pos_embs_2_train= get_edge_embeddings(train_edges_sampled,ego_user,DATASET, Flag, flag='pos-train')
        neg_train_edge_embs,neg_train_sim_matrix,neg_embs_1_train,neg_embs_2_train = get_edge_embeddings(train_edges_false, ego_user,DATASET,Flag, flag='neg-train')
        train_edge_embs = pos_train_edge_embs
        train_sim_matrix= pos_train_sim_matrix
        train_embs_1 = pos_embs_1_train
        train_embs_2 = pos_embs_2_train

        # Create train-set edge labels: 1 = real edge, 0 = false edge
        train_edge_labels = np.ones(len(train_edges_sampled))

        # Val-set edge embeddings, labels
        if len(val_edges) > 0 and len(val_edges_false) > 0:
            pos_val_edge_embs,pos_val_sim_matrix,pos_embs_1_val,pos_embs_2_val = get_edge_embeddings(val_edges,ego_user,DATASET,Flag, flag='pos-val')
            neg_val_edge_embs,neg_val_sim_matrix,neg_embs_1_val,neg_embs_2_val = get_edge_embeddings(val_edges_false,ego_user,DATASET,Flag, flag='neg-val')
            val_edge_embs = np.concatenate([pos_val_edge_embs, neg_val_edge_embs])
            val_edge_labels = np.concatenate((np.ones(len(val_edges)), np.zeros(len(val_edges_false))), axis=0)
            val_sim_matrix = np.concatenate((pos_val_sim_matrix, neg_val_sim_matrix), axis=0)
            val_embs_1 = np.concatenate((pos_embs_1_val, neg_embs_1_val), axis=0)
            val_embs_2 = np.concatenate((pos_embs_2_val, neg_embs_2_val), axis=0)

        # Test-set edge embeddings, labels
        pos_test_edge_embs,pos_test_sim_matrix,pos_embs_1_test,pos_embs_2_test = get_edge_embeddings(test_edges,ego_user,DATASET,Flag, flag='pos-test')
        neg_test_edge_embs ,neg_test_sim_matrix,neg_embs_1_test,neg_embs_2_test= get_edge_embeddings(test_edges_false,ego_user,DATASET,Flag, flag='neg-test')
        test_edge_embs = pos_test_edge_embs
        test_sim_matrix = pos_test_sim_matrix
        test_embs_1 = pos_embs_1_test
        test_embs_2 = pos_embs_2_test

        # Create val-set edge labels: 1 = real edge, 0 = false edge
        test_edge_labels = np.ones(len(test_edges))


    #     # Train logistic regression classifier on train-set edge embeddings
    #     edge_classifier = LogisticRegression(random_state=0)
    #     edge_classifier.fit(train_edge_embs, train_edge_labels)
    #
    #     # Predicted edge scores: probability of being of class "1" (real edge)
    #     if len(val_edges) > 0 and len(val_edges_false) > 0:
    #         val_preds = edge_classifier.predict_proba(val_edge_embs)[:, 1]
    #     test_preds = edge_classifier.predict_proba(test_edge_embs)[:, 1]
    #     print(test_preds)
    #     print(np.shape(test_preds))
    #
    #     runtime = time.time() - start_time
    #
    #     # Calculate scores
    #     if len(val_edges) > 0 and len(val_edges_false) > 0:
    #         n2v_val_roc = roc_auc_score(val_edge_labels, val_preds)
    #         # n2v_val_roc_curve = roc_curve(val_edge_labels, val_preds)
    #         n2v_val_ap = average_precision_score(val_edge_labels, val_preds)
    #     else:
    #         n2v_val_roc = None
    #         n2v_val_roc_curve = None
    #         n2v_val_ap = None
    #
    #     n2v_test_roc = roc_auc_score(test_edge_labels, test_preds)
    #     # n2v_test_roc_curve = roc_curve(test_edge_labels, test_preds)
    #     n2v_test_ap = average_precision_score(test_edge_labels, test_preds)
    #
    #
    # # Generate edge scores using simple dot product of node embeddings (like in GAE paper)
    # elif args.edge_score_mode == "dot-product":
    #     score_matrix = np.dot(emb_matrix, emb_matrix.T)
    #     runtime = time.time() - start_time
    #
    #     # Val set scores
    #     if len(val_edges) > 0:
    #         n2v_val_roc, n2v_val_ap = get_roc_score(val_edges, val_edges_false, score_matrix, apply_sigmoid=True)
    #     else:
    #         n2v_val_roc = None
    #         n2v_val_roc_curve = None
    #         n2v_val_ap = None
    #
    #     # Test set scores
    #     n2v_test_roc, n2v_test_ap = get_roc_score(test_edges, test_edges_false, score_matrix, apply_sigmoid=True)
    #
    # else:
    #     print
    #     "Invalid edge_score_mode! Either use edge-emb or dot-product."
    #
    # # Record scores
    # n2v_scores = {}
    #
    # n2v_scores['test_roc'] = n2v_test_roc
    # # n2v_scores['test_roc_curve'] = n2v_test_roc_curve
    # n2v_scores['test_ap'] = n2v_test_ap
    #
    # n2v_scores['val_roc'] = n2v_val_roc
    # # n2v_scores['val_roc_curve'] = n2v_val_roc_curve
    # n2v_scores['val_ap'] = n2v_val_ap
    #
    # n2v_scores['runtime'] = runtime

    return train_edge_labels,test_edge_labels,train_sim_matrix,test_sim_matrix,train_edge_embs,test_edge_embs,train_embs_1,train_embs_2,test_embs_1,test_embs_2,train_edges_sampled



def train_dp1(args,sigma):
    data_loader = DBLPDataLoader(graph_file=args.graph_file)
    suffix = args.proximity
    args.num_of_nodes = data_loader.num_of_nodes

    model = LINEModel(args)


    C=1

    with tf.Session() as sess:
        print(args)
        print('batches\tloss\tsampling time\ttraining_time\tdatetime')
        tf.global_variables_initializer().run()
        initial_embedding = sess.run(model.embedding)
        learning_rate = args.learning_rate
        sampling_time, training_time = 0, 0
        loss_dp=[]
        for b in range(args.num_batches):
            t1 = time.time()
            u_i, u_j, label = data_loader.fetch_batch(batch_size=args.batch_size, K=args.K)
            feed_dict = {model.u_i: u_i, model.u_j: u_j, model.label: label, model.learning_rate: learning_rate}
            # tf.Print(model.u_i)
            # print(sess.run(model.u_i))

            t2 = time.time()
            sampling_time += t2 - t1
            if b % 100 != 0:
                sess.run(model.train_op, feed_dict=feed_dict)
                training_time += time.time() - t2
                if learning_rate > args.learning_rate * 0.0001:
                    learning_rate = args.learning_rate * (1 - b / args.num_batches)
                else:
                    learning_rate = args.learning_rate * 0.0001

                # print(args.uid)
                gradients_list,loss = sess.run([model.gradients_list,model.loss], feed_dict=feed_dict)

                # print(loss)
                # print('****')

            else:
                # print(model.gradients_list)
                # print(model.loss)
                # print(feed_dict)
                gradients_list,loss = sess.run([model.gradients_list,model.loss], feed_dict=feed_dict)
                print('%d\t%f\t%0.2f\t%0.2f\t%s' % (b, loss, sampling_time, training_time,
                                                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
                sampling_time, training_time = 0, 0

                variables_list=tf.trainable_variables()
                # print(variables_list)
                # print(variables_list[0][0][0])
                # print(variables_list[1])


                # print(loss)
                # print('****')
                # print(gradients_list)
                # print(np.shape(gradients_list))
                # print(type(gradients_list))


                grads = [tf.zeros(p.shape) for p in tf.trainable_variables()]

                l2_norm=tf.constant(0.000)


                for gd_list in gradients_list:
                    # print('*****')
                    # print(gd_list)
                    # print(type(gd_list))
                    # print((gd_list * gd_list))
                    #

                    l2_norm=tf.add(l2_norm,tf.rsqrt(tf.reduce_sum(gd_list * gd_list) + 0.000001))
                    # print(l2_norm.eval(session=sess))
                    # print(l2_norm / C)
                    # exit()

                # divisor = max(tf.convert_to_tensor(1.00000001), l2_norm / C)
                divisor = tf.maximum(tf.convert_to_tensor(1.00000001), l2_norm/ C)
                for gd in gd_list:
                    # print(gd)
                    gd += gd / divisor
                    # print(gd)


                grads_noisy=(gd_list +tf.random_normal(tf.shape(grads),stddev=sigma))/args.batch_size

                # print(gradients_list)

                # print(grads_noisy)
                # # print(variables_list)
                # print('******')

                grads_noisy = grads_noisy.eval(session=sess)
                #
                # print(grads_noisy)
                # print(type(grads_noisy))

                # print(np.shape(grads_noisy))

                # print(type(variables_list))
                # # print((variables_list).size())
                # print(np.shape(variables_list))
                # print(np.shape(variables_list[0]))

                # print(variables_list)



                model.optimizer.apply_gradients(zip(list(grads_noisy), variables_list))

                model.gradients_list=tf.convert_to_tensor(grads_noisy)

                embedding = sess.run(model.embedding)
                # print(embedding)
                # print(type(embedding))
                # print(np.shape(embedding))

                # print(model.gradients_list)
                # print(model.gradients_list.eval(session=sess))

                # exit()
                gradients_list, loss = sess.run([model.gradients_list, model.loss], feed_dict=feed_dict)



                # print(gradients_list)
                # print(type(gradients_list))
                #
                # print(np.shape(gradients_list))

                for i in range(np.shape(embedding)[0]):
                    for j in range(np.shape(embedding)[1]):
                        if args.proximity == 'first-order':
                            embedding[i][j]+=gradients_list[0][i][j]*learning_rate
                        if args.proximity == 'second-order':
                            embedding[i][j]+=gradients_list[1][i][j]*learning_rate


                # print(embedding)

                model.embedding = tf.convert_to_tensor(embedding)

                # print(loss)
                #
                # gradients_list, loss = sess.run([model.gradients_list, model.loss], feed_dict=feed_dict)
                # print(loss)


            # exit()




            if b % 1000 == 0 or b == (args.num_batches - 1):
                embedding = sess.run(model.embedding)

                normalized_embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
                pickle.dump(data_loader.embedding_mapping(normalized_embedding),
                            open('data/embedding_%s_%s.pkl' % (args.uid,args.flag), 'wb'))
    return (embedding)


def train_defense(args,train_edges,test_edges,g_train):
    data_loader = DBLPDataLoader(graph_file=args.graph_file)
    suffix = args.proximity
    args.num_of_nodes = data_loader.num_of_nodes
    model = LINEModel(args)
    with tf.Session() as sess:
        print(args)
        print('batches\tloss\tsampling time\ttraining_time\tdatetime')
        tf.global_variables_initializer().run()
        initial_embedding = sess.run(model.embedding)
        learning_rate = args.learning_rate
        sampling_time, training_time = 0, 0
        for b in range(args.num_batches):
            print(sess.run(model.embedding))
            t1 = time.time()
            u_i, u_j, label = data_loader.fetch_batch(batch_size=args.batch_size, K=args.K)
            feed_dict = {model.u_i: u_i, model.u_j: u_j, model.label: label, model.learning_rate: learning_rate}
            t2 = time.time()
            sampling_time += t2 - t1
            if b % 100 != 0:
                sess.run(model.train_op, feed_dict=feed_dict)
                training_time += time.time() - t2
                if learning_rate > args.learning_rate * 0.0001:
                    learning_rate = args.learning_rate * (1 - b / args.num_batches)
                else:
                    learning_rate = args.learning_rate * 0.0001
            else:
                loss = sess.run(model.loss, feed_dict=feed_dict)
                print('%d\t%f\t%0.2f\t%0.2f\t%s' % (b, loss, sampling_time, training_time,
                                                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
                sampling_time, training_time = 0, 0

            loss_dis = discriminator(train_edges, test_edges, embedding= sess.run(model.embedding))
            print(loss,loss_dis)
            loss_new = loss - 100*tf.to_double(loss_dis)
            model.loss= tf.convert_to_tensor(loss_new)

            if b % 1000 == 0 or b == (args.num_batches - 1):
                embedding = sess.run(model.embedding)

                normalized_embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
                pickle.dump(data_loader.embedding_mapping(normalized_embedding),
                            open('data/embedding_%s_%s.pkl' % (args.uid,args.flag), 'wb'))
    return (embedding)

def train_defense2(args,train_edges,test_edges,g_train):
    data_loader = DBLPDataLoader(graph_file=args.graph_file)
    suffix = args.proximity
    args.num_of_nodes = data_loader.num_of_nodes
    model = LINEModel(args)
    with tf.Session() as sess:
        print(args)
        print('batches\tloss\tsampling time\ttraining_time\tdatetime')
        tf.global_variables_initializer().run()
        initial_embedding = sess.run(model.embedding)
        learning_rate = args.learning_rate
        sampling_time, training_time = 0, 0
        for b in range(args.num_batches):
            # print(sess.run(model.embedding))
            t1 = time.time()
            u_i, u_j, label = data_loader.fetch_batch(batch_size=args.batch_size, K=args.K)
            feed_dict = {model.u_i: u_i, model.u_j: u_j, model.label: label, model.learning_rate: learning_rate}
            t2 = time.time()
            sampling_time += t2 - t1
            if b % 100 != 0:
                sess.run(model.train_op, feed_dict=feed_dict)
                training_time += time.time() - t2
                if learning_rate > args.learning_rate * 0.0001:
                    learning_rate = args.learning_rate * (1 - b / args.num_batches)
                else:
                    learning_rate = args.learning_rate * 0.0001
            else:
                loss = sess.run(model.loss, feed_dict=feed_dict)
                print('%d\t%f\t%0.2f\t%0.2f\t%s' % (b, loss, sampling_time, training_time,
                                                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
                sampling_time, training_time = 0, 0

                loss_dis = discriminator(train_edges, test_edges, embedding= sess.run(model.embedding))
                print(loss,loss_dis)
                loss_new = loss - 10*tf.to_double(loss_dis)
                model.loss= tf.convert_to_tensor(loss_new)

            if b % 1000 == 0 or b == (args.num_batches - 1):
                embedding = sess.run(model.embedding)

                normalized_embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
                pickle.dump(data_loader.embedding_mapping(normalized_embedding),
                            open('data/embedding_%s_%s.pkl' % (args.uid,args.flag), 'wb'))
    return (embedding)

def train_defense3(args,train_edges,test_edges,g_train,F, res_dir):
    data_loader = DBLPDataLoader(graph_file=args.graph_file)
    suffix = args.proximity
    args.num_of_nodes = data_loader.num_of_nodes
    model = LINEModel(args)
    with tf.Session() as sess:
        print(args)
        print('batches\tloss\tsampling time\ttraining_time\tdatetime')
        tf.global_variables_initializer().run()
        initial_embedding = sess.run(model.embedding)
        learning_rate = args.learning_rate
        sampling_time, training_time = 0, 0
        cnt_it = 0
        min=1000000

        edgeall = list([list(edge_tuple) for edge_tuple in train_edges])

        # train_edges_sampled = random.sample(edgeall, np.shape(test_edges)[0])
        train_edges_sampled = random.sample(edgeall, np.shape(test_edges)[0])
        for b in range(args.num_batches):
            # print(sess.run(model.embedding))
            t1 = time.time()
            u_i, u_j, label = data_loader.fetch_batch(batch_size=args.batch_size, K=args.K)
            feed_dict = {model.u_i: u_i, model.u_j: u_j, model.label: label, model.learning_rate: learning_rate}
            t2 = time.time()
            sampling_time += t2 - t1
            if b % 100 != 0:
                sess.run(model.train_op, feed_dict=feed_dict)
                training_time += time.time() - t2
                # if learning_rate > args.learning_rate * 0.0001:
                #     learning_rate = args.learning_rate * (1 - b / args.num_batches)
                # else:
                #     learning_rate = args.learning_rate * 0.0001
            else:
                loss = sess.run(model.loss, feed_dict=feed_dict)
                print('%d\t%f\t%0.2f\t%0.2f\t%s' % (b, loss, sampling_time, training_time,
                                                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
                sampling_time, training_time = 0, 0

                if cnt_it==0:
                    loss_dis,acc_kmeans_sim, acc_mlp_sim, acc_rf_sim, acc_svm_sim = discriminator_gain(train_edges_sampled, test_edges, F, res_dir,cnt_it, embedding= sess.run(model.embedding))
                else:

                    loss_dis = discriminator_gain(train_edges_sampled, test_edges, F, res_dir,cnt_it, embedding= sess.run(model.embedding))
                print(loss,loss_dis)
                loss_new = loss+0.001*tf.to_double(loss_dis)
                model.loss= tf.convert_to_tensor(loss_new)
                print('loss:',model.loss)

                if min > loss_new.eval():
                    print('~~~~~~')
                    idx2vec = copy.deepcopy(sess.run(model.embedding))
                    # print('min')
                    # print(idx2vec[9])
                    min = loss
                    cnt_it = 0

                if min < loss_new.eval():
                    print('@@@@@@@@')
                    print(cnt_it)
                    cnt_it += 1
                    if cnt_it == 10:
                        if b >= 1:
                            break

            # if b >0 and cnt_it == 10:
            #     break

        print(acc_kmeans_sim, acc_mlp_sim, acc_rf_sim, acc_svm_sim)



        normalized_embedding = idx2vec / np.linalg.norm(idx2vec, axis=1, keepdims=True)
        pickle.dump(data_loader.embedding_mapping(normalized_embedding),
                    open('data/embedding_%s_%s.pkl' % (args.uid,args.flag), 'wb'))
    return (idx2vec)


def discriminator(train_edges,test_edges,embedding):

    # print(num_nodes)
    # print(word2idx)

    # start_time = time.time()
    # Generate bootstrapped edge embeddings (as is done in node2vec paper)
    # Edge embedding for (v1, v2) = hadamard product of node embeddings for v1, v2
    emb_matrix=embedding
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

    # y_score = mlp.predict(X_test)
    # ls = 0
    # for i in range(len(y_score)):
    #     if y_score[i] != y_test[i][2]:
    #         if y_score[i] == 1:
    #             ls += y_score[i]
    #         else:
    #             ls += 1-y_score[i]
    # loss = ls / len(y_score)

    # print("Training set score: %f" % mlp.score(X_train, y_train[:, 2]))
    # print("Test set score: %f" % mlp.score(X_test, y_test[:, 2]))
    #
    # y_score = mlp.predict(X_test)
    # print(metrics.f1_score(y_test[:, 2], y_score, average='micro'))
    # print(metrics.classification_report(y_test[:, 2], y_score, labels=range(3)))

    return loss


def discriminator_gain(train_edges_sampled,test_edges, F, res_dir,cnt_it,embedding):

    # print(num_nodes)
    # print(word2idx)

    # start_time = time.time()
    # Generate bootstrapped edge embeddings (as is done in node2vec paper)
    # Edge embedding for (v1, v2) = hadamard product of node embeddings for v1, v2
    emb_matrix=embedding
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
    gain = 0
    cnt_true = 0

    pree = svm.predict(X_test)
    # print('prob')
    # print(prob)
    # print('pree')
    # print(pree)
    for i in range(len(pree)):
        if i < np.shape(X_train_test)[0]:
            if pree[i] == 1:
                cnt_true += 1
                # gain+=np.log(prob[i][1])

        else:
            if pree[i] == 0:
                cnt_true += 1
                # gain += np.log(prob[i][0])
    # gain=gain/np.shape(X_train)[0]


    acc = cnt_true / np.shape(X_test)[0]
    gain = acc
    print('acc', acc)

    acc_sim = accuracy_score(pree, y_test[:, 2])
    print(acc_sim)
    # exit()
    # print('gain')
    # print(gain)

    # y_score = mlp.predict(X_train)
    # print(y_score)
    #
    # exit()
    if cnt_it == 0:
        acc_kmeans_sim, acc_mlp_sim, acc_rf_sim, acc_svm_sim = discriminator_gain2(train_edges_sampled, test_edges,
                                                                                   embedding, F, res_dir)
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


def discriminator_gain2(train_edges_sampled, test_edges, embedding, F, res_dir):
    # idx2vec = embedding.ivectors.weight.data.cpu().numpy()
    # print(num_nodes)
    # print(word2idx)

    emb_matrix = embedding

    # start_time = time.time()
    # Generate bootstrapped edge embeddings (as is done in node2vec paper)
    # Edge embedding for (v1, v2) = hadamard product of node embeddings for v1, v2

    def get_edge_embeddings(edge_list):
        embs = []
        sim_matrix = []
        embs_1 = []
        embs_2 = []
        for edge in edge_list:
            node1 = edge[0]
            node2 = edge[1]
            emb1 = emb_matrix[node1]
            # print(np.shape(emb1))
            emb2 = emb_matrix[node2]
            edge_emb = np.multiply(emb1, emb2)
            sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            # edge_emb = np.array(emb1) + np.array(emb2)
            # print(np.shape(edge_emb))
            embs.append(edge_emb)
            embs_1.append(emb1)
            embs_2.append(emb2)
            sim_matrix.append(sim)
        embs = np.array(embs)
        sim_matrix = np.array(sim_matrix)
        embs_1 = np.array(embs_1)
        embs_2 = np.array(embs_2)

        # #with open('/Users/xiulingwang/Downloads/line-master/data/embds/' + str(ego_user) + flag + '-' + Flag, 'w') as f:
        # with open('E:/python/banlance/code/' + DATASET + '/' + METHOD + '/embeds/' + Flag + '-' + str(ego_user) + flag,'w') as f:
        # #with open('/Users/xiulingwang/Downloads/'+DATASET+'/line/3-split/' + str(ego_user)+ '-' + flag + '-' + Flag+'-' +'embds','w') as f:
        #     f.write('%d %d\n' % (edge_list.shape[0], args.representation_size))
        #     for i in range(edge_list.shape[0]):
        #         e = ' '.join(map(lambda x: str(x), embs[i]))
        #         f.write('%s %s %s\n' % (str(edge_list[i][0]), str(edge_list[i][1]), e))

        return embs, sim_matrix, embs_1, embs_2

    # edgeall = list([list(edge_tuple) for edge_tuple in train_edges])
    #
    # # train_edges_sampled = random.sample(edgeall, np.shape(test_edges)[0])
    # train_edges_sampled = random.sample(edgeall, np.shape(test_edges)[0])


    # Train-set edge embeddings
    pos_train_edge_embs, pos_train_sim_matrix, pos_embs_1_train, pos_embs_2_train = get_edge_embeddings(
        train_edges_sampled)
    # neg_train_edge_embs,neg_train_sim_matrix,neg_embs_1_train,neg_embs_2_train = get_edge_embeddings(train_edges_false, ego_user,DATASET,Flag, flag='neg-train')
    train_edge_embs = pos_train_edge_embs
    train_sim_matrix = pos_train_sim_matrix
    train_embs_1 = pos_embs_1_train
    train_embs_2 = pos_embs_2_train

    # Create train-set edge labels: 1 = real edge, 0 = false edge
    train_edge_labels = np.ones(len(train_edges_sampled))

    # Test-set edge embeddings, labels
    pos_test_edge_embs, pos_test_sim_matrix, pos_embs_1_test, pos_embs_2_test = get_edge_embeddings(test_edges)
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

    tsts = []
    print(len(kmeans.labels_))
    for i in range(len(kmeans.labels_)):
        node1 = edges_list[i][0]
        node2 = edges_list[i][1]
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
