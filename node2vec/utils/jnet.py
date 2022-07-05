import sys
import numpy


class one_mode_network(object):
    def __init__(self, data=[]):
        self.adj = data
        self.nodes = len(data)

    def get_contacts(self, node):
        contacts = []
        for num, c in enumerate(self.adj[node]):
            if c != 0:
                contacts.append(num)
        return contacts

    def get_direct_weight(self, node1, node2):
        if node2 in self.get_contacts(node1):
            total = 0
            for i in self.adj[node1]:
                total += i
            return total
        else:
            return 0

    def get_direct_proportion(self, node1, node2):
        if node2 in self.get_contacts(node1):
            return_val = self.adj[node1][node2] / float(self.get_direct_weight(node1, node2))
            return return_val
        else:
            return 0

    def get_shared_contacts(self, node1, node2):
        n1_contacts = self.get_contacts(node1)
        n2_contacts = self.get_contacts(node2)
        overlap = set(n1_contacts).intersection(n2_contacts)
        return overlap

    def get_constraint(self, i, j):
        # if i has no contacts, it is completely constrainted by every j
        if len(self.get_contacts(i)) == 0:
            return 1
        else:
            p_ij = self.get_direct_proportion(i, j)
            total = 0
            for q in self.get_shared_contacts(i, j):
                p_iq = self.get_direct_proportion(i, q)
                p_qj = self.get_direct_proportion(q, j)
                total += p_iq * p_qj

            return ((p_ij + total) ** 2)

    def get_marginal_contact(self, q, j):
        numerator = self.adj[q][j]
        cur_max = 0
        for k in self.get_contacts(j):
            if self.adj[j][k] > cur_max:
                cur_max = self.adj[j][k]

        denominator = cur_max

        return numerator / float(denominator)

    def get_effective_size(self, i):
        sum1 = 0
        for j in self.get_contacts(i):
            if j == i:
                continue
            sum2 = 0
            for q in self.get_shared_contacts(i, j):
                if (q == i) or (q == j):
                    continue
                p_iq = self.get_direct_proportion(i, q)
                m_qj = self.get_marginal_contact(q, j)
                sum2 += p_iq * m_qj

            sum1 += 1 - sum2
        return sum1

    def get_agg_constraint(self, i):
        agg = 0
        for j in range(self.nodes):
            if j != i:
                agg += self.get_constraint(i, j)

        return agg

    def get_avg_constraint(self, i):
        non_zero_list = []
        for j in range(self.nodes):
            if j != i:
                c_ij = self.get_constraint(i, j)
                if c_ij != 0:
                    non_zero_list.append(c_ij)

        if len(non_zero_list) == 0:
            return 0
        else:
            return self.get_agg_constraint(i) / float(len(non_zero_list))


class two_mode_network(object):

    def __init__(self, data=[]):
        # if no data, create empty matrix
        if not data:
            self.p, self.s = 100, 100
            self.adj = [[0 for x in range(self.p)] for y in range(self.s)]
            self.adjT = [list(i) for i in zip(*self.adj)]


        else:
            self.p, self.s = len(data), len(data[0])
            self.adj = data

            # the transpose of the adjacency matrix
            # secondary and primary nodes swapped
            self.adjT = [list(i) for i in zip(*self.adj)]

    # return a projected one-mode network from a given two-mode one.
    # project primary node set if 'Primary' true, otherwise secondary
    def project(self, primary, better):
        if primary:
            adj = self.adj
        else:
            adj = self.adjT

        om_net = [[0 for x in range(len(adj))] for y in range(len(adj))]
        for n1, row1 in enumerate(adj):
            for n2, row2 in enumerate(adj):
                if n1 != n2:
                    n1_contacts = self.get_contacts(n1, primary)
                    n2_contacts = self.get_contacts(n2, primary)
                    overlap = set(n1_contacts).intersection(n2_contacts)
                    overlap_size = len(overlap)
                    if overlap_size > 0:
                        if better:
                            om_net[n1][n2] = overlap_size
                        else:
                            om_net[n1][n2] = 1
        return om_net

    # return list of contacts for a given primary or secondary node
    def get_contacts(self, node, primary):
        contacts = []
        if primary:
            adj = self.adj
        else:
            adj = self.adjT

        for num, c in enumerate(adj[node]):
            if c != 0:
                contacts.append(num)

        return contacts

    # bridging measure
    def get_effective_size(self, i, primary):

        # two-step contacts of i
        s_2_i = []

        # one-step contacts of i
        s_1_i = self.get_contacts(i, primary)
        # print s_1_i

        if len(s_1_i) == 0:
            return 0

        # fill set of two-step contacts for i
        for j in s_1_i:
            s_2_i = set(s_2_i).union(self.get_contacts(j, not primary))

        if i in s_2_i:
            s_2_i.remove(i)

        if len(s_2_i) == 0:
            return 0

        # list of all i's one-step contacts that are shared by each of i's
        # two-step contacts
        list_contacts = []

        # calculate effective size
        for j in s_2_i:
            # get one-step contacts of j
            s_1_j = self.get_contacts(j, primary)

            # get intersection of j's one-step contacts and i's one-step
            # contacts
            ss_1_j = set(s_1_j).intersection(s_1_i)

            list_contacts.append(list(ss_1_j))

        agg_redundancy = 0
        for c1, j in enumerate(list_contacts):
            j_redundancy = 0
            for c2, k in enumerate(list_contacts):
                if c1 == c2:
                    continue
                overlap = set(j).intersection(k)

                if len(overlap) > 0:
                    # OLD WAY
                    j_redundancy += 1

                # NEW MULTIPLIER IDEA
                # j_redundancy += 1 * (1/float(len(j)))
            agg_redundancy += j_redundancy / float(len(s_2_i))

        # agg_redundancy = agg_redundancy / float(len(s_2_i))
        ES_i = len(s_2_i) - agg_redundancy
        return ES_i

    # calculate the constraint on n1 by n2 (default both are primary)
    def get_constraint(self, n1, n2, primary):
        if primary:
            adj = self.adj
        else:
            adj = self.adjT

        # get a list of contacts for each given node
        n1_contacts = self.get_contacts(n1, primary)
        n2_contacts = self.get_contacts(n2, primary)

        # if a node has no non-pendant contacts, it has a constraint of 1
        if len(n1_contacts) == 0:
            return 1

        overlap = set(n1_contacts).intersection(n2_contacts)
        overlap_size = len(overlap)
        denom_size = len(n1_contacts)
        if denom_size == 0:
            return 1

        sigma = 0
        for e in overlap:
            sigma += adj[n1][e]
        # sigma += max((self.adj[n1][e]), (self.adj[n2][e]))

        return ((overlap_size * sigma) / float(denom_size)) ** 2

    # calculate the aggregate constraint on node
    def get_agg_constraint(self, node, primary):
        agg = 0
        if primary:
            adj = self.adj
        else:
            adj = self.adjT

        for i in range(len(adj)):
            if i != node:
                agg += self.get_constraint(node, i, primary)

        return agg

    # calculate the average constraint on node
    def get_avg_constraint(self, i, primary):
        if primary:
            adj = self.adj
        else:
            adj = self.adjT

        non_zero_list = []
        for j in range(len(adj)):
            if j != i:
                c_ij = self.get_constraint(i, j, primary)
                if c_ij != 0:
                    non_zero_list.append(c_ij)

        if len(non_zero_list) == 0:
            return 0
        else:
            return self.get_agg_constraint(i, primary) / float(len(non_zero_list))

    # determine if node is a pendant
    def is_pendant(self, node, primary):
        if not primary:
            adj = self.adj
        else:
            adj = self.adjT

        total = 0
        for row in adj:
            if total > 1:
                return False
            if row[node] == 1:
                total += 1

        return True

    # any ties between primary nodes and secondary pendants are assigned 0
    # if primary = True
    def remove_pendant_ties(self, primary):
        if primary:
            adj = self.adj
        else:
            adj = self.adjT

        for num1, row in enumerate(adj):
            for num2, col in enumerate(row):
                if self.is_pendant(num2, not primary):
                    adj[num1][num2] = 0

    # assign weights evenly
    def assign_weights(self, primary):
        if primary:
            adj = self.adj
        else:
            adj = self.adjT

        self.remove_pendant_ties(primary)
        for num1, row in enumerate(adj):
            total = 0
            for num2, col in enumerate(row):
                if (adj[num1][num2] == 1):
                    total += 1
            for a, b in enumerate(row):
                if (adj[num1][a] == 1) and (total != 0):
                    adj[num1][a] = (1 / float(total))


# create a random 2-mode network with p primary nodes and s secondary nodes,
# with each p having a probability t of forming a connection between any s
def create_random_tm_net(t):
    rand_tm_net = two_mode_network()
    for num1, row in enumerate(rand_tm_net.adj):
        for num2, col in enumerate(row):
            if (numpy.random.random() < t):
                rand_tm_net.adj[num1][num2] = 1
    rand_tm_net.adjT = [list(i) for i in zip(*rand_tm_net.adj)]
    return rand_tm_net
