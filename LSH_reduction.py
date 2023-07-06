## Exact same implementation, just a few more variables for the shrinkage process
class LSH_rp():
    def __init__(self, trainset, d = 784):
        self.array = trainset.data.numpy().reshape(len(trainset), d).astype('float') ## Turn dataset into a numpy array
        self.trainset_array = self.array - self.array.mean(axis = 0) ## Re-center the dataset array
        self.d = d

        self.ix_to_label = [] ## digit by index in the dataset

        self.label_to_ixs = {} ## indexes that represent each digit
        self.label_to_reps = {} ## How many repetitions by label

        for ix, (vec, label) in enumerate(trainset):
            self.ix_to_label.append(label)

            self.label_to_ixs[label] = self.label_to_ixs.get(label, [])
            self.label_to_ixs[label].append(ix)

            self.label_to_reps[label] = self.label_to_reps.get(label, 0) + 1

        self.num_labels = len(self.label_to_ixs)

    def hash_values(self, nbits = None, n_hashtables = None):
        if not nbits:
            nbits = int(math.log2(self.array.shape[0])/2) + 1
        if not n_hashtables:
            n_hashtables = int(math.log10(self.array.shape[0]))

        rand_tables = np.random.normal(0,1,(n_hashtables, nbits, self.d))

        self.tb_inthash_ix = [{} for i in range(n_hashtables)] ## save which indexes have each hash for each hash table
        self.n_hashtables = n_hashtables

        binary_array = (np.matmul(rand_tables, self.trainset_array.T) > 0).astype(int).transpose(0,2,1)
        self.hashes_array = binary_array.dot(np.flip(1<<np.arange(nbits))).T

        for ix in range(self.trainset_array.shape[0]):
            hash = self.hashes_array[ix]
            label = self.ix_to_label[ix]
            for i in range(self.n_hashtables):
                self.tb_inthash_ix[i][hash[i]] = self.tb_inthash_ix[i].get(hash[i], {lab:[] for lab in self.label_to_ixs.keys()})
                self.tb_inthash_ix[i][hash[i]][label].append(ix)

    def find_most_alike(self, ix):
        label = self.ix_to_label[ix]
        hash = self.hashes_array[ix]

        return set([j for i in range(self.n_hashtables) for j in self.tb_inthash_ix[i][hash[i]][label]]) - set([ix])

    def mean_dist(self):
        mean_th = {}

        Z = 1.65
        E = 0.05
        p = 0.5

        for label, N in self.label_to_reps.items():
            n = int(((Z**2 * N)*(p * (1-p))) / (((E**2)*(N-1)) + ((Z**2)*(p *(1-p))))) # Determine sample size
            curr_sample = np.random.choice(self.label_to_ixs[label], n)
            distance_sum = 0
            for ix in curr_sample:
                vec = self.array[ix]
                distance_sum += np.array([1-cos_sim(vec, self.array[i]) for i in self.find_most_alike(ix)]).mean()

            mean_th[label] = distance_sum/n

        return mean_th