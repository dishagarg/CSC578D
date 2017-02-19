"""
Extending the linear regression algorithm to compute the error curve
and find a good learning rate.
"""
import math
import main as mn

global total_count

text = []
labels = []
test_text = []
test_label = []
files = ['trainlabels.txt', 'traindata.txt', 'testlabels.txt', 'testdata.txt', ]
labels = mn.read_txt(files[0], labels)
text = mn.read_txt(files[1], text)
test_label = mn.read_txt(files[0], test_label)
test_text = mn.read_txt(files[1], test_text)
total_test_count = len(test_label)
total_count = len(labels)


class NaiveBayesClassifier:
    """A Multinomial Naive Bayes Classifier."""

    def extract_vocabulary(self, text, len_count):
        """Extract vocabulary as list of words from the whole text."""
        vocab = []
        for i in range(len_count):
            vocab.append(text[i].split())
        vocab = sum(vocab, [])
        return list(set(vocab)), vocab

    def count_docs_in_class(self, labels, clas):
        """Count the docs of the specified class."""
        count_class = 0
        for i in range(total_count):
            if labels[i] == clas:
                count_class += 1
        return count_class

    def count_tokens_of_term(self, class_text, term):
        """Count the frequency of the specific term in its class text."""
        count = 0
        uniq_words, class_words = self.extract_vocabulary(class_text, len(class_text))
        for i in class_words:
            if term == i:
                count += 1
        return count, len(class_words)

    def concatenate_text_of_class(self, labels, text, clas):
        """Concatenate the text for each class."""
        class_text = []
        for i in range(total_count):
            if clas == labels[i]:
                class_text.append(text[i])
        return class_text

    def train_multinomial(self, class_list, labels, text):
        """Train the data and compute the probabilities for each term."""
        vocab, whole = self.extract_vocabulary(text, len(text))
        prior = []
        cond_prob = [[] for i in range(len(class_list))]
        for i in range(len(class_list)):
            N_c = self.count_docs_in_class(labels, class_list[i])
            prior.append(float(N_c) / total_count)
            class_text = self.concatenate_text_of_class(labels, text, class_list[i])
            for term in vocab:
                T_ct, all_terms = self.count_tokens_of_term(class_text, term)
                cond_prob[i].append(float(T_ct + 1) / (all_terms + len(vocab)))
        return vocab, prior, cond_prob

    def extract_tokens_from_doc(self, vocab, test_doc):
        """Extract the frequency of each vocab term in test_doc."""
        list_word = []
        for i in range(len(vocab)):
            list_word.append(test_doc.count(vocab[i]))
        return list_word

    def apply_multinomial_nb(self, class_list, vocab, prior, cond_prob, test_doc):
        """Test the data by computing the probabilities."""
        test_words = self.extract_tokens_from_doc(vocab, test_doc)
        score = [None] * len(class_list)
        for i in range(len(class_list)):
            score[i] = math.log(prior[i])
            for j in range(len(test_words)):
                score[i] += math.log(cond_prob[i][j]) * test_words[j]
        return abs(score.index(max(score)) - 1)

    def __init__(self, parent=None):
        """Start training and testing."""
        class_list = list(set(labels))
        vocab, prior, cond_prob = self.train_multinomial(class_list, labels, text)

        maxx = []
        for line in test_text:
            test_doc = line.split()
            mx = self.apply_multinomial_nb(class_list, vocab, prior, cond_prob, test_doc)
            maxx.append(mx)

        # Accuracy computation:
        true_count = 0
        false_count = 0
        for i in range(len(maxx)):
            if maxx[i] == int(test_label[i]):
                true_count += 1
            else:
                false_count += 1
        accuracy = (float(true_count) / total_test_count) * 100
        print "Accuracy: ", accuracy


if __name__ == "__main__":
    NaiveBayesClassifier()
