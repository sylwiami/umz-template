import nltk
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.stem import SnowballStemmer


class classifier():
    @classmethod
    def set_apriori(cls, data_set):
        spam_number = len(list(filter(lambda x: x.is_spam, data_set)))
        ham_number = len(list(filter(lambda x: not x.is_spam, data_set)))
        cls.total_number = len(data_set)
        cls.spam_number = spam_number
        cls.ham_number = ham_number
        cls.apriori_spam = spam_number / cls.total_number
        cls.apriori_ham = ham_number / cls.total_number


class zero_rule(classifier):
    def train(data_set):
        zero_rule.set_apriori(data_set)
        if zero_rule.apriori_spam > zero_rule.apriori_ham:
            zero_rule.is_spam = True
        else:
            zero_rule.is_spam = False

    def predict(data_set):
        return [zero_rule.is_spam for _ in range(len(data_set))]


class Bayes(classifier):

    @classmethod
    def train(cls, data_set):
        cls.spam_words_dict = dict()
        cls.ham_words_dict = dict()
        cls.set_apriori(data_set)
        for email in data_set:
            if email.is_spam:
                cls.add_tokens_to_dict(email, cls.spam_words_dict)
            else:
                cls.add_tokens_to_dict(email, cls.ham_words_dict)
            cls.print_progress(data_set)
        cls.all_words_list = list(set(cls.spam_words_dict.keys()) | set(
            cls.ham_words_dict.keys()))
        cls.set_all_words_probs()

    @classmethod
    def predict(cls, data_set):
        preds = []
        progress = 0
        for email in data_set:
            if progress % 500 == 0:
                print('testing:\t', progress, '/', len(data_set))
            progress += 1
            v = cls.mail_body_to_vec(email)
            if cls.get_mail_probability(v, 'spam') > cls.get_mail_probability(v, 'ham'):
                preds.append(True)
            else:
                preds.append(False)
        return preds

    @classmethod
    def set_all_words_probs(cls,):
        cls.all_words_spam_probs = [
            -1 for i in range(len(cls.all_words_list))]
        cls.all_words_ham_probs = [
            -1 for i in range(len(cls.all_words_list))]
        for i in range(len(cls.all_words_list)):
            cls.all_words_spam_probs[i] = cls.get_word_prob(
                cls.all_words_list[i], 'spam')
            cls.all_words_ham_probs[i] = cls.get_word_prob(
                cls.all_words_list[i], 'ham')

    @classmethod
    def add_tokens_to_dict(cls, email, word_dict):
        words = [w for w in nltk.word_tokenize(email.body) if w.isalpha()]
        words = set(words)
        for word in words:
            if word in word_dict:
                word_dict[word] += 1
            else:
                word_dict[word] = 1

    @classmethod
    def print_progress(cls, data_set):
        if 'progress' not in dir(cls):
            cls.progress = 0
        if cls.progress % 1000 == 0:
            print('training:\t', cls.progress, '/', len(data_set))
        cls.progress += 1

    @classmethod
    def get_word_prob(cls, word, type):
        assert type in ('spam', 'ham')
        if type == 'spam':
            if word in cls.spam_words_dict.keys():
                return (cls.spam_words_dict[word] / cls.total_number) * (cls.apriori_spam / cls.apriori_ham)
            else:
                return 0
        else:
            if word in cls.ham_words_dict.keys():
                return (cls.ham_words_dict[word] / cls.total_number) * (cls.apriori_ham / cls.apriori_spam)
            return 0

    @classmethod
    def mail_body_to_vec(cls, email):
        email_dict = dict()
        cls.add_tokens_to_dict(email, email_dict)
        mail_words = email_dict.keys()
        mail_vec = [-1 for i in range(len(cls.all_words_list))]
        for i in range(len(cls.all_words_list)):
            if cls.all_words_list[i] in mail_words:
                mail_vec[i] = 1
            else:
                mail_vec[i] = 0
        return mail_vec

    @classmethod
    def get_mail_probability(cls, mail_vec, type):
        assert type in ('spam', 'ham')
        if type == 'spam':
            word_list = cls.all_words_spam_probs
        else:
            word_list = cls.all_words_ham_probs
        p = 1
        for i in range(len(mail_vec)):
            if mail_vec[i]:
                p = p * word_list[i]
            else:
                p = p * (1 - word_list[i])
        return p


class Bayes_stemmed(Bayes):
    @classmethod
    def add_tokens_to_dict(cls, email, word_dict):
        stemmer = cls.STEMMER
        words = [stemmer.stem(w)
                 for w in nltk.word_tokenize(email.body) if w.isalpha()]
        words = set(words)
        for word in words:
            if word in word_dict:
                word_dict[word] += 1
            else:
                word_dict[word] = 1


class Bayes_stemmed_porter(Bayes_stemmed):

    STEMMER = PorterStemmer()


class Bayes_stemmed_lancester(Bayes_stemmed):

    STEMMER = LancasterStemmer()


class Bayes_stemmed_snowball(Bayes_stemmed):

    STEMMER = SnowballStemmer('english')


class Bayes_laplac(Bayes):
    @classmethod
    def get_word_prob(cls, word, type):
        alfa = 0.1
        beta = 0.5
        assert type in ('spam', 'ham')
        if type == 'spam':
            if word in cls.spam_words_dict.keys():
                return ((alfa + cls.spam_words_dict[word]) / (beta + cls.total_number)) * (cls.apriori_spam / cls.apriori_ham)
            else:
                return ((alfa + 0) / (beta + cls.total_number)) * (cls.apriori_spam / cls.apriori_ham)
        else:
            if word in cls.ham_words_dict.keys():
                return ((alfa + cls.ham_words_dict[word]) / (beta + cls.total_number)) * (cls.apriori_ham / cls.apriori_spam)
            else:
                return ((alfa + 0) / (beta + cls.total_number)) * (cls.apriori_ham / cls.apriori_spam)
