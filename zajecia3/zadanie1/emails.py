import os
import codecs
import random


class Email():

    emails_list = list()

    def load_emails(ham_path='ham', spam_path='spam'):
        if Email.emails_list:
            pass
        else:
            for file_name in os.listdir(ham_path):
                Email.emails_list.append(
                    Email(ham_path + '/' + file_name, False))
            for file_name in os.listdir(spam_path):
                Email.emails_list.append(
                    Email(spam_path + '/' + file_name, True))
        random.shuffle(Email.emails_list)
        Email.spam_len = len(
            list(filter(lambda x: x.is_spam, Email.emails_list)))
        Email.ham_len = len(
            list(filter(lambda x: not x.is_spam, Email.emails_list)))

    def __init__(self, email_path, is_spam):
        file = codecs.open(email_path, 'r', encoding='utf-8', errors='ignore')
        subject = file.readline()[9:]
        body = file.read()
        self.subject = subject
        self.body = body
        self.is_spam = is_spam


Email.load_emails()
