import emails
import classifier


def get_acc(test, predictions_list):
    correct_number = len(list(filter(lambda x: x[0] == x[1], zip(
        map(lambda x: x.is_spam, test), predictions_list)))) / len(test)
    return correct_number


def get_sensivity(test, predictions_list):
    actually_spam = sum([1 for x in test if x.is_spam])
    TP = len([1 for x, y in zip(test, predictions_list)
              if (y == True and x.is_spam)])
    return TP / actually_spam


def get_specifity(test, predictions_list):
    # TO TRZEBA NAPISAĆ (przekleić z get_sensivity i zmienić jedną linijkę zgodnie z
    # definicją specifity
    pass


def get_precision(test, predictions_list):
    # JW
    pass


def get_fmeas(test, predictions_list):
    precision = get_precision(test, predictions_list)
    sens = get_sensivity(test, predictions_list)
    return 2 * (precision * sens) / (precision + sens)


def evaluate(train_set, test_set, classifier):
    classifier.train(train_set)
    predictions_list = classifier.predict(test_set)
    acc = get_acc(test_set, predictions_list)
    sens = get_sensivity(test_set, predictions_list)
    spec = get_specifity(test_set, predictions_list)
    prec = get_precision(test_set, predictions_list)
    fmeas = get_fmeas(test_set, predictions_list)
    return acc, sens, spec, prec, fmeas


emails_list = emails.Email.emails_list
train_set = emails_list[:int(0.9 * len(emails_list))]
test_set = emails_list[int(0.9 * len(emails_list)):]
acc, sens, spec, prec, fmeas = evaluate(
    train_set, test_set, classifier.Bayes)
print('accuracy:\t', acc)
print('sensivity:\t', sens)
print('specifity:\t', spec)
print('precision:\t', prec)
print('f-measure:\t', fmeas)
