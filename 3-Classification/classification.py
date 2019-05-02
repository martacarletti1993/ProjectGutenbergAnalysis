from pymongo import MongoClient
from random import seed, sample
import pprint
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import dump_svmlight_file, load_svmlight_file
from sklearn.preprocessing import normalize
import numpy as np
import spacy
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import matplotlib.pyplot as plt
import warnings
from scipy.sparse import hstack

warnings.filterwarnings('ignore')


pp = pprint.PrettyPrinter(indent=4)

#Parameters to connect to the MongoDB server
MongoHost = 'localhost'
MongoPort = 27017
collection = 'lab1'

client = MongoClient('mongodb://'+MongoHost+':'+str(MongoPort)+'/lab1')
db=client[collection]

# Check Connection
print(db.client)

nlp = spacy.load('en', entity=True)


#Tokenize sentences
def sentences(text):
	to_replace = '\n\t'
	for x in to_replace:
		text = text.replace(x,'')
	return nltk.sent_tokenize(text.strip())

#Read a file and join the lines together
def read_file_join(filepath):
	with open(filepath, 'r') as f:
		text = f.readlines()
	return ''.join(text)

#Read a file but do not join the lines together
def read_file_no_join(filepath):
	with open(filepath, 'r') as f:
		text = f.readlines()
	return text



def preprocess_data(instances_num, authors_num, n):
	'''Preprocess data based on input paramenter. Select authors pseudorandomly
	and select sentences pseudorandomly too. In order to ensure a balanced set,
	the instances per author are obtained by deviding the number of instances
	by the number of authors.
	'''
	seed(1)
	instances = []
	labels = []
	authors = db[collection].find()
	selection = sorted(sample(range(0,authors.count(),1),authors_num))
	selected_authors = [authors[i] for i in selection]
	for author in selected_authors:
		sentences_list = []
		print(author['_id'])
		text = ''.join([read_file_join(book['path']) for book in author['books']])
		sentences_list = sentences(text)
		grouped_sentences = [sentences_list[i:i+n] for i in range(0,len(sentences_list)-n, n)]
		sent_selection = sorted(sample(range(0,len(grouped_sentences),1),int(instances_num/authors_num)))
		for i in sent_selection:
			instances.append(grouped_sentences[i])
			labels.append(author['_id'])

	#Map unique labels to integers
	d = dict([(y,x+1) for x,y in enumerate(sorted(set(labels)))])
	labels = [d[x] for x in labels]

	return instances, labels

def nltk2wn_tag(nltk_tag):
	'''Given a tag, return its WordNet equivalent
	'''
	if nltk_tag.startswith('J'):
		return nltk.corpus.wordnet.ADJ
	elif nltk_tag.startswith('V'):
		return nltk.corpus.wordnet.VERB
	elif nltk_tag.startswith('N'):
		return nltk.corpus.wordnet.NOUN
	elif nltk_tag.startswith('R'):
		return nltk.corpus.wordnet.ADV
	else: 
		# Missing: CD, IN, MD, PRP, CC, TO, punct(, : .)
		return 'Other'

def extract_features(instances, labels):
	'''Extract features from a set of instances. The features extracted are tokens
	POS, NER and averange sentence length. Features per instance along with a label
	for each are then stored
	'''
	#TF-IDF Vectorizer on tokens
	train_to_vectorize = [' '.join(instance) for instance in instances]


	# print(train_to_vectorize)




	#TF-IDF Vectorizer on tokens
	vectorizer = TfidfVectorizer()
	vectorized_train_data = vectorizer.fit_transform(train_to_vectorize)

	dump_svmlight_file(vectorized_train_data, labels, './lab3_data/1000/train.tokens.svmlight')
	print(vectorized_train_data.shape)


	# Create POS and NER arrays
	pos_to_vectorize = []
	ner_to_vectorize = []
	for instance in train_to_vectorize:
		NLP = nlp(instance)

		parts_of_speech = ''
		for token in NLP:
			tag = nltk2wn_tag(token.tag_)
			parts_of_speech += ' '+tag
		pos_to_vectorize.append(parts_of_speech)

		ners = ''
		for ent in NLP.ents:
			ners += ' '+ent.label_
		ner_to_vectorize.append(ners)
	print(len(pos_to_vectorize))
	print(len(ner_to_vectorize))

	# print(pos_to_vectorize)

	#TF-IDF Vectorizer on NER and on POS
	vectorizer = TfidfVectorizer()
	vectorized_pos_data = vectorizer.fit_transform(pos_to_vectorize)
	vectorizer = TfidfVectorizer()
	vectorized_ner_data = vectorizer.fit_transform(ner_to_vectorize)

	#Dump data to svmlight files
	dump_svmlight_file(vectorized_pos_data, labels, './lab3_data/1000/train.pos.svmlight')
	dump_svmlight_file(vectorized_ner_data, labels, './lab3_data/1000/train.ner.svmlight')
	print(vectorized_pos_data.shape)
	print(vectorized_ner_data.shape)


	#Average Sentence length
	length_train_data = []
	for instance in instances:
		instance_length = 0
		for sent in instance:
			instance_length += len(nltk.word_tokenize(sent))
		length_train_data.append(instance_length/len(instance))
	normalized_length = normalize([[x for x in length_train_data]])
	dump_svmlight_file(normalized_length.T, labels, './lab3_data/1000/train.sent_length.svmlight')

def train_clf(X, y, classifier='nb', alpha=1.0):
	'''Train a classifier
	'''
	if classifier == 'nb':
		classifier = MultinomialNB(alpha=alpha, fit_prior=True, class_prior=None)
	elif classifier == 'maxent':
		classifier = LogisticRegression(solver='lbfgs', multi_class='auto')
	classifier.fit(X, y)
	return classifier

def evaluate( classifier, X):
	'''	Predict y given a classifier and X
	'''
	y_pred = classifier.predict( X )
	return y_pred 

def visualize_best_alpha(measure, X_train, X_test, y_train, y_test, feature):
	'''Given train and test sets along with their labels, a measure and a feature,
	visualise how Naive Bayes performs on the train data with different alphas
	'''
	alphas = np.linspace(0,1,11)
	f1_micro = []
	f1_macro = []
	f1_weighted = []
	for alpha in alphas:
		clf = MultinomialNB(alpha=alpha)
		clf.fit(X_train, y_train)
		y_test_predict = clf.predict(X_test)
		class_report = classification_report(y_test,y_test_predict, output_dict=True)
		f1_micro.append(class_report['micro avg'][measure])
		f1_macro.append(class_report['macro avg'][measure])
		f1_weighted.append(class_report['weighted avg'][measure])
	fig = plt.figure()
	plt.plot(alphas,f1_micro,label="Micro vs alpha")
	plt.plot(alphas,f1_macro,label="Macro vs alpha")
	plt.plot(alphas,f1_weighted,label="Weighted vs alpha")
	plt.xlabel('alpha')
	plt.ylabel(measure)
	plt.title(feature)
	plt.legend()
	plt.show()

def compute_score(y_true, y_pred, score='acc', average=None ):
	'''Given a set of true labels, a set of predicted labels and a metric,
	compute the score
	'''
	if score == 'acc':
		return metrics.accuracy_score( y_true, y_pred )
	elif score == 'f1':
		p,r,f,_ = precision_recall_fscore_support( y_true, y_pred, average=average )
		return f
	else:
		sys.exit("Unknown score:", score)


def classification(test_size):
	'''Perform classification on each of the features, using each of Naive Bayes and Linear Regression.
	The test_size is given.
	'''
	test_features = [('./lab3_data/1000/train.tokens.svmlight', 'tokens', 0.1),
					('./lab3_data/1000/train.pos.svmlight', 'pos', 0.5),
					('./lab3_data/1000/train.ner.svmlight', 'ner', 0.5),
					('./lab3_data/1000/train.sent_length.svmlight', 'sent_length', 0.5)]
	all_acc, all_f1 = [], []
	# X_train_all, X_test_all, y_train_all, y_test_all = [], [], [], []
	for file, feature, alpha in test_features:
		X, y = load_svmlight_file(file)
		print(X.shape, y.shape)
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

		#Stacking for training on all features
		# X_train_all, X_test_all, y_train_all, y_test_all = hstack( X_train_all, X_train ), hstack( X_test_all, X_test), hstack(y_train_all, y_train), hstack(y_test_all, y_test)


		#Classify and report on scores
		for classifier in ['nb', 'maxent']:
			clf = train_clf(X_train, y_train, classifier, alpha)
			y_pred = evaluate(clf, X_test)
			classifier_name = 'Naive Bayes' if classifier == 'nb' else 'Logistic Regression'
			print('='*53)
			print(classifier_name + ' on ' + feature)
			print('-'*53)


			acc = compute_score(y_test, y_pred, score='acc')
			f1 = compute_score(y_test, y_pred, score='f1', average='macro')
			
			all_acc.append(acc)
			all_f1.append(f1)

			print('acc = ', acc)
			target_names = ['author '+str(n) for n in range(20)]
			print(classification_report(y_test, y_pred, target_names=target_names))


	#Prepare a figure to display
	clf_types = ['NB tokens', 'LR tokens',
				'NB POS', 'LR POS',
				'NB NER', 'LR NER',
				'NB Size', 'LR Size']
	fig, ax = plt.subplots()
	index = np.arange(len(all_f1))
	bar_width = 0.35
	opacity = 0.8

	rects1 = plt.bar(index, all_acc, bar_width,
	alpha=opacity,
	color='b',
	label='Acc')
	 
	rects2 = plt.bar(index + bar_width, all_f1, bar_width,
	alpha=opacity,
	color='g',
	label='Macro-F1')
	 
	plt.xlabel('Clf type')
	plt.ylabel('Scores')
	plt.title('Scores by clf type')
	plt.xticks(index + bar_width, [str(m) for m in clf_types])
	plt.legend()
	plt.show()


	#Train on all features
	# for classifier in ['nb', 'maxent']:
	# 		clf = train_clf(X_train_all, y_train_all, classifier, 0.1)
	# 		y_pred = evaluate(clf, X_test_all)
	# 		classifier_name = 'Naive Bayes' if classifier == 'nb' else 'Logistic Regression'
	# 		print('='*53)
	# 		print(classifier_name + ' on ' + feature)
	# 		print('-'*53)

	# 		print('acc = ', acc)
	# 		target_names = ['author '+str(n) for n in range(20)]
	# 		print(classification_report(y_test_all, y_pred, target_names=target_names))


def best_alpha_figures(test_size):
	test_features = [('./lab3_data/1000/train.tokens.svmlight', 'tokens'),
					('./lab3_data/1000/train.pos.svmlight', 'pos'),
					('./lab3_data/1000/train.ner.svmlight', 'ner'),
					('./lab3_data/1000/train.sent_length.svmlight', 'sent_length')]
	for file, feature in test_features:
		X, y = load_svmlight_file(file)
		print(X.shape, y.shape)
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
		visualize_best_alpha('f1-score', X_train, X_test, y_train, y_test, feature)


if __name__=="__main__":

	# Preprocess the data, setting number of instance, authors and sentences per instance and then extract features
	# instances, labels = preprocess_data(1000, 20, 10)
	# extract_features(instances, labels)
	

	#Run classification, setting the test set size
	classification(0.2)

	#Run to visualise performance of Naive Bayes with different alphas
	# best_alpha_figures(0.2)

	