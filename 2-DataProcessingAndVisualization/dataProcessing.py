import nltk
import string
import spacy
from nltk.tag import pos_tag
from pymongo import MongoClient
from bson.objectid import ObjectId
import os
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.parse.stanford import StanfordParser
from nltk.parse.corenlp import CoreNLPParser

#Parameters to connect to the MongoDB server
MongoHost = 'localhost'
MongoPort = 27017
collection = 'lab1'

client = MongoClient('mongodb://'+MongoHost+':'+str(MongoPort)+'/lab1')
db=client[collection]

# Check Connection
print(db.client)

parser = CoreNLPParser()
nlp = spacy.load('en', entity=True)
translator = str.maketrans('', '', string.punctuation)

java_path = r'/usr/lib/jvm/java-8-openjdk-amd64/'
os.environ['JAVAHOME'] = java_path
scp = StanfordParser(path_to_jar='./stanford-parser-full-2018-10-17/stanford-parser.jar',
           path_to_models_jar='./stanford-parser-full-2018-10-17/stanford-parser-3.9.2-models.jar')

def sentences(text, n):
	'''Tokenize sentences
	'''
	to_replace = '\n\t'
	for x in to_replace:
		text = text.replace(x,'')
	return '\n\n'.join(nltk.sent_tokenize(text.strip())[:n])

def remove_punct(text):
	'''Remove puctuation
	'''
	return text.translate(translator)

def tokenize_text(text):
	'''Word tokenize a text
	'''
	return '\t'.join(nltk.word_tokenize(text))

# def syntactic_parse(text):

# 	return [next(parser.raw_parse(sent)) for sent in text]

def save_file(text, directory, filename):
	'''Given a string a directory and a filename, save the text 
	into a file with the specified name in the specified directory
	Parameters: text - a string
				directory - a directory
				filename - the name that the file should have
	Output: the confirmed filepath
	'''
	if not os.path.exists(directory):
		os.makedirs(directory)
	with open(directory+filename, 'w') as f:
		f.write(text)
	return directory+filename

def read_file_join(filepath):
	'''Read a file and join lines
	'''
	with open(filepath, 'r') as f:
		text = f.readlines()
	return ''.join(text)

def read_file_no_join(filepath):
	'''Read a file but do not join lines
	'''
	with open(filepath, 'r') as f:
		text = f.readlines()
	return text

def text_processing(text, save_path, filename, n):
	'''Process text and create 4 files, corresponding to
	sentences, tokens, NER and POS
	'''
	sent = sentences(text, n)
	sent_file = save_file(sent, save_path, filename+'.sentences')

	sent = remove_punct(sent)

	tokens = tokenize_text(sent)
	tokens_file = save_file(tokens, save_path, filename+'.tokens')

	NLP = nlp(sent)

	# NER = apply_NER(text)
	with open(save_path+filename+'.ner','w') as f:
		for ent in NLP.ents:
			f.write(ent.text)
			f.write('\t')
			f.write(ent.label_)
			f.write('\n')

	with open(save_path+filename+'.pos','w') as f:
		for token in NLP:
			f.write(token.text)
			f.write('\t')
			f.write(token.tag_)
			f.write('\n')
	return sent_file, tokens_file, save_path+filename+'.ner', save_path+filename+'.pos'

def process_author(author, n):
	'''Given an author and n, process the first n sentences for the author
	'''
	text = ''.join([read_file_join(book['path']) for book in author['books']])

	#path is lab2_data/ author_id plus - plus n / author_id, then corresponding extension
	sent, tokens, ner, pos = text_processing(text, './lab2_data/'+str(author['_id']) + '-' + str(n) +'/', str(author['_id']), n)
	processed_files = author['processed_files'] if author.get('processed_files') else {}
	processed_files[str(n)] = {'sentences': sent, 'tokens': tokens, 'NER': ner, 'POS': pos}
	db[collection].update_one({'_id':author['_id']}, {'$set': {'processed_files': processed_files}})


def const_parse(sentences):
	'''Apply constituency parsing to a sentence
	'''
	trees = []
	print(sentences)
	for sentence in sentences:
		print(sentence)
		parse_trees = list(scp.raw_parse(sentence))
		trees.append(parse_trees[0])
		print(parse_trees[0])
	trees = '\n\n'.join([str(tree) for tree in trees])
	return(trees)

def const_parse_author(author, n):
	'''Apply constituency parsing to n sentences for an author
	'''
	text = ''.join([read_file_join(book['path']) for book in author['books']])
	if author['processed_files'].get(str(n)):
		sentences = read_file_no_join(author['processed_files'][str(n)]['sentences'])
		processed_files = author['processed_files']
		trees = const_parse(sentences)
		saved_trees = save_file(trees, './lab2_data/'+str(author['_id']) + '-' + str(n) +'/', str(author['_id']) + '.trees')
		processed_files[str(n)]['trees'] = saved_trees
		print(processed_files)
	
def get_vocab_size(tokens):
	'''Get vocabulary size given a string with tab separated tokens
	'''
	return len(set(tokens.split('\t')))

def sent_size_stats(sentences):
	'''Get stats on sentence length - avg, min and max size
	'''
	tok_per_sent = []
	for sent in sentences:
		tok_per_sent.append(len(nltk.word_tokenize(sent)))
	avg_size = np.mean(tok_per_sent)
	min_size = np.min(tok_per_sent)
	max_size = np.max(tok_per_sent)
	print('Average:', avg_size)
	print('Min:', min_size)
	print('Max:', max_size)
	return avg_size, min_size, max_size

def get_freq_distribution(tags):
	'''Calculate frequency distribution
	'''
	return nltk.FreqDist(tags).most_common()

def descriptive_stats(author_id, n):
	'''Get descriptive statistics for a given author
	'''
	folder_name = './lab2_data/'+author_id+'-'+str(n)+'/'+author_id
	tokens = read_file_join(folder_name+ '.tokens')
	print(get_vocab_size(tokens))

	sentences = read_file_no_join(folder_name+'.sentences')
	sent_size_stats(sentences)

	tags_raw = read_file_no_join(folder_name+'.pos')
	tags = [line.split('\t')[1].strip() for line in tags_raw if len(line.split('\t')) == 2]
	freq = get_freq_distribution(tags)
	print(freq)

	ner_raw = read_file_no_join(folder_name+'.ner')
	ner_tags = [line.split('\t')[1].strip() for line in ner_raw if len(line.split('\t')) == 2]
	print(get_freq_distribution(ner_tags)[:10])

def plot_vocab_size(n):
	'''Plot vocabulary size comparison between authors for n sentences
	'''
	vocab_size_per_author = []
	for author in db[collection].find({'processed_files': {'$exists': str(n)}}):
		tokens = read_file_join(author['processed_files'][str(n)]['tokens'])
		vocab_size_per_author.append(get_vocab_size(tokens))


	fig1, ax1 = plt.subplots()
	ax1.bar(range(len(vocab_size_per_author)), vocab_size_per_author, color="blue")
	ax1.set_title('Vocabulary size by author')
	ax1.set_ylabel('Vocabulary size')
	ax1.set_xlabel('Author')
	plt.show()

def nltk2wn_tag(nltk_tag):
	'''Given an nltk tag, return the corresponding Wordnet tag
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
		return None

def nltk_lemmatize( postag_data ):
	'''
	Wordnet lemmatizer needs word + pos tag
	postag_data: list of list of pos tags  (each list of tokens corresponds to one review)
	Return: list of all the lemmas in the data
	'''
	lemmas_mi = []
	lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
	for item in postag_data:
		if len(item) == 2:
			wn_pos = nltk2wn_tag(item[1]) # Conversion to WordNet pos tags
			if wn_pos:#only Adj, Verbs, Nouns and Adverbs
				lemmas_mi.append( lemmatizer.lemmatize(item[0], wn_pos ) )
	return lemmas_mi


def produce_word_cloud(n, author_id):
	'''Given an author id from the database, and n, produce a word cloud
	with lemmatized and non-lemmatized entries for the n processed sentences for that author
	'''
	author = db[collection].find_one({'_id':ObjectId(author_id)})
	tokens = read_file_join(author['processed_files'][str(n)]['tokens'])
	tokens = tokens.split('\t')

	postag_data = read_file_join(author['processed_files'][str(n)]['POS'])
	postag_data = [line.split('\t') for line in postag_data.split('\n')]


	postag_data = [[item[0].lower(), item[1]] if len(item) == 2 else item for item in postag_data]


	print(postag_data)
	# for item in postag_data:

	# print(postag_data)
	lemmatized_tokens = nltk_lemmatize(postag_data)

	print(len(tokens))
	print(len(lemmatized_tokens))

	wordcloud = WordCloud().generate(' '.join(tokens))
	lem_wordcloud = WordCloud().generate(' '.join(lemmatized_tokens))

	fig=plt.figure()
	# plt.title('Word Clouds for '+author['name'])
	ax1 = fig.add_subplot(2,1,1)
	plt.imshow(wordcloud, interpolation='bilinear')
	ax1.set_title("Non lemmatized")
	plt.axis("off")
	ax2 = fig.add_subplot(2,1,2)
	plt.imshow(lem_wordcloud, interpolation='bilinear')
	ax2.set_title("Lemmatized")
	plt.axis("off")

	if not os.path.exists('./images/word_clouds/'):
		os.makedirs('./images/word_clouds/')

	plt.savefig('./images/word_clouds/'+str(author['_id'])+'-'+str(n)+'.png')

def plot_sent_size_stats(n, author_id):
	'''Make a box plot for an author and n processed sentences showing
	their sentence size
	'''
	author = db[collection].find_one({'_id':ObjectId(author_id)})
	sentences = read_file_no_join(author['processed_files'][str(n)]['sentences'])
	avg_size, min_size, max_size = sent_size_stats(sentences)

	fig1, ax1 = plt.subplots()
	ax1.set_title('Sentence length for '+author['name'])
	ax1.boxplot([avg_size,min_size,max_size])
	ax1.set_ylabel('Sentence Length')

	if not os.path.exists('./images/sent_length/'):
		os.makedirs('./images/sent_length/')

	plt.savefig('./images/sent_length/'+str(author['_id'])+'-'+str(n)+'.png')


def plot_pos_distribution(n, author_id):
	'''Plot POS distribution for an author
	'''
	author = db[collection].find_one({'_id':ObjectId(author_id)})
	tags_raw = read_file_no_join(author['processed_files'][str(n)]['POS'])
	tags = [line.split('\t')[1].strip() for line in tags_raw if len(line.split('\t')) == 2]
	freq = get_freq_distribution(tags)

	bar_width = 0.35
	freq_tags = [tag for tag, value in freq]
	freq_values = [value for tag, value in freq]

	index = np.arange(len(freq_values))
	fig, ax = plt.subplots()
	ax.bar(index, freq_values, 0.35, color="blue")
	plt.title('POS distribution for '+author['name'])
	plt.xticks(index + bar_width, [str(m) for m in freq_tags], rotation = 70)
	ax.set_ylabel('Absolute frequency')
	ax.set_xlabel('NLTK POS Tag')

	if not os.path.exists('./images/pos_tags/'):
		os.makedirs('./images/pos_tags/')

	plt.savefig('./images/pos_tags/'+str(author['_id'])+'-'+str(n)+'.png')


if __name__ == "__main__":
	# for author in db[collection].find():
	# 	process_author(author,100)
	
	#Visualizations
	plot_vocab_size(100)

	# author = db[collection].find_one({'_id':ObjectId('5cbdf975b580213d1e9d5494')})
	# produce_word_cloud(100, author['_id'])
	# plot_pos_distribution(100, author['_id'])
	# plot_sent_size_stats(100, author['_id'])

	# tokens = [['woman','NN'], ['Women','NN']]
	# lemmatized_tokens = nltk_lemmatize(tokens)
	# print(lemmatized_tokens)
