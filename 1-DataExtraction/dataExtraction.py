import requests
from bs4 import BeautifulSoup
import string
import re
import nltk
import os
import pprint
from unidecode import unidecode
import wikipedia
from pymongo import MongoClient
from author import Author
import urllib.request
from SPARQLWrapper import SPARQLWrapper, JSON
# import urllib2

#Parameters to connect to the MongoDB server
MongoHost = 'localhost'
MongoPort = 27017

client = MongoClient('mongodb://'+MongoHost+':'+str(MongoPort)+'/lab1')
db=client['lab1']

# Check Connection
print(db.client)

pp = pprint.PrettyPrinter(indent=4)

def preprocessing(text):
	'''Given a string containing the plain text of a Gutenberg book, get everything
	between the start and the end of the book, if its language is English
	Parameters: text - plain text of a Gutenberg book
	Output: the new text or None
	'''
	start = re.search(r'\*\*\*(\sSTART OF|START OF).*\*\*\*', text)
	end = re.search(r'\*\*\*(\sEND OF|END OF).*\*\*\*', text)
	if start != None and end != None:
		start = start.group(0)
		
		lang = re.search(r'Language: English', text.split(start)[0])
		if lang == None:
			return None

		end = end.group(0)
		full_text = text.split(start)[1].split(end)[0]
		return full_text
	return None

def get_k_sentences(text, k):
	'''Given a text and a number k, return the whole text if it contains no more
	than k sentences and update k by subtracting the number of sentences left;
	or return the first k sentences of the text and 0
	Parameters:
	text - the text to be tokenized
	k - the number of sentences k needed from the text
	Output: the new text and the updated k
	'''
	lines = nltk.sent_tokenize(text)
	sent_num = len(lines)
	if sent_num <= k:
		return text, k - sent_num
	return ' '.join(lines[:k]), 0

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
	with open(directory+filename[:200], 'w') as f:
		f.write(text)
	return directory+filename[:200]

def clean_name(name):
	'''Given a string, clean it from a set of symbols and replace all spaces
	with underscores in order to make the name filename friendly
	Parameters: name - a string
	Output: the cleaned name
	'''
	for char in ',.¶:!?/\\()"\';':
		name = name.replace(char, '')
		name = name.strip()
		name = name.replace(' ', '_')
	return name

def get_books_up_to_k_sentences(books, k):
	'''Given a list soup 'li' objects containing information about Gutenberg books,
	get from that list, until k sentences are reached
	Parameters: books - a list of soup 'li' objects containing infor about Gutenberg books
				k - the requied number of sentences
	Output: books_to_save - a list of tuples. Each tuple corresponds to a book and contains
			the text of the book before k was reached, the filename of the book and the name
			of the book
			k - the value of k at the end of the process
	'''
	books_to_save = []
	for book in books: #for each of the author's books
		#get the book name
		num = book.find('a')['href']
		book_name = book.find('a').getText()
		book_filename = unidecode(clean_name(book_name)) #generate a filename for the book
		book_link = 'http://www.gutenberg.org'+num #get the link to the book's page
		response = requests.get(book_link) #request the page
		all_versions = BeautifulSoup(response.content, 'html.parser')
		#from the book's page, get the plain text page (try in two different ways)
		plain_text_link = all_versions.find('a', {"type": "text/plain; charset=utf-8"})
		if plain_text_link == None: 
			plain_text_link = all_versions.find('a', {"type": "text/plain"})
		#if there is no plain text page for the book, continue to the next book
		if plain_text_link == None:
			continue
		response = requests.get("http:"+plain_text_link.attrs["href"]) #if there is a plain text page, get that page and the plain text from it
		plain_text = BeautifulSoup(response.content, "html.parser")
		full_text = preprocessing(plain_text.getText()) #preprocess the plain text
		#if full_text is None, continue to the next book
		if full_text == None:
			continue
		
		text_to_write, k = get_k_sentences(full_text,k) #get k sentences from the book and update k

		books_to_save.append((text_to_write, book_filename, book_name))
		
		#if k has reached 0, stop iterating through the books
		if k == 0:
			break
	return books_to_save, k

def get_multilingual_abstracts(author):
	'''Given an author name, get a list of abstracts in all available
	languages for that page
	Parameters: author - a string containing an author name
	Output: abstracts - a list of abstracts. Each abstract is a dictionary containing
						the abstract language, the link to the wikipedia page and the
						summary
			url_author - a link to the English wikipedia page for that author
	'''
	abstracts = []
	url_author = None
	wikipedia.set_lang("en")
	try:	
		author_wiki_page = wikipedia.page(authorclean) #retrieve the author wikipage
	except:
		author_wiki_page = "Not found" 		
	if author_wiki_page != "Not found": # if the wikipage is not -not found-
		url_author = author_wiki_page.url #get the url of the wikipage of the author

		soup = BeautifulSoup(urllib.request.urlopen(url_author))
		links = [(el.get('lang'), el.get('title'), el.get('href')) for el in soup.select('li.interlanguage-link > a')]

		for language, title, href in links:
			try:
			    page_title = title.split(u' – ')[0]
			    wikipedia.set_lang(language)
			    page = wikipedia.page(page_title)
			    abstracts.append({'language':title.split(u' – ')[1], 'link':href, 'text':page.summary})
			except:
				continue
	return abstracts, url_author

def get_literary_movements(wiki_link):
	'''Given a wikipedia link, retrieve the literary movements from
	the corresponding DBPedia page
	Parameters: wiki_link - a wikipedia link
	Output: movements - list of literary movements
	'''
	movements = []
	dbpedia_link = 'http://dbpedia.org/resource/'+wiki_link.split('/')[-1]
	sparql = SPARQLWrapper("http://dbpedia.org/sparql")
	sparql.setQuery('''
		PREFIX dbo: <http://dbpedia.org/ontology/>
		PREFIX dbp: <http://dbpedia.org/property/>

		SELECT ?movement
		WHERE {	<''' + dbpedia_link + '''> dbo:movement|dbp:movement ?movement .}
		''')
	sparql.setReturnFormat(JSON)
	results = sparql.query().convert()
	for binding in results['results']['bindings']:
		movements.append(binding['movement']['value'])
	return movements

def process_author(author, k):
	'''Given an authos soup object, collects information and books 
	for the author if they have at least k sentences throughout their books
	Parameters:
	author - an author soup object
	k - the number of sentences required
	Output: books_list - the list of books for that author
			author_path - the folder where books for that author are saved
	'''
	author_name = author.getText() #get the author name
	authorclean = ''.join(x for x in author_name.split('(')[0] if x.isalpha() or x in ' ') #leave alpha characters and spaces
	author_path = unidecode(clean_name(author_name))

	#if the author is not yet in the db
	if author_path > "Verna Draba" and not db['lab1'].find({'folder':author_path}).count():
		print(authorclean)
		#get the next sibling of the author and proceed only if it is ul
		next_sibling = author.find_next_sibling()
		if next_sibling.name != 'ul':
			return None, None
		list_of_books = author.find_next_sibling("ul") #get a list of books for the author (using the ul sibling of the h2 tag)
		if list_of_books != None: #if there is a list of books
			books = list_of_books.findAll("li", {"class" : "pgdbetext"}) #get all books
			if "Anonymous" not in authorclean and "Unknown" not in authorclean: #if author is not Anonymous or Unknown
				books_to_save, k = get_books_up_to_k_sentences(books, k)
				
				#if k has reached 0, save all the books for that author
				if k == 0:
					books_list = []
					for book in books_to_save:
						book_path = save_file(book[0], './lab1_data/'+author_path+'/', book[1]) #write the returned sentence to a file with the book name
						# books_by_author[author_path].append({'name': book[2], 'path':book_path}) #add the book to the respective author in the books_by_author dictionary
						books_list.append({'name': book[2], 'path':book_path})
					
					#get the multilingual abstracts and the literary movements
					abstracts, url_author = get_multilingual_abstracts(authorclean)

					movements = []
					if url_author != None:
						movements = get_literary_movements(url_author)
					#create and author object, record all the related the data and save it to the db
					a = Author(url_author)
					a.name = authorclean.strip()
					a.folder = author_path
					a.books = books_list
					a.abstracts = abstracts
					a.movements = movements
					out = db['lab1'].insert_one(a.__dict__)
					return books_list, author_path
	return None, None


if __name__ == "__main__":
	main_page = 'http://www.gutenberg.org/browse/authors/' # set the main page to http://www.gutenberg.org/browse/authors/
	books_by_author = {} #create an empty dictionary where the books by author will be stored
	# for letter in string.ascii_lowercase: #for each lowercase letter	
	for letter in 'vwxyz': #for each lowercase letter	
		page = requests.get(main_page+letter) #get the page for the letter
		soup = BeautifulSoup(page.content, 'html.parser') #parse the page
		for author in soup.find("div", {"class": "pgdbbyauthor"}).findAll("h2")[1:]: #for each author found (using the h2 tag of the div with class "pddbbyauthor")
			books_list, author_path = process_author(author, 10000) #process the author and get book list and author path
			if books_list:
				books_by_author[author_path] = books_list #add author and book list to books_by_author

	#Print stats at the end of the process
	print('-'*10)
	print('Collected {0} new authors'.format(len(books_by_author)))
	print('DB now containing {0} authors'.format(db['lab1'].find().count()))

