from pymongo import MongoClient

MongoHost = 'localhost'
MongoPort = 27017

client = MongoClient('mongodb://'+MongoHost+':'+str(MongoPort)+'/lab1')
db=client['lab1']

def get_book_titles_by_author(author_name):
	'''Given an author name, retrieve all the books by that
	author from the database
	Parameters: author name - a string with the name of the author
	Output: either a list of books or None
	'''
	books = []
	author = db['lab1'].find_one({'name':author_name})
	if author != None:
		for book in author['books']:
			books.append(book['name'])
		return books
	return None

def get_sorted_author_list():
	'''Return the sorted author list
	Parameters: no parameters
	Output: either a list of author names
	'''
	return sorted(author['name'] for author in db['lab1'].find())

def number_of_books_by_author(author_name):
	'''Given an author name, retrieve the number of books by that author
	Parameters: author name - a string with the name of the author
	Output: either a number or None
	'''
	author = db['lab1'].find_one({'name':author_name})
	if author != None:
		return len(author['books'])
	return None

def get_book_files_for_author(author_name):
	'''Given an author name, retrieve all the books paths for that
	author from the database
	Parameters: author name - a string with the name of the author
	Output: either a list of file paths or None
	'''
	paths = []
	author = db['lab1'].find_one({'name':author_name})
	if author != None:
		for book in author['books']:
			paths.append(book['path'])
		return paths
	return None

if __name__=="__main__":
	# print(get_book_titles_by_author('Cabell James Branch'))
	# print(get_sorted_author_list())
	# print(number_of_books_by_author('Cabell James Branch'))
	print(get_book_files_for_author('Cabell James Branch'))