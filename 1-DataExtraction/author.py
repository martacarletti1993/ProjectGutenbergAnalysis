import uuid
from datetime import datetime

# Define Author class.
# Used for creating Author objects
# for upload to MongoDB
class Author(object):
	def __init__(self,url):
		self.name=None
		self.abstracts=None
		self.url=url
		self.folder=None
		self.books=None
		self.movements=None