import sys
from utils import take_input
# from analyse_document_bert import analyse
from analyse_document_transformers import analyse

def main(json_string):
	return analyse(json_string)

if __name__ == '__main__':
	with open('frame.json') as file:
		json_string = file.read()
	print(main(json_string))
