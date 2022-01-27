import sys
from utils import take_input
from analyse_document_bert import analyse

def main(json_string):
	return analyse(json_string)

if __name__ == '__main__':
	input_json = sys.argv[1]
	print(main(input_json))
