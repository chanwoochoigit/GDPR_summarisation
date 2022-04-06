from flask import Flask, request, jsonify
from cluster import take_input
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
model = SentenceTransformer('snt_tsfm_model/')
model.encode('random query')

@app.route('/')
def queryHandler():
	url = request.args.get('url')
	n_best = request.args.get('n_best')
	return jsonify(take_input(int(n_best), url))


app.run(host='0.0.0.0', port=8080)