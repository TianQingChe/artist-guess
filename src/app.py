from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from src.serve import get_model_api
from time import sleep
from multiprocessing import Process


app = Flask(__name__)
CORS(app)
model_api = get_model_api()

@app.route('/')
def index():
	return render_template('find-artist.html')


#@app.route('/keep_alive')
def keep_alive():
	while True:
		print("App is alive!!")
		sleep(20)


@app.errorhandler(404)
def url_error(e):
    return """
    Wrong URL!
    <pre>{}</pre>""".format(e), 404


@app.errorhandler(500)
def server_error(e):
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500


@app.route('/find_artist', methods=['POST'])
def api():
	input_url = request.json
	output_data = model_api(input_url)
	response = jsonify(output_data)
	return response


if __name__ == '__main__':
	p = Process(target=keep_alive)
	p.start()
	app.run(debug=False, use_reloader=False)
	p.join()
