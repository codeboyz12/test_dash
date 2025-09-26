from flask import Flask, jsonify

app = Flask(__name__)
@app.route("/", methods=['GET'])
def main():
	print("Have request to /data sending a response.", flush=True)
	return "<center><h1>Hello from Y095</h></center>"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)
