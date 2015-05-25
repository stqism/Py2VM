from flask import Flask
app = Flask(__name__)

@app.route("/")
def hello():
    return "Py2VM web rewrite WIP, source: https://github.com/stqism/py2vm"

if __name__ == "__main__":
    app.run(host= '0.0.0.0')
