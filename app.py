from flask import Flask, render_template, request, flash
import logic

app = Flask(__name__)
app.secret_key = "jesse"


@app.route('/')
def hello():
    return render_template("index.html")


@app.route("/echo", methods=['POST', 'GET'])
def calculate():
    name = str(request.form["name"])
    start = str(request.form["start"])
    end = str(request.form["end"])
    interval = str(request.form["interval"])
    final = "You chose: " + name + start + end + interval
    flash(final)
    df, labels, values = logic.getData(name, interval)

    return render_template("index.html", tables=[df.tail(10).to_html(classes='data')], titles=df.columns.values, labels=labels, values=values)

