from flask import Flask, render_template, request, flash
import logic

app = Flask(__name__)
app.secret_key = "jesse"


@app.route('/')
def hello():
    return render_template("index.html")


@app.route("/echo", methods=['POST', 'GET'])
def calculate():
    names = [str(request.form["name1"]), str(request.form["name2"]), str(request.form["name3"]),
             str(request.form["name4"]), str(request.form["name5"])]
    start = str(request.form["start"])
    end = str(request.form["end"])
    interval = str(request.form["interval"])
    df, labels, prices, tickers, rand, sharpe = logic.getdata(names, interval)
    flash("You chose: " + tickers + " from" + start + " to " + end + " with interval of " + interval)

    return render_template("index.html", tables=[df.tail(10).to_html(classes='data')], titles=df.columns.values,
                           labels=labels, values=prices, names=names, rand=rand, color=sharpe)
