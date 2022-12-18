from flask import Flask, render_template, request, flash
import logic

app = Flask(__name__)
app.secret_key = "jesse"


@app.route('/')
def hello():
    return render_template("index.html")


@app.route("/echo", methods=['POST', 'GET'])
def calculate():
    names = [str(request.form["name1"]) or "", str(request.form["name2"]) or "", str(request.form["name3"]) or "",
             str(request.form["name4"]) or "", str(request.form["name5"]) or ""]
    start = str(request.form["start"])
    end = str(request.form["end"])
    interval = str(request.form["interval"])

    df, labels, prices, tickers, rand, color, eff_frontier, weights, info = logic.getdata(names, interval)
    flash("You chose: " + str(tickers) + " from" + start + " to " + end + " with interval of " + interval)

    return render_template("index.html", tables1=[df.tail(10).to_html(index=False, index_names=False)], tables2=[info.to_html()], labels=labels,
                           values=prices, names=tickers, rand=rand, color=color, eff=eff_frontier, weights=weights, allocations=info["Weight"].to_list())
