from flask import Flask, render_template, request
import logic

app = Flask(__name__)
app.secret_key = "jesse"

@app.route('/')
def hello():
    return render_template("start_page.html")


@app.route("/echo", methods=['POST', 'GET'])
def calculate():
    test = request.referrer
    names = [str(request.form["name1"]) or "", str(request.form["name2"]) or "", str(request.form["name3"]) or "",
             str(request.form["name4"]) or "", str(request.form["name5"]) or ""]
    start = str(request.form["start"])
    end = str(request.form["end"])
    interval = str(request.form["interval"])

    names_wo_empty = list(filter(None, names))
    names_wo_empty.sort()
    tickers = logic.check_tickers(names)

    if request.form["page"] == "start_page":
        html = "start_page.html"
    else:
        html = "index_error_page.html"

    temp = []
    if names_wo_empty != tickers or len(names_wo_empty) < 2:
        for element in names_wo_empty:
            if element not in tickers:
                temp.append(element)
        return render_template(html, error=temp, len=len(names_wo_empty))

    df, labels, prices, tickers, rand, color, eff_frontier, weights, info, yearly_returns, year_dates, target_returns = logic.getdata(names, interval)

    return render_template("index.html", tables1=[df.tail(10).to_html(index=False, index_names=False)],
                           tables2=[info.to_html()], labels=labels, values=prices, names=tickers, rand=rand,
                           color=color, eff=eff_frontier, weights=weights, allocations=info["Weight"].to_list(),
                           yearly_returns=yearly_returns, year_dates=year_dates, target_returns=target_returns, error=[])


@app.errorhandler(400)
def bad_request(e):
    return render_template("start_page.html")