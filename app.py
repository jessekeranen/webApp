from flask import Flask, render_template, request, flash
import logic

app = Flask(__name__)
app.secret_key = "jesse"


@app.route('/first')
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
    df = logic.getData(name, interval)

    png = logic.plotImage(df)

    return render_template("index.html", tables=[df.to_html(classes='data')], titles=df.columns.values, image=png)