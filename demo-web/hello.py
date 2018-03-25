# -*- coding: utf-8 -*-
from flask import Flask, render_template, request, redirect
app = Flask(__name__)
@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/', methods=['POST', 'GET'])
def upload_file():
   if request.method == 'POST':
      f1 = request.form['text1']
      f2 = request.form['text2']
      if f1 == f2:
          whether_dup = u'являются'
      else:
          whether_dup = u'не являются'
   return render_template('results.html', whether_dup = whether_dup)

if __name__ == '__main__':
    app.run(debug=True)
