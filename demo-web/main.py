# -*- coding: utf-8 -*-
from flask import Flask, render_template, request, send_from_directory
from model import form_dataset, mystem_combined, obrabotka, features, classifier
app = Flask(__name__)
@app.route('/')
def hello_world():
   visible = False
   return render_template('index.html')

@app.route('/', methods=['POST', 'GET'])
def upload_file():
   if request.method == 'POST':
      f1 = request.form['text1']
      f2 = request.form['text2']
      df, result = form_dataset(f1, f2)
      df.both = df.both.apply(mystem_combined)
      from model import obrabotka
      obrabotka(df.both)
      from model import features
      geo_class = features(df, result)
      from model import classifier
      answers, predictions = classifier(geo_class)
      if answers[0] == 1:
          whether_dup = u'являются'
          proba = round(predictions[0][1]*100, 2)
      else:
          whether_dup = u'не являются'
          proba = round(predictions[0][0]*100, 2)
      visible = True
   return render_template('index.html', visible = visible, proba = proba, whether_dup = whether_dup)

if __name__ == '__main__':
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    app.run(host='0.0.0.0', port=8000)

