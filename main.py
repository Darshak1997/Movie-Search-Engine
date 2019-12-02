# -*- coding: utf-8 -*-`
"""
Created on Wed Oct  9 20:58:06 2019

@author: Darshak
"""

from flask import Flask, render_template, request, jsonify
import timeit
application = Flask(__name__)
application.debug = True
@application.route('/')
def index():
    return render_template('index.html')

@application.route('/search/', methods=['GET', 'POST'])
def query_search():
    search_query = request.args.get('query','')
    print("# Our Keyword(s): ", search_query)
    
    import text_search_1
    start = timeit.default_timer()
    res, highlight=text_search_1.get_results(search_query)
#    print(res[0:5])
#    print(highlight)
    stop = timeit.default_timer()
    print('Time: ', stop - start)  
    return render_template('result.html',result=res[0:10], highlight_text = highlight)

@application.route('/image/', methods=['GET', 'POST'])
def query_image():
    search_query = request.args.get('query','')
#    import tfidf
    import image_tf_idf
    result_final = image_tf_idf.get_results(10,str(search_query))
#    print(result_final)
    return render_template('Image_Search.html',result=result_final, query = search_query)

@application.route('/classify/', methods=['GET', 'POST'])
def bar():
    search_query = request.args.get('query','')
    import classifier_OG_train
    res1, percentage=classifier_OG_train.get_results(search_query)
    print("Result: ", res1)
    bar_labels = res1
    print(type(bar_labels))
    bar_values = percentage
    return render_template('bar_chart.html', title='Movie Genres', max=100, labels=bar_labels, values=bar_values)


if __name__ == '__main__':
   application.run(use_reloader=False)