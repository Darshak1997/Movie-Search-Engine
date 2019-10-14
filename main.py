# -*- coding: utf-8 -*-`
"""
Created on Wed Oct  9 20:58:06 2019

@author: Darshak
"""

from flask import Flask, render_template, request
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
    res, highlight=text_search_1.get_results(search_query)
    print(res[0:5])
    print(highlight)
    return render_template('result.html',result=res[0:10], highlight_text = highlight)

if __name__ == '__main__':
   application.run(use_reloader=False)