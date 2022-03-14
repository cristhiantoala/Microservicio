from flask import Flask, render_template, url_for, request, jsonify
# coding=utf-8
import re
import math
import datetime
import pymysql
from flask_cors import CORS
from FuncionesPython import procesarDatos, prediccion, prediccion2, estadoEmocion, estadoEmocion2
#######################################################################
app = Flask(__name__)
CORS(app)
@app.route('/')
def index():
  return render_template('index.html')
@app.route('/datos', methods=['POST', 'GET'])
def processEmotion(): 
  if request.method == "POST":
    datos = request.get_json()
 ######################################################################
    pd=procesarDatos(datos)
    pr=prediccion(pd)
    print(pd)
    print(pr)
    ee=estadoEmocion(pr)
    print(ee)
#######################################################################
  results = {"Emocion":ee}
  return jsonify(results)
if __name__ == "__main__":
  app.run(host="0.0.0.0",debug=True)
