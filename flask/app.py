from flask import Flask, render_template, url_for, request, jsonify #Importa libreria de flask
from flask_cors import CORS #Importa libreria flask_cors para el intercambio de datos entre servidores
from FuncionesPython import procesarDatos, prediccion, estadoEmocion
#######################################################################
app = Flask(__name__) #Crea la app
CORS(app) #Le da permiso de intercambio de datos entre servidores a la app
@app.route('/') #asigan ruta raiz
def index(): #Crea la función dentro de la ruta raiz
  return render_template('index.html')  #retorna el index encontrado en la carpeta template
@app.route('/datos', methods=['POST', 'GET']) #Crea una ruta al servidor hacia /datos que recibe como metodos POST y GET
def processEmotion(): #Función para procesar las emociones
  if request.method == "POST": #Si la solicitud es post entonces asigna el valor de los datos a la variable "datos"
    datos = request.get_json()
 ######################################################################
    pd=procesarDatos(datos) #Los datos se envian a la funcion procesarDatos y retorna los datos procesados
    pr=prediccion(pd) #Los datos procesados se envian a la función importada predicción que los cargará al modelo para retornar una matriz de predicción
    print(pd)
    print(pr)
    ee=estadoEmocion(pr)#Se envia la matriz de predicción a la funcion estadoEmocion para asignarle una emoción 
    print(ee)
#######################################################################
  results = {"Emocion":ee,"Prediccion":str(pr),"Data":str(pd)} #Se asigna la emocion a un diccionario
  return jsonify(results) #Se retorna la emoción hacia la dirección de la solicitud
if __name__ == "__main__":
  app.run(host="0.0.0.0",port="5000",debug=True) #ejecuta la aplicación
