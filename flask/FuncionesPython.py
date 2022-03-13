import re
import math
import datetime
import numpy as np
import joblib
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import layers


####################################################################
def procesarDatos(d):
    mouseLogM = []
    vel = []
    angs = []
    lastClick = []
    lastPoint = []
    distDif = []
    clickD = []
    logCountM = []
    clickTotalM = []
    saveNum1 = 0  # salvar onde ficou a última iteração
    correctNum1 = 0  # boolean só para garantir que retomamos na última posição calculada
    saveNum2 = 0
    correctNum2 = 0
    firstClick = 0
    distSlLog = -1
    logCount = 0
    clickTotal = 0
    count3 = 0
    dateFinal = 0
    time2 = 0
    timeConvert = 0
    num = 0
    ang = 0
    start1 = 1
    start3 = 1
    start4 = 1 # diferenciar a primeira iteração
    dist = 0
    dist2 = 0
    dist3 = 0
    distSL = 0
    # TECLADO
    keyLogM = []
    leftSideM = []  # M -> Matriz
    rigthSideM = []
    backSpaceM = []
    leftSide = 0
    rightSide = 0
    backSpace = 0
    saveNum3 = 0
    correctNum3 = 0
    leftSideMatriz = ["§", "±", "1", "2", "3", "4", "5", "!", "#", "@", "€", "$", "%", "Key.tab", "q", "Q", "w", "W", "e", "E", "r", "R", "t", "T",
                    "a", "A", "s", "S", "d", "D", "f", "F", "g", "G", "Key.shift", "Key.caps_lock", "<", ">", "z", "Z", "x", "X", "c", "C", "v", "V", "b", "B", "Key.crtl", "Key.alt", "Key.cmd"]


    #Comprobar si algun dato está vacio 
    if d[1].get("Teclas"):
        cTecla=1
    else:
        cTecla=0

    if d[0].get("Movimientos"):
        cMovimiento=1
    else:
        cMovimiento=0


    # STRESS
    stressLogM = []
    timeGlobal = 0  # Variável Global das iterações, dada pelo stress
    regex = re.compile("\d+[-]\d+[-]\d+|\d+[:]\d+[:]\d+[,]\d+|-?\d+[.]\d+|\d+")
    tiempo=str(d[2])
    lines = tiempo.split('\n')
    for line in lines:
        result = regex.findall(line)
        stressLogM.append(result)

    regex = re.compile(
        "\d+[-]\d+[-]\d+|\d+[:]\d+[:]\d+[,]\d+|-?\d+[.]\d+|\d+|\w*Button[.]left|\w*Button[.]right")
    movimientoaux=d[0]
    movimientoaux=movimientoaux.values()
    movimientoaux=list(movimientoaux)
    movimiento='\n'.join(map(str, *movimientoaux))
    lines = re.split("\n",movimiento)
    for line in lines:
        result = regex.findall(line)
        mouseLogM.append(result)

    regex = re.compile("\d+\-\d+\-\d+|\d+[: ]\d+[:]\d+[,]\d+|[a-zA-Z]+[.]?[a-zA-Z]*[_]?[a-zA-Z]*|[0-9]+|[^\n\']+x03")
    teclasaux=d[1]
    teclasaux=teclasaux.values()
    teclasaux=list(teclasaux)    
    teclas='\n'.join(map(str, *teclasaux))
    lines = re.split("\n",teclas)
    for line in lines:
        result = regex.findall(line)
        keyLogM.append(result)


    date2 = datetime.datetime.strptime(
    stressLogM[0][1], '%H:%M:%S,%f').time()
    timeGlobal = date2.hour * 60 * 60 + date2.minute * \
                60 + date2.second + date2.microsecond / 1000000
    if cMovimiento==1:
        for num, l in enumerate(mouseLogM, start=saveNum1):
            if start1 == 1 and correctNum1 == 1:
                date2 = datetime.datetime.strptime(
                    mouseLogM[num][1], '%H:%M:%S,%f').time()
                time2 = date2.hour * 60 * 60 + date2.minute * \
                    60 + date2.second + date2.microsecond / 1000000
                #diff = time2 - time1
                # estabelece o intervalo de tempo
                if time2 <= timeGlobal and num != len(mouseLogM)-1:
                    logCount += 1
                    dist += math.sqrt((float(mouseLogM[num][2]) - float(mouseLogM[num - 1][2]))**2 + (
                        float(mouseLogM[num][3]) - float(mouseLogM[num - 1][3]))**2)
                    if mouseLogM[num][2] == "0.0" or mouseLogM[num][2] =="0":
                        mouseLogM[num][2] = 0.01
                    # math.fabs -> valor absoluto
                #  print(mouseLogM[num][3],"   ",mouseLogM[num][2])
                    ang += math.degrees(
                        math.fabs(math.tan(float(mouseLogM[num][3]) / float(mouseLogM[num][2]))))
                        #FAAAAAAAAAAAAAAAAAAAAAAALTAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
                    # print(mouseLogM[num][3]+"/"+mouseLogM[num][2])
                    # print (str(ang)+" en "+str(num))
                    # input()
                else:  # termina um ciclo de 15seg e inicia outro
                    time1 = time2  # guardar o ultimo registo aceite, para iniciar um novo ciclo
                    saveNum1 = num
                    correctNum1 = 0
                    vel.append(dist)
                    angs.append(ang)
                    logCountM.append(logCount)
                    dist = 0
                    ang = 0
                    logCount = 0
                    break
            if start1 == 0:  # primeira iteracao
                vel.append(dist)
                angs.append(ang)
                logCountM.append(logCount)
                dist = 0
                ang = 0
                logCount = 0
                start1 = 1
                break
            if correctNum1 == 0:
                num = saveNum1
                correctNum1 = 1

        for num, l in enumerate(mouseLogM, start=saveNum2):
            # print("a"+str(correctNum2)+"este es num comienzo"+str(num))
            # input()
            if start3 == 1 and correctNum2 == 1:
                date2 = datetime.datetime.strptime(
                    mouseLogM[num][1], '%H:%M:%S,%f').time()
                time2 = date2.hour * 60 * 60 + date2.minute * \
                    60 + date2.second + date2.microsecond / 1000000
                # estabelece o intervalo de tempo
                if time2 <= timeGlobal and num != len(mouseLogM)-1:
                    if firstClick == 1:  # iniciar o somatório após o primeiro click
                        dist2 += math.sqrt((float(mouseLogM[num][2]) - float(lastPoint[2])) ** 2 + (float(
                            mouseLogM[num][3]) - float(lastPoint[3])) ** 2)  # somatório das distâncias todas entre dois cliques
                        # guardar o anterior para fazer a soma
                        lastPoint = mouseLogM[num]
                    if len(mouseLogM[num]) > 4:  # significa que fez um clique
                        distSlLog += 1
                        clickTotal += 1
                        if distSlLog == 0:  # guardar o primeiro registo, só precisa de entrar 1x
                            lastClick = l
                            date = datetime.datetime.strptime(
                                l[1], '%H:%M:%S,%f').time()
                            time3 = date.hour * 60 * 60 * 1000 + date.minute * 60 * \
                                1000 + date.second * 1000 + date.microsecond/1000
                            distSlLog += 1
                        if distSlLog == 2:  # sempre que houver um click entra aqui, só no primeiro é que não
                            distSL += math.sqrt((float(mouseLogM[num][2]) - float(lastClick[2]))**2 + (float(
                                mouseLogM[num][3]) - float(lastClick[3]))**2)  # calculo das distâncias entre dois cliques
                            time2 = datetime.datetime.strptime(
                                mouseLogM[num][1], '%H:%M:%S,%f').time()
                            timeConvert = time2.hour * 60 * 60 * 1000 + time2.minute * 60 * 1000 + \
                                time2.second * 1000 + time2.microsecond/1000  # to miliseconds
                            dateFinal += timeConvert - time3
                            lastClick = l
                            time3 = timeConvert
                            distSlLog = 1
                        if count3 == 1:
                            dist3 += dist2
                            dist2 = 0
                            count3 = 0
                        if count3 == 0:
                            lastPoint = mouseLogM[num]
                            count3 = 1  # para no segundo click determinar a distância
                            firstClick = 1  # começa a somar
                else:  # termina um ciclo e inicia outro
                    # verificar o distDif, aparenta nao estar totalmente correto
                    distDif.append(math.fabs(dist3 - distSL))
                    clickD.append(dateFinal)
                    clickTotalM.append(clickTotal)
                    count2 = -1
                    correctNum2 = 0
                    saveNum2 = num
                    dateFinal = 0
                    distSL = 0
                    clickTotal = 0
                    dist3 = 0
                    break
            if start3 == 0:  # primeira iteracao
                distDif.append(dist3 - distSL)
                clickD.append(dateFinal)
                clickTotalM.append(clickTotal)
                dateFinal = 0
                distSL = 0
                clickTotal = 0
                dist3 = 0
                start3 = 1
                break

            if correctNum2 == 0:
                num = saveNum2
                correctNum2 = 1
    else:
        distDif.append(0)
        clickD.append(0)
        clickTotalM.append(0)
        vel.append(0)
        angs.append(0)
        logCountM.append(0)

    # for cq in range (saveNum3, len(keyLogM)):
    if cTecla==1:
        for num in range(saveNum3, len(keyLogM)):
                date2 = datetime.datetime.strptime(keyLogM[num][1], '%H:%M:%S,%f').time()
                time2 = date2.hour * 60 * 60 + date2.minute * 60 + date2.second + date2.microsecond / 1000000
                if time2 <= timeGlobal and num != len(keyLogM)-1:
                    if keyLogM[num][2] == "Key.backspace":
                        backSpace += 1
                    if leftSideMatriz.__contains__(keyLogM[num][2]):
                        leftSide += 1
                    else:
                        rightSide += 1
                else:  # termina um ciclo e inicia outro
                    rigthSideM.append(rightSide)
                        # nº medio de teclas usadas do ld es
                    leftSideM.append(leftSide)
                    backSpaceM.append(backSpace)
                    saveNum3 = num
                    correctNum3 = 0
                    rightSide = 0
                    leftSide = 0
                    backSpace = 0
                    break
    else:
        rigthSideM.append(0)
        leftSideM.append(0)
        backSpaceM.append(0)
    return vel[0], angs[0],logCountM[0], distDif[0], clickD[0], clickTotalM[0], backSpaceM[0], leftSideM[0], rigthSideM[0]
################################model###########################


def estadoEmocion(e):
    if e[0][0]==1:
        resultado="Negativo"
    if e[0][1]==1:
        resultado="Neutro"
    if e[0][2]==1:
        resultado="Positivo"
    return resultado
            

def prediccion(p):
    new_model = keras.models.load_model('/root/Microservicio/flask/modelo_entrenado.h5')
    auxi=np.array([float(p[0]),float(p[1]),float(p[2]),float(p[3]),float(p[4]),float(p[5]),float(p[6]),float(p[7]),float(p[8])])
    predic=auxi.reshape(1,-1)
    prediction = new_model.predict(predic)
    out =prediction.round().astype(int)
    return out

def prediccion2(p):
    knn_from_joblib = joblib.load('/root/Microservicio/flask/modelo_knn_entrenado.pkl')
    auxi=np.array([float(p[0]),float(p[1]),float(p[2]),float(p[3]),float(p[4]),float(p[5]),float(p[6]),float(p[7]),float(p[8])])
    predic=auxi.reshape(1,-1)
    prediction = knn_from_joblib.predict(predic)
    out =prediction.round().astype(int)
    return out


def estadoEmocion2(e):
    if e==1:
        resultado="Negativo"
    if e==2:
        resultado="Neutro"
    if e==3:
        resultado="Positivo"
    return resultado
