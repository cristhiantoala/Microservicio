import re
import math
import datetime
import numpy as np
datos={'Movimientos': ['2022-2-26 0:47:52,719: Mouse clicked at (524, 186) with Button.left', '2022-2-26 0:47:53,7: Mouse clicked at (524, 186) with Button.left', '2022-2-26 0:47:53,414: Mouse moved to (523, 186)', '2022-2-26 0:47:53,422: Mouse moved to (523, 185)', '2022-2-26 0:47:53,430: Mouse moved to (522, 185)', '2022-2-26 0:47:53,446: Mouse moved to (521, 185)', '2022-2-26 0:47:53,454: Mouse moved to (520, 184)', '2022-2-26 0:47:53,462: Mouse moved to (520, 183)', '2022-2-26 0:47:53,470: Mouse moved to (518, 183)', '2022-2-26 0:47:53,478: Mouse moved to (516, 181)', '2022-2-26 0:47:53,494: Mouse moved to (514, 179)', '2022-2-26 0:47:53,502: Mouse moved to (512, 179)', '2022-2-26 0:47:53,510: Mouse moved to (511, 178)', '2022-2-26 0:47:53,518: Mouse moved to (510, 177)', '2022-2-26 0:47:53,534: Mouse moved to (509, 176)', '2022-2-26 0:47:53,542: Mouse moved to (508, 175)', '2022-2-26 0:47:53,558: Mouse moved to (507, 175)', '2022-2-26 0:47:53,566: Mouse moved to (506, 174)', '2022-2-26 0:47:53,582: Mouse moved to (505, 173)', '2022-2-26 0:47:53,590: Mouse moved to (504, 173)', '2022-2-26 0:47:53,598: Mouse moved to (504, 172)', '2022-2-26 0:47:53,606: Mouse moved to (504, 171)', '2022-2-26 0:47:53,614: Mouse moved to (503, 171)', '2022-2-26 0:47:53,759: Mouse clicked at (503, 171) with Button.left']}, {'Teclas': ['2022-2-26 0:47:55,70: a', '2022-2-26 0:47:55,222: s', '2022-2-26 0:47:55,334: d', '2022-2-26 0:47:55,998: j', '2022-2-26 0:47:56,86: k', '2022-2-26 0:47:56,207: l', '2022-2-26 0:47:56,942: Key.space', '2022-2-26 0:47:57,534: Key.enter', '2022-2-26 0:47:58,326: Key.backspace']}, {'Hora': '2022-2-26 0:47:59,760: cierre_ciclo'}
####################################################################
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

# STRESS
stressLogM = []
timeGlobal = 0  # Variável Global das iterações, dada pelo stress
regex = re.compile("\d+[-]\d+[-]\d+|\d+[:]\d+[:]\d+[,]\d+|-?\d+[.]\d+|\d+")
tiempo=str(datos[2])
lines = tiempo.split('\n')
#print(lines)
for line in lines:
    result = regex.findall(line)
    print(result)
    stressLogM.append(result)

regex = re.compile(
    "\d+[-]\d+[-]\d+|\d+[:]\d+[:]\d+[,]\d+|-?\d+[.]\d+|\d+|\w*Button[.]left|\w*Button[.]right")
#print(datos[0])
movimientoaux=datos[0]
movimientoaux=movimientoaux.values()
movimientoaux=list(movimientoaux)
movimiento='\n'.join(map(str, *movimientoaux))
lines = re.split("\n",movimiento)
for line in lines:
    result = regex.findall(line)
    mouseLogM.append(result)

regex = re.compile("\d+\-\d+\-\d+|\d+[: ]\d+[:]\d+[,]\d+|[a-zA-Z]+[.]?[a-zA-Z]*[_]?[a-zA-Z]*|[0-9]+|[^\n\']+x03")
teclasaux=datos[1]
print(type(teclasaux))
teclasaux=teclasaux.values()
teclasaux=list(teclasaux)    
print(type(teclasaux))
teclas='\n'.join(map(str, *teclasaux))
print(type(teclas))
lines = re.split("\n",teclas)
for line in lines:
    result = regex.findall(line)
    keyLogM.append(result)


date2 = datetime.datetime.strptime(
stressLogM[0][1], '%H:%M:%S,%f').time()
timeGlobal = date2.hour * 60 * 60 + date2.minute * \
            60 + date2.second + date2.microsecond / 1000000
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

# for cq in range (saveNum3, len(keyLogM)):
       
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
            print(keyLogM[num][2])
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
for l in range(0, len(vel)):
    print(str(vel[l])+' '+str(angs[l])+' '+str(logCountM[l])+' '+str(distDif[l])+' '+str(clickD[l])+' '+str(clickTotalM[l])+' '+str(backSpaceM[l])+' '+str(leftSideM[l])+' '+str(rigthSideM[l])+'\n')

################################model###########################
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import layers

new_model = keras.models.load_model('C:/Users/crism/OneDrive/Desktop/PruebasIA/flask/modelo_entrenado.h5')
l=0
auxi=np.array([float(vel[l]),float(angs[l]),float(logCountM[l]),float(distDif[l]),float(clickD[l]),float(clickTotalM[l]),float(backSpaceM[l]),float(leftSideM[l]),float(rigthSideM[l])])
predic=auxi.reshape(1,-1)
prediction = new_model.predict(predic)
out =prediction.round().astype(int)
print(out)