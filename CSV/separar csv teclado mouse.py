
import pandas as pd
c=0
m=1
aux=5
teclado=[]
mouse=[]
with open('C:/Users/crism/OneDrive/Desktop/Tesis Final/CSV/metrics.txt',"r") as archivo:
    for linea in archivo:
        c = c+1
        aux= 5*m
        if c==aux:
            m=m+1
            teclado.append(linea)
        else:
            mouse.append(linea)
f = open(r'C:\Users\crism\OneDrive\Desktop\Tesis Final\CSV\teclado.txt', "a")
for l in range(0, len(teclado)):
    f.write(str(teclado[l]))
f.close()

f = open(r'C:\Users\crism\OneDrive\Desktop\Tesis Final\CSV\raton.txt', "a")
for l in range(0, len(mouse)):
    f.write(str(mouse[l]))
f.close()