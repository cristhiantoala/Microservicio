
import pandas as pd

read_file = pd.read_csv (r'C:\Users\crism\OneDrive\Desktop\Tesis Final\CSV\raton.txt', delim_whitespace=True, names=["vel", "angs", "logCountM", "distDif","clickD", "clickTotalM", "backSpaceM", "leftSideM", "rigthSideM", "stressLevelM"])
read_file.to_csv (r'C:\Users\crism\OneDrive\Desktop\Tesis Final\CSV\raton.csv', index=None)

readfile = pd.read_csv (r'C:\Users\crism\OneDrive\Desktop\Tesis Final\CSV\raton.csv')
print(readfile)