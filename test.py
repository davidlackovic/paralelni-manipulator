from finished import funkcije_v3 as fun
import numpy as np

#Dejanski parametri
b = 0.071014 # m
p = 0.223723 # m
l_1 = 0.2 # m
l_2 = 0.23481 # m

out = np.array(fun.izracun_kotov(b, p, l_1, l_2, 0.3, 0, 10))
print(-out*4*16/1.8/100)
