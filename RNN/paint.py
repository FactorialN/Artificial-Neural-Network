import numpy as np
import matplotlib.pyplot as plt

f = open("paintdata.txt", "r")
s = f.read()
s = s.split('epoch')[1:]
a = 0
tr_loss = []
tr_acc = []
ts_loss = []
ts_acc = []
for x in s:
    a = a + 1
    if a % 2 == 1:
        continue
    x = x.split(' ')[1:]
    tr_loss.append(eval(x[2]))
    tr_acc.append(eval(x[4].lstrip('[').rstrip('\n').rstrip(']')))
    ts_loss.append(eval(x[14]))
    ts_acc.append(eval(x[16].lstrip('[').rstrip('\n').rstrip(']')))


plt.plot(tr_acc, label='training acc')
plt.plot(ts_acc, label='test acc')
plt.xlabel('epoch')
plt.legend()
plt.show()