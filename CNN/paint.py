import matplotlib.pyplot as plt


f = open("a.txt", "r")
st = f.read()
st = st.split("#")
#plt.ylim(0, 2.5)

a = 0

for sta in st:
    if sta == "":
        continue
    Acur = eval(sta)
    k = Acur[:]
    a += 1
    plt.plot(range(0, len(k)), k, label=str(a))

plt.show()