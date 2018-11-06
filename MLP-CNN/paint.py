import matplotlib.pyplot as plt


f = open("acc.txt", "r")
st = f.read()
st = st.split("#")
plt.ylim(0.94, 0.985)


def change(k):
    a = []
    for i in k:
        a.append(float(i))
    return a


k = change(eval(st[0]))
plt.plot(range(0, len(k)), k, label='MLP 0.0')
k = change(eval(st[1]))
plt.plot(range(0, len(k)), k, label='MLP 0.3')
k = change(eval(st[2]))
plt.plot(range(0, len(k)), k, label='MLP 0.5')
k = change(eval(st[3]))
plt.plot(range(0, len(k)), k, label='CNN 0.0')
k = change(eval(st[4]))
plt.plot(range(0, len(k)), k, label='CNN 0.3')
k = change(eval(st[5]))
plt.plot(range(0, len(k)), k, label='CNN 0.5')

plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()

plt.show()