import pprint,pickle
import matplotlib
# matplotlib.rcParams['backend'] = 'SVG'###！！！矢量图格式svg
import matplotlib.pyplot as plt
import numpy as np
from openpyxl import Workbook

plt.rcParams['font.sans-serif'] = ['SimHei']  # 正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

ReCat_dir='D:/DOWNLOAD/ZX-Optimizers-master/tmp/cifar10_train_10NALAAlexa=0.4b/'
p=open(ReCat_dir+'LOSSandACCURACY.pkl', 'rb')
d=pickle.load(p)
# print(d)
Nlog = len(d)
step = list(range(0, Nlog*100, 100))
# print(step)
print('step=', Nlog*100)

loss=[]
accuracy=[]
convergence=[]
# for l, a in d:
#     loss.append(l)
#     accuracy.append(a)

for l, a, c in d:
    loss.append(l)
    accuracy.append(a)
    convergence.append(c)

print(loss)
print(accuracy)# print(np.shape(loss)) #
print(convergence)

wb = Workbook()#######创建并保存Excel文件
ws = wb.active
ws.append(step)
ws.append(loss)
ws.append(accuracy)
ws.append(convergence)
wb.save(ReCat_dir+'LOSSandACCURACY.xlsx')

ls = loss.copy()
sp = step.copy()
del(ls[0])
del(sp[0])
plt.plot(sp, ls, label="loss")
# plt.plot(step, loss, label="loss")
plt.plot(step, accuracy, label="accuracy")
plt.xlabel('Step')

plt.figure()
plt.plot(step, convergence, label="conbergence")
plt.xlabel('Step')

#plt.savefig('run_results.svg',format='svg')###！！！保存矢量图在…/ZX-Optimizers-master文件夹下
plt.show()
