from matplotlib import pyplot as plt
import os
import numpy as np
 
file_path_data = "assets/Result"
groupTestCases =  os.listdir(r'{}'.format(file_path_data))
namefile = [i.split('.')[0] for i in os.listdir(r'{}'.format(file_path_data+'/'+groupTestCases[0]))]
Runtime = []
listName = []
dataset = {
    'n00050':(),
    'n00100':(),
    'n00200':(), 
    'n00500':(),
    'n01000':(), 
    'n02000':(), 
    'n05000':(), 
    'n10000':()
}
for name in namefile:
    n = name.split('.')[0].split('_')[0]
    listName.append(n)
    for groupCase in groupTestCases:
        path_result = "./assets/Result/{}/{}.txt".format(groupCase,name)
        with open(path_result,'r') as f:
            file = f.read().splitlines()
            runtime = round(float(file[2].split(' ')[1]),2)
            dataset[n] = dataset[n] +  (runtime,)

groupTest = tuple([x[:8] for x in groupTestCases])
listTest = dataset

x = np.arange(len(groupTest))  # the label locations
width = 0.1  # the width of the bars
multiplier = -3

fig, ax = plt.subplots(figsize=(10, 4), layout="constrained",)
for attribute, measurement in listTest.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute,align='center')
    # ax.bar_label(rects)
    multiplier += 1

ax.set_ylabel('Giây(s)')
ax.set_title('Thời gian thực thi thuật toán Snapsack(timelimit:180s)')
ax.set_xticks(x + width, groupTest)
ax.legend(loc='upper left')
ax.set_ylim(0, 300)

plt.show()