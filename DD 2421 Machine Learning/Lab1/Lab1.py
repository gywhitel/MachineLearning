import dtree
import monkdata


entropy = [dtree.entropy(monkdata.monk1),
         dtree.entropy(monkdata.monk2),
        dtree.entropy(monkdata.monk3)]

for i in range(0,3):
    print(entropy[i])
