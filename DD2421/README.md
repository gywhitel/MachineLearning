# Lab 1 Decision tree
## Assignment 0
> Q: Motivate which problem is the most difficult for a decision tree algorithm to learn.

The MONK-2 problem will be the most difficult, because we have to rely on all of the attributes.

## Assignment 1
```python
import dtree
import monkdata

entropy = [dtree.entropy(monkdata.monk1),
         dtree.entropy(monkdata.monk2),
        dtree.entropy(monkdata.monk3)]

for i in range(0,3):
    print(entropy[i])
```
| Dataset | Entropy            |
| ------- | ------------------ |
| Monk1   | 1.0                |
| Monk2   | 0.957117428264771  |
| Monk3   | 0.9998061328047111 |

## Assignment 2
Entropy for a uniform distribution (die) is larger than it is for non-uniform distribution (fake die).

## Assignment 3
![](https://i.imgur.com/niZZTxY.png)
## Assignment 4
>Q: For splitting we choose the attribute that maximizes the information gain, Eq.3. Looking at Eq.3 how does the entropy of the subsets, Sk, look like when the information gain is maximized?

Sk subset will be the minimum entropy for the best gain

![](https://i.imgur.com/09Eoppm.png)

>How can we motivate using the information gain as a heuristic for picking an attribute for splitting?

We want to minimize the entropy of the full tree, if we choose the best information gain as our root, we will decrease the entropy as much as we can (sometimes, this will lead to a misplaced choice, for example a5 is arguably more important for MONK-3, but a2 is classified as best information gain.) - this attribute gives us the most information available about the test.

>Think about reduction in entropy after the split and what the entropy implies.

It's more predictable after we remove it, because that's what lower entropy means. (Fake die, lower entropy)

## Assignment 5
> Explain the results you get for the training and test datasets.

![](https://i.imgur.com/NJLR0FX.png)

MONK-1 and MONK-3 have higher information gain, and thus they are easier to build correctly for a training set, when we try the actual data on these trees the results should be much better, which is also shown in the data above.

## Assignment 6
> Explain pruning from bias-variance trade-off perspective.

We want to catch bad branches because they can seriously affect our decision trees effectivity. 

![](https://i.imgur.com/e1goF0o.png)

Bottom right branch, should be FFT, could be avoided by pruning.

## Assignment 7
![](https://i.imgur.com/7RTh6CQ.png)

```python
import monkdata as m
import random
from matplotlib import pyplot as pp
import statistics
import dtree

def partition(data, fraction):
    ldata = list(data)
    random.shuffle(ldata)
    breakPoint = int(len(ldata) * fraction)
    return ldata[:breakPoint], ldata[breakPoint:]


def main1():
    partitionset = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    best_scores = []
    avg_scores = []
    variances = []
    for p in partitionset:
        all_scores = []
        best = 0
        for x in range(100):
            monk1train, monk1val = partition(m.monk1, p)
            initial_tree = dtree.buildTree(monk1train, m.attributes)
            initial_score = dtree.check(initial_tree, monk1val)
            max_score = 0

            while True:
                tree_to_handle = dtree.allPruned(initial_tree)
                max_in_while = 0
                best_tree_in_while = 0
                for a in tree_to_handle:
                    a_score = dtree.check(a, monk1val)
                    if a_score > max_score:
                        max_in_while = a_score
                        best_tree_in_while = a

                if max_in_while < max_score:
                    break
                elif max_in_while > max_score:
                    max_score = max_in_while
                    initial_tree = best_tree_in_while

            all_scores.append(max_score)
            if initial_score > best:
                best = initial_score

        best_scores.append(best)
        avg_scores.append(sum(all_scores) / len(all_scores))
        variances.append(statistics.variance(all_scores))

    print("Best")
    print(best_scores)
    print("Mean")
    print(avg_scores)
    print("Variance")
    print(variances)

    fig, axs = pp.subplots(1, 2, figsize=(5, 5))

    axs[0].plot(partitionset, avg_scores, "r")
    axs[0].set_xlabel("Fraction")
    axs[0].set_ylabel("Average accuracy")

    axs[1].plot(partitionset, variances, "b")
    axs[1].set_xlabel("Fraction")
    axs[1].set_ylabel("Variance")

    axs[0].axis([0.3, 0.8, 0.5, 1])

    for i, j in zip(partitionset, avg_scores):
        axs[0].annotate("{:.4f}".format(j), xy=(i, j))
    for i, j in zip(partitionset, variances):
        axs[1].annotate("{:.6f}".format(j), xy=(i, j))

    pp.show()

main1()
```
---
# Lab 2 Support Vector Machine

## 1.Move the clusters around to make it easier or harder for the classifier to find a decent boundary. Pay attention to when the qt function prints an error message that it can not find a solution

![](https://i.imgur.com/yJM1NJP.png)

Linear, unsolvable (Everything is on a straight line)

![](https://i.imgur.com/Vj8Bx3z.png)

Polynomial, unsolvable?, p = 2, test generated with great variance (V = 1)

![](https://i.imgur.com/NZICYHJ.png)

Radial, generated with overlapping (V=1)

```python
classA = numpy.concatenate((numpy.random.randn(10,2) * V + [1.5,0.5] , numpy.random.rand(10,2) * V + [-1.5, 0.5]))
classB = numpy.random.randn(20,2) * V + [1.5, 0.5]
```



# Lab 3 BAYESIAN LEARNING AND BOOSTING
```python
import numpy

class data:
    'Features and labels'

    def __init__(self, feature, label):
        self.feature = feature
        self.label = label


# Read data from txt files
irisx = []
irisy = []
with open('D:\Python\MachineLearning\Lab3\irisX.txt') as f:
	for vector in f:
		irisx.append(vector.strip().split('\n'))

with open('D:\Python\MachineLearning\Lab3\irisY.txt') as f:
	for vector in f:
		irisy.append(vector.strip().split('\n'))

# len(irisX)*2
iris = []
for i in range(0,len(irisx)):
    iris.append(data(irisx[i],irisy[i]))
```

