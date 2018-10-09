# example 1
class Node:
    def __init__(self):
        self.weight = 1
        self.score = []

alternatives = ('A','B','C')

location_m = Node()
location_m.weight = 0.17
location_m.score = [0.129,0.277,0.594]

reputation_m = Node()
reputation_m.weight = 0.83
reputation_m.score = [0.545, 0.273, 0.182]

location_j = Node()
location_j.weight = 0.3
location_j.score = [0.2,0.3,0.5]

reputation_j = Node()
reputation_j.weight = 0.7
reputation_j.score = [0.5, 0.2, 0.3]

martin = Node()
martin.weight = 0.0
martin.score = [location_m,reputation_m]

jane = Node()
jane.weight = 1.0
jane.score = [location_j,reputation_j]

root = Node()
root.score = [martin,jane]

def ComputeScore(node,i):
    child = node.score[0]
    if isinstance(child, float):
        return node.score[i]*node.weight
    else:
        return node.weight*sum([ ComputeScore(c,i) for c in node.score])

N = len(alternatives) # number of alternatives
score = []

for i,a in enumerate(alternatives):
    a_score = sum( ComputeScore(child,i) for child in root.score )
    score.append((a_score,a))

print("Total score: ", score)
best = sorted(score,key=lambda x:-x[0])[0]
print("Best University: {0} with rate {1:.2f}".format(best[1],best[0]) )