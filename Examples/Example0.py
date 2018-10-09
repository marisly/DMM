# example 0
class Node:
    def __init__(self):
        self.weight = 1
        self.score = []

alternatives = ('A','B','C')
root = Node()

location = Node()
location.weight = 0.17
location.score = [0.129,0.277,0.594]

reputation = Node()
reputation.weight = 0.83
reputation.score = [0.545, 0.273, 0.182]

root.score.append(location)
root.score.append(reputation)

def ComputeScore(node,i):
    child = node.score[0]
    if isinstance(child, float):
        # print(node.score[i]," x ",node.weight)
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