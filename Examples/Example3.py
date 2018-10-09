# example 2
A_Loc = [[1,1/2.0,1/5.0],
         [2,1,1/2.0],
         [5,2,1]]

A_Reput = [[1,2,3],
           [1/2.0,1,3/2.0],
           [1/3,2/3,1]]

# transform matrix
def TransMatrix(M):
    N = len(M[0])
    for i in range(N):
        col_sum = sum([c[i] for c in M])
        print(col_sum)
        for j in range(N):
            M[j][i] = M[j][i]/col_sum
    row_sum = []
    for i in range(N):
        row_sum.append(sum(M[i])/N)
    return row_sum

# calculations
print("Location: ")
row_sum = TransMatrix(A_Loc)
#print(A_Loc)
print(row_sum)

print("Reputation: ")
row_sum = TransMatrix(A_Reput)
#print(A_Reput)
print(row_sum)