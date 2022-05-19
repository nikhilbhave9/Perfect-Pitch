arr1 = [int(x) for x in input().split()]
arr2 = [int(x) for x in input().split()]

match = 0
for i in range(len(arr1)):
    if arr1[i] != arr2[i]:
        match += 1
    
print(match/len(arr1))