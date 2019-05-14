import matplotlib.pyplot as plt
import numpy as np

# transitions are noted by the following abbreviations
# A = all
# B = Back
# T = Thigh
# N = None
# e.g A -> B = All to only back

# transition order are as follow;
'''
A-T     index   0
A-B             1    
A-N             2
T-A             3
T-B             4
T-N             5
B-A             6
B-T             7
B-N             8
N-B             9
N-T             10
N-A             11
'''

labels = [
    "A-T",
    "A-B",
    "A-N",
    "T-A",
    "T-B",
    "T-N",
    "B-A",
    "B-T",
    "B-N",
    "N-B",
    "N-T",
    "N-A"
]

            #  0 1 2 3 4 5 6 7 8 9 10 11
p1_atle     = [1,0,0,0,1,0,0,0,0,0,0,0]
p2_atle     = [1,1,0,1,0,0,1,0,0,0,0,0]
p1_vegar    = [1,0,0,0,1,0,0,0,0,0,0,0]
p2_vegar    = [1,1,0,1,0,0,1,0,0,0,0,0]

thomas      = [2,3,2,4,0,0,2,0,1,0,2,0]
thomas2     = [0,1,5,4,1,1,2,1,1,0,3,3]
thomas3     = [1,2,1,3,1,1,0,0,2,0,3,1]

sigve       = [0,1,5,4,1,1,2,1,1,0,3,3]
sigve2      = [3,2,0,2,0,1,2,0,1,1,0,0]

shower_atle = [0,0,4,0,0,0,0,0,0,0,0,3]
nshower_paul= [0,0,2,0,0,0,0,0,0,0,0,2]
vegard = [0,0,2,0,0,0,0,0,0,0,0,2]
eivind = [0,0,2,0,0,0,0,0,0,0,0,2]


def add_list_elements_together(*lists):
    res = []

    # iterate trough all the lists
    for col in range(len(lists[0])):
        sum = 0
        for l in lists:
            sum += l[col]
        res.append(sum)

    return res


print("SUM ALL DATASETS CONCATINATED")
# transitions = add_list_elements_together(thomas, thomas2, thomas3, sigve, sigve2, shower_atle, nshower_paul, vegard, eivind)
transitions = add_list_elements_together(p1_atle, p1_vegar, p2_atle, p2_vegar)
print(transitions)


plt.bar(labels, transitions)
plt.xlabel('Transition')
plt.ylabel('# Transitions')
plt.yticks(np.arange(0,26, step=5))
plt.savefig("transitionsReinsveSNT.png")
