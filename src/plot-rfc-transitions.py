import matplotlib.pyplot as plt


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

thomas      = [2,3,2,4,0,0,2,0,1,0,2,0]
thomas2     = [0,1,5,4,1,1,2,1,1,0,3,3]
thomas3     = [1,2,1,3,1,1,0,0,2,0,3,1]

sigve       = [0,1,5,4,1,1,2,1,1,0,3,3]
sigve2      = [3,2,0,2,0,1,2,0,1,1,0,0]

shower_atle = [0,0,4,0,0,0,0,0,0,0,0,3]
nshower_paul= [0,0,2,0,0,0,0,0,0,0,0,2]


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
transitions = add_list_elements_together(thomas, thomas2, thomas3, sigve, sigve2, shower_atle, nshower_paul)
print(transitions)


plt.bar(labels, transitions)
plt.savefig("transitionPlotTest.png")
