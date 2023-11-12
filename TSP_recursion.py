from itertools import permutations

points = {"A":0, "B":1, "C":2, "D":3}
dist = [[0, 20, 42, 35],
        [20, 0, 30, 34],
        [42, 30, 0, 12],
        [35, 34, 12, 0]]

def calc_dist(route):

    prev_r = None
    final_dist = 0
    for i, r in enumerate(route):
        if i==0:
            prev_r = r
        else:
            final_dist = final_dist+dist[points[prev_r]][points[r]]

    return final_dist

all_possibles = [''.join(p) for p in permutations('BCD')]
all_possibles = ["A"+p+"A" for p in all_possibles]

min_dist=10000
for p in all_possibles:
   d = calc_dist(p)
   if d<min_dist:
       min_dist = d

print(min_dist)


