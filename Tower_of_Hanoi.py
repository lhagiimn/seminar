
A = [4, 3, 2, 1]
B = []
C = []
last_disc = None
#while C != [4, 3, 2, 1]:
for i in range(20):
    print(A, B, C)
    if len(C)>2 and (C[-1]!=4 and C[-2]!=3):
        if len(A) == 0 or (C[-1] < A[-1] and last_disc != "A"):
            print('fifth')
            A.append(C[-1])
            C.remove(C[-1])
            last_disc = "A"
        elif len(B) == 0 or (C[-1] < B[-1] and last_disc != "C"):
            print('sixth')
            B.append(C[-1])
            C.remove(C[-1])
            last_disc = "B"

    elif  len(B)==0 or (len(A)!=0 and A[-1] < B[-1] and last_disc!="A"):
        print('first')
        B.append(A[-1])
        A.remove(A[-1])
        last_disc="B"
    elif len(C)==0 or (len(A)!=0 and A[-1] < C[-1] and last_disc!="A") and len(A)!=0 :
        print('second')
        C.append(A[-1])
        A.remove(A[-1])
        last_disc = "C"
    elif  len(C)==0 or (B[-1]<C[-1] and last_disc!="B") and len(B)!=0:
        print('third')
        C.append(B[-1])
        B.remove(B[-1])
        last_disc = "C"
    elif  len(A)==0 or (B[-1]<A[-1] and last_disc!="B"):
        print('fourth')
        A.append(B[-1])
        B.remove(B[-1])
        last_disc = "A"
    elif len(A)==0 or (C[-1]<A[-1] and last_disc!="A"):
        print('fifth')
        A.append(C[-1])
        C.remove(C[-1])
        last_disc = "A"
    elif len(B)==0 or (C[-1]<B[-1] and last_disc!="C"):
        print('sixth')
        B.append(C[-1])
        C.remove(C[-1])
        last_disc = "B"

