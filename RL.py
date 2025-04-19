import numpy as np
rows=3
cols=4
#initialization
grid=np.asmatrix(([-0.01,-0.01,-0.01,1],[-0.01,0,-0.01,-1],[-0.01,-0.01,-0.01,-0.01]))
actions=np.asmatrix(([-1,0],[1,0],[0,1],[0,-1]))
dirn={'N':[-1,0],'E':[0,1],'W':[0,-1],'S':[1,0]}
print("Original grid :\n"+str(grid))
sucprob=0.8
failprob=0.1
policy = np.full((rows, cols), ' ', dtype='<U2')
#G->Goal L->lose X->wall
policy[0,3]='G'
policy[1,3]='L'
policy[1,1]='X'
#gamma=np.random.rand()
gamma=0.9
V=np.zeros((rows,cols))
V[0,3]=1
V[1,3]=-1
itr=100
#value iteration function
while(itr>0):
    check=np.copy(V)
    for r in range(rows):
        for c in range(cols):
            if([r,c]==[0,3] or [r,c]==[1,3] or [r,c]==[1,1]):
                continue
            max_v=float(-99999999)
            for a in dirn:
                best_action=a
                move=dirn[a]
                v=grid[r,c]
                for not_a in dirn:
                    if(a!=not_a):
                        if(a=='N'):
                            if(not_a!='S'):
                                nr=r+dirn[not_a][0]
                                nc=c+dirn[not_a][1]
                                if(nr>=0 and nr<3 and nc>=0 and nc<4):
                                    v+=gamma*V[nr,nc]*failprob
                                else:
                                    v+=gamma*V[r,c]*failprob
                        if(a=='E'):
                            if(not_a!='W'):
                                nr=r+dirn[not_a][0]
                                nc=c+dirn[not_a][1]
                                if(nr>=0 and nr<3 and nc>=0 and nc<4):
                                    v+=gamma*V[nr,nc]*failprob
                                else:
                                    v+=gamma*V[r,c]*failprob
                        if(a=='S'):
                            if(not_a!='N'):
                                nr=r+dirn[not_a][0]
                                nc=c+dirn[not_a][1]
                                if(nr>=0 and nr<3 and nc>=0 and nc<4):
                                    v+=gamma*V[nr,nc]*failprob
                                else:
                                    v+=gamma*V[r,c]*failprob
                        if(a=='W'):
                            if(not_a!='E'):
                                nr=r+dirn[not_a][0]
                                nc=c+dirn[not_a][1]
                                if(nr>=0 and nr<3 and nc>=0 and nc<4):
                                    v+=gamma*V[nr,nc]*failprob
                                else:
                                   v+=gamma*V[r,c]*failprob
                    else:
                        nr=r+dirn[a][0]
                        nc=c+dirn[a][1]
                        if(nr>=0 and nr<3 and nc>=0 and nc<4):
                            v+=gamma*V[nr,nc]*sucprob
                        else:
                            v+=gamma*V[r,c]*sucprob
                    max_v=max(max_v,v)
            check[r,c]=max_v
    V=check
    itr-=1
#policy iteration function
for r in range(rows):
    for c in range(cols):
        if (r, c) in [(0, 3), (1, 3)] or grid[r, c] == 0:
            policy[r, c] = 'G' if (r, c) == (0, 3) else 'L' if (r, c) == (1, 3) else 'X'
            continue
        
        best_action = None
        best_value = float("-inf")
        
        for a in dirn:
            v = 0
            
            
            for not_a in dirn:
                    if(a!=not_a):
                        if(a=='N'):
                            if(not_a!='S'):
                                nr=r+dirn[not_a][0]
                                nc=c+dirn[not_a][1]
                                if(nr>=0 and nr<3 and nc>=0 and nc<4):
                                    v+=gamma*V[nr,nc]*failprob
                                else:
                                    v+=gamma*V[r,c]*failprob
                        if(a=='E'):
                            if(not_a!='W'):
                                nr=r+dirn[not_a][0]
                                nc=c+dirn[not_a][1]
                                if(nr>=0 and nr<3 and nc>=0 and nc<4):
                                    v+=gamma*V[nr,nc]*failprob
                                else:
                                    v+=gamma*V[r,c]*failprob
                        if(a=='S'):
                            if(not_a!='N'):
                                nr=r+dirn[not_a][0]
                                nc=c+dirn[not_a][1]
                                if(nr>=0 and nr<3 and nc>=0 and nc<4):
                                    v+=gamma*V[nr,nc]*failprob
                                else:
                                    v+=gamma*V[r,c]*failprob
                        if(a=='W'):
                            if(not_a!='E'):
                                nr=r+dirn[not_a][0]
                                nc=c+dirn[not_a][1]
                                if(nr>=0 and nr<3 and nc>=0 and nc<4):
                                    v+=gamma*V[nr,nc]*failprob
                                else:
                                    v+=gamma*V[r,c]*failprob
                    else:
                        nr=r+dirn[a][0]
                        nc=c+dirn[a][1]
                        if(nr>=0 and nr<3 and nc>=0 and nc<4):
                            v+=gamma*V[nr,nc]*sucprob
                        else:
                            v+=gamma*V[r,c]*sucprob
            if v > best_value:
                best_value = v
                best_action = a
        
        policy[r, c] = best_action
         
V=np.round(V,2)
print("Optimal Value function for each state: \n"+str(V))
print("Optimal policy for each state: \n" +str(policy))
                    
            

