from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

k=1	
g = 0.05

T1_original=.3
T2_original=.1

def init(k,g,sigma):
    T1=.3*sigma
    T2=.1*sigma
    T3=.1*sigma
    T4=.1*sigma


    t1=np.zeros(3)
    for i in range(len(t1)):
        t1[i]=k*T1_original
    t2=k*T2_original

    gammas= np.zeros((3,6))
    for i,gi in enumerate(gammas): 
        for j,gj in enumerate(gi): 
            gammas[i][j]=max(1/t1[i],1/(T1+j*g))

    betas=np.zeros(6)
    for i,bi in enumerate(betas):
        betas[i]=max(1/t2,1/(T2+i*g))

    deltas=np.zeros(6)
    for i,di in enumerate(deltas):
        deltas[i]=1/(T3+i*g)
        
    w1s=np.zeros((3,6))
    for i,wi in enumerate(w1s): 
        for j,wj in enumerate(wi): 
            w1s[i][j]=1-np.exp(-(T1/t1[i]))

    w2s=np.zeros(6)
    for i,bi in enumerate(w2s):
        w2s[i]=1-np.exp(-(T2/t2))
    
    n=1000000.0
    pFF=1/3			   #All validators have the same probability to propose
    pF=1/3
    pS=1/3
    #TEST THROUGHPUT MODELLING PROPOSE AND PREVOTE WITH k	   
    # #Test the troughput with k starting from 0.3 to 0.9 step 0.3
    #T1 = 0.3,  T2 = 0.1,  T3 = 0.1,  T4 = 0.1,  g = 0.05

    #PROPOSE SECTION
    #Same rates to Propose
    gamma1_FF=gammas[0][0]        #max (1/k*t1, 1/T1) = 11.111
    w1_FF=w1s[0][0]
    gamma1_F=gammas[1][0]
    w1_F=w1s[1][0]
    gamma1_S=gammas[2][0]
    w1_S=w1s[2][0]
    #PREVOTE TO COMMIT SECTION
    beta_1=betas[0]          #max (1/k*t1, 1/T1) = 33.333
    w2_1=w2s[0]
    delta_1=deltas[0]
    w3_1=0.999999999         
    eta=1/(T4)

    #ROUND 2 RATES DEFINITION
    gamma2_FF=gammas[0][1]        
    w2_FF=w1s[0][1]
    gamma2_F=gammas[1][1]
    w2_F=w1s[1][1]
    gamma2_S=gammas[2][1]
    w2_S=w1s[2][1]
    #PREVOTE TO COMMIT SECTION
    beta_2=betas[1]          
    w2_2=w2s[1]
    delta_2=deltas[1]
    w3_2=0.999999999  

    #ROUND 3 RATES DEFINITION

    gamma3_FF=gammas[0][2]        
    w3_FF=w1s[0][2]
    gamma3_F=gammas[1][2]
    w3_F=w1s[1][2]
    gamma3_S=gammas[2][2]
    w3_S=w1s[2][2]
    #PREVOTE TO COMMIT SECTION
    beta_3=betas[2]          
    w2_3=w2s[2]
    delta_3=deltas[2]
    w3_3=0.999999999 

    #ROUND 4 RATES DEFINITION

    gamma4_FF=gammas[0][3]        
    w4_FF=w1s[0][3]
    gamma4_F=gammas[1][3]
    w4_F=w1s[1][3]
    gamma4_S=gammas[2][3]
    w4_S=w1s[2][3]
    #PREVOTE TO COMMIT SECTION
    beta_4=betas[3]          
    w2_4=w2s[3]
    delta_4=deltas[3]
    w3_4=0.999999999 

    #ROUND 5 RATES DEFINITION
    gamma5_FF=gammas[0][4]        
    w5_FF=w1s[0][4]
    gamma5_F=gammas[1][4]
    w5_F=w1s[1][4]
    gamma5_S=gammas[2][4]
    w5_S=w1s[2][4]
    #PREVOTE TO COMMIT SECTION
    beta_5=betas[4]          
    w2_5=w2s[4]
    delta_5=deltas[4]
    w3_5=0.999999999 

    #ROUND 6 RATES DEFINITION
    gamma6_FF=gammas[0][5]        
    w6_FF=w1s[0][5]
    gamma6_F=gammas[1][5]
    w6_F=w1s[1][5]
    gamma6_S=gammas[2][5]
    w6_S=w1s[2][5]
    #PREVOTE TO COMMIT SECTION
    beta_6=betas[5]          
    w2_6=w2s[5]
    delta_6=deltas[5]
    w3_6=0.999999999999
    
    x=np.zeros((55,55))
    st=[["_" for _ in range(55)] for _ in range(55)]
    st[0][1]="nh"
    x[0][1]=n
    st[1][2]="r"
    x[1][2]=pFF*n
    st[1][3]="r"
    x[1][3]=pF*n
    st[1][4]="r"
    x[1][4]=pS*n
    st[2][5]="p"
    x[2][5]=w1_FF*gamma1_FF
    st[2][6]="p"
    x[2][6]=(1-w1_FF)*gamma1_FF
    st[3][5]="p"
    x[3][5]=w1_F*gamma1_F
    st[3][6]="p"
    x[3][6]=(1-w1_F)*gamma1_F
    st[4][5]="p"
    x[4][5]=w1_S*gamma1_S
    st[4][6]="p"
    x[4][6]=(1-w1_S)*gamma1_S
    st[5][7]="pv"
    x[5][7]=w2_1*beta_1
    st[5][8]="pv"
    x[5][8]=(1-w2_1)*beta_1
    st[6][8]="pv"
    x[6][8]=beta_1
    st[8][9]="pc"
    x[8][9]=delta_1
    st[7][10]="pc"
    x[7][10]=w3_1*delta_1
    st[7][9]="pc"
    x[7][9]=(1-w3_1)*delta_1
    st[10][0]="c1"
    x[10][0]=eta
    st[9][11]="r"
    x[9][11]=pFF*n
    st[9][12]="r"
    x[9][12]=pF*n
    st[9][13]="r"
    x[9][13]=pS*n
    st[11][14]="p"
    x[11][14]=w2_FF*gamma2_FF
    st[11][15]="p"
    x[11][15]=(1-w2_FF)*gamma2_FF
    st[12][14]="p"
    x[12][14]=w2_F*gamma2_F
    st[12][15]="p"
    x[12][15]=(1-w2_F)*gamma2_F
    st[13][14]="p"
    x[13][14]=w2_S*gamma2_S
    st[13][15]="p"
    x[13][15]=(1-w2_S)*gamma2_S
    st[14][16]="pv"
    x[14][16]=w2_2*beta_2
    st[14][17]="pv"
    x[14][17]=(1-w2_2)*beta_2
    st[15][17]="pv"
    x[15][17]=beta_2
    st[17][18]="pc"
    x[17][18]=delta_2
    st[16][19]="pc"
    x[16][19]=w3_2*delta_2
    st[16][18]="pc"
    x[16][18]=(1-w3_2)*delta_2
    st[19][0]="c2"
    x[19][0]=eta
    st[18][20]="r"
    x[18][20]=pFF*n
    st[18][21]="r"
    x[18][21]=pF*n
    st[18][22]="r"
    x[18][22]=pS*n
    st[20][23]="p"
    x[20][23]=w3_FF*gamma3_FF
    st[20][24]="p"
    x[20][24]=(1-w3_FF)*gamma3_FF
    st[21][23]="p"
    x[21][23]=w3_F*gamma3_F
    st[21][24]="p"
    x[21][24]=(1-w3_F)*gamma3_F
    st[22][23]="p"
    x[22][23]=w3_S*gamma3_S
    st[22][24]="p"
    x[22][24]=(1-w3_S)*gamma3_S
    st[23][25]="pv"
    x[23][25]=w2_3*beta_2
    st[23][26]="pv"
    x[23][26]=(1-w2_3)*beta_2
    st[24][26]="pv"
    x[24][26]=beta_3
    st[26][27]="pc"
    x[26][27]=delta_3
    st[25][28]="pc"
    x[25][28]=w3_3*delta_3
    st[25][27]="pc"
    x[25][27]=(1-w3_3)*delta_3
    st[28][0]="c3"
    x[28][0]=eta
    st[27][29]="r"
    x[27][29]=pFF*n
    st[27][30]="r"
    x[27][30]=pF*n
    st[27][31]="r"
    x[27][31]=pS*n
    st[29][32]="p"
    x[29][32]=w4_FF*gamma4_FF
    st[29][33]="p"
    x[29][33]=(1-w4_FF)*gamma4_FF
    st[30][32]="p"
    x[30][32]=w4_F*gamma4_F
    st[30][33]="p"
    x[30][33]=(1-w4_F)*gamma4_F
    st[31][32]="p"
    x[31][32]=w4_S*gamma4_S
    st[31][33]="p"
    x[31][33]=(1-w4_S)*gamma4_S
    st[32][34]="pv"
    x[32][34]=w2_4*beta_4
    st[32][35]="pv"
    x[32][35]=(1-w2_4)*beta_4
    st[33][35]="pv"
    x[33][35]=beta_4
    st[35][36]="pc"
    x[35][36]=delta_4
    st[34][37]="pc"
    x[34][37]=w3_4*delta_4
    st[34][36]="pc"
    x[34][36]=(1-w3_4)*delta_4
    st[37][0]="c4"
    x[37][0]=eta
    st[36][38]="r"
    x[36][38]=pFF*n
    st[36][39]="r"
    x[36][39]=pF*n
    st[36][40]="r"
    x[36][40]=pS*n
    st[38][41]="p"
    x[38][41]=w5_FF*gamma5_FF
    st[38][42]="p"
    x[38][42]=(1-w5_FF)*gamma5_FF
    st[39][41]="p"
    x[39][41]=w5_F*gamma5_F
    st[39][42]="p"
    x[39][42]=(1-w5_F)*gamma5_F
    st[40][41]="p"
    x[40][41]=w5_S*gamma5_S
    st[40][42]="p"
    x[40][42]=(1-w5_S)*gamma5_S
    st[41][43]="pv"
    x[41][43]=w2_5*beta_5
    st[41][44]="pv"
    x[41][44]=(1-w2_5)*beta_5
    st[42][44]="pv"
    x[42][44]=beta_5
    st[44][45]="pc"
    x[44][45]=delta_5
    st[43][46]="pc"
    x[43][46]=w3_5*delta_5
    st[43][45]="pc"
    x[43][45]=(1-w3_5)*delta_5
    st[46][0]="c5"
    x[46][0]=eta
    st[45][47]="r"
    x[45][47]=pFF*n
    st[45][48]="r"
    x[45][48]=pF*n
    st[45][49]="r"
    x[45][49]=pS*n
    st[47][50]="p"
    x[47][50]=w6_FF*gamma6_FF
    st[47][51]="p"
    x[47][51]=(1-w6_FF)*gamma6_FF
    st[48][50]="p"
    x[48][50]=w6_F*gamma6_F
    st[48][51]="p"
    x[48][51]=(1-w6_F)*gamma6_F
    st[49][50]="p"
    x[49][50]=w6_S*gamma6_S
    st[49][51]="p"
    x[49][51]=(1-w6_S)*gamma6_S
    st[50][52]="pv"
    x[50][52]=w2_6*beta_6
    st[50][53]="pv"
    x[50][53]=(1-w2_6)*beta_6
    st[51][53]="pv"
    x[51][53]=beta_6
    st[53][45]="pc"
    x[53][45]=delta_6
    st[52][54]="pc"
    x[52][54]=w3_6*delta_6
    st[52][45]="pc"
    x[52][45]=(1-w3_6)*delta_6
    st[54][0]="c6"
    x[54][0]=eta
    state=['NewHeight', 'Round_1', 'ProposeFF_1', 'ProposeF_1', 'ProposeS_1', 'Prevote_1', 'NilPrevote_1', 'Precommit_1', 'Unsuccess_1', 'Round_2', 'Commit_1', 'ProposeFF_2', 'ProposeF_2', 'ProposeS_2', 'Prevote_2', 'NilPrevote_2', 'Precommit_2', 'Unsuccess_2', 'Round_3', 'Commit_2', 'ProposeFF_3', 'ProposeF_3', 'ProposeS_3', 'Prevote_3', 'NilPrevote_3', 'Precommit_3', 'Unsuccess_3', 'Round_4', 'Commit_3', 'ProposeFF_4', 'ProposeF_4', 'ProposeS_4', 'Prevote_4', 'NilPrevote_4', 'Precommit_4', 'Unsuccess_4', 'Round_5', 'Commit_4', 'ProposeFF_5', 'ProposeF_5', 'ProposeS_5', 'Prevote_5', 'NilPrevote_5', 'Precommit_5', 'Unsuccess_5', 'Round_6', 'Commit_5', 'ProposeFF_6', 'ProposeF_6', 'ProposeS_6', 'Prevote_6', 'NilPrevote_6', 'Precommit_6', 'Unsuccess_6', 'Commit_6']
    for i in range(len(state)):
        x[i][i]=-sum(x[i])
    return x,st,state

x,st,state=init(1,g,1)




def Markov_Steady_State_Prop(p):
    p=p.copy()
    for ii in range(p.shape[0]):
        p[0,ii] = 1
    
    P0 = np.zeros((p.shape[0],1))    
    P0[0] = 1
    return np.matmul(np.linalg.inv(p),P0)


stt=Markov_Steady_State_Prop(np.transpose(x))
print(stt)
print(sum(stt))


def trp(actypes,stt):
    trog=0
    for actype in actypes:
        for i,sti in enumerate(st):
            for j in [k for k,xx in enumerate(sti) if xx==actype]:
                trog+=stt[i]*x[i][j]
    return trog

def trougput(actype,stt):
    trog=0
    for i,sti in enumerate(st):
        for j in [k for k,xx in enumerate(sti) if xx==actype]:
            trog+=stt[i]*x[i][j]
    return trog

'''
acttype=[]
for sti in st:
    for stj in sti:
        if stj not in acttype:
            acttype.append(stj)
acttype.remove("_")
'''
actype=["c2","c3","c4","c5","c6"]
data_c1=[]
data_pv=[]
data_cm=[]
ks=[.1,.3,.5,.7,.9,1,1.1,1.3,1.5,2,3,4,5,6,7]
for i in ks:
    x,st,state=init(i,g,1)
    stt=Markov_Steady_State_Prop(np.transpose(x))
    data_pv.append(trougput("nh",stt))
    data_c1.append(trougput("c1",stt))
    data_cm.append(trp(actype,stt))

   
'''plt.subplot() 
plt.plot(ks,data_pv,color="C0") 
plt.title("block troughput")
plt.show()'''

plt.subplot() 
plt.plot(ks,data_c1,label="Single round",color="C2")  
plt.plot(ks,data_cm,label="Multiple round",color="C3") 
plt.plot(ks,data_pv,color="C0",label="total") 
plt.axhline(y=1/.7,linestyle=":",c="purple",label="estimated performance",xmax=1)
plt.title("throughput as function of k in homogeneous scenario")
plt.ylabel("throughput")
plt.xlabel("K")
plt.legend()
plt.show()
np.savetxt("h.txt",np.array([np.array(data_c1).flatten(),np.array(data_cm).flatten(),np.array(data_pv).flatten()]))


gs=[i/100 for i in range(1,11)]
print(gs)
ks=[.6,.8,1,2,4]
plt.subplot() 
for i,ki in enumerate(ks):
    data_g=[]
    for gi in gs:
        x,st,state=init(ki,gi,1)
        stt=Markov_Steady_State_Prop(np.transpose(x))
        data_g.append(trougput("nh",stt))
        #print(data_g)
    plt.plot(gs,data_g,color="C"+str(i),label="K="+str(ki))
plt.xlabel("g coefficient")
plt.ylabel("trougput") 
plt.axvline(x=.05,c="grey",linestyle=":")
plt.title("throughput as function of timeout increase in homogeneous scenario")
plt.legend()
plt.show()


sigmas=[.1,.3,.5,.7,.9,1,1.1,1.3,1.5,2,3]
ks=[.6,.8,1,2,4,8]
plt.subplot() 
for i,ki in enumerate(ks):
    data_g=[]
    for sigma in sigmas:
        x,st,state=init(ki,g,sigma)
        stt=Markov_Steady_State_Prop(np.transpose(x))
        data_g.append(trougput("nh",stt))
        #print(data_g)
    plt.plot(sigmas,data_g,color="C"+str(i),label="K="+str(ki))
plt.xlabel("coefficient sigma of timeout")
plt.ylabel("trougput") 
plt.axvline(x=1,c="grey",linestyle=":")
plt.title("throughput as function of timeouts in homogeneous scenario")
plt.legend()
plt.show()




