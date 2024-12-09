input=open("injective.txt")
output=open("pyjective.txt","w")

state=[]

for i in input:
    if i[0]in ["/"," ","\n"]:
        continue
    i=i.split("/")[0]
    name,act=i.split("=")
    name =name.strip(" ")
    if not name in state:
        state.append(name)
    s='x['+str(state.index(name))+"]["
    for j in act.split("+"):
        j=j.strip(" ").strip(";").strip("\n").strip(";")
        acr,nxt=j.split(".")
        if not nxt in state:
            state.append(nxt)
        s1=s+str(state.index(nxt))+"]="
        s2="st"+s1[1:]
        acr=acr[1:len(acr)-1].split(",")
        s2+='"'+acr[0]+'"\n'
        s1+=acr[1]
        output.write(s2+s1+"\n")
            

output.write("\nstete="+str(state)) 
    
    
    