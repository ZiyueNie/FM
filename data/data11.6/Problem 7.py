import numpy as np
import matplotlib.pyplot as plt
import math
import csv
from beautifultable import BeautifulTable

names=['AAPL','AMKR','TIF','NKE','C','DOV','FB','GOOGL','TM','PG','JPM','KO','MCD','GE','MSFT','NMR','SBUX','TSLA','TWTR','V']
mu=np.zeros(20)
def getData():
    Is=[i for i in range(20)]
    Rows=[]
    for name,i in zip(names,Is):
        with open(name+'.csv') as file:
            reader=csv.reader(file)
            rows=[row[0] for row in reader]
        mu[i]=np.mean([float(row) for row in rows[1:]])
        Rows.append([float(row) for row in rows[1:]])
    Omega=np.cov(np.array(Rows))
    return Omega

def setting(Omega):
    one=np.ones(20).T
    A=np.dot(one,np.dot(np.linalg.inv(Omega),mu))
    B=np.dot(mu,np.dot(np.linalg.inv(Omega),mu))
    C=np.dot(one,np.dot(np.linalg.inv(Omega),one))
    D=B*C-A**2
    g=(B/D)*np.dot(np.linalg.inv(Omega),one)-(A/D)*np.dot(np.linalg.inv(Omega),mu)
    h=(C/D)*np.dot(np.linalg.inv(Omega),mu)-(A/D)*np.dot(np.linalg.inv(Omega),one)
    mu_m=-(np.dot(g.T,np.dot(Omega,h)))/(np.dot(h.T,np.dot(Omega,h)))
    return mu_m

def getw(Omega,mu_f,mu_p,mu):
    one=one=np.ones(20).T
    lam=(mu_p-mu_f)/np.dot(np.dot(mu-mu_f*one,np.linalg.inv(Omega)),mu-mu_f*one)
    w_p=lam*(np.dot(np.linalg.inv(Omega),mu-mu_f*one))
    w_f=1-np.dot(one,w_p)
    return w_p,w_f

def optimalPortfolio(mu,Omega,mu_p):
    one=np.ones(20).T
    A=np.dot(one,np.dot(np.linalg.inv(Omega),mu))
    B=np.dot(mu,np.dot(np.linalg.inv(Omega),mu))
    C=np.dot(one,np.dot(np.linalg.inv(Omega),one))
    D=B*C-A**2
    g=(B/D)*np.dot(np.linalg.inv(Omega),one)-(A/D)*np.dot(np.linalg.inv(Omega),mu)
    h=(C/D)*np.dot(np.linalg.inv(Omega),mu)-(A/D)*np.dot(np.linalg.inv(Omega),one)
    mu_m=-(np.dot(g.T,np.dot(Omega,h)))/(np.dot(h.T,np.dot(Omega,h)))
    return g+mu_p*h

def draw(mu_f,mu_ps,Omega,mu_m):
    sigmas=[]
    for mu_p in mu_ps:
        w=optimalPortfolio(mu,Omega,mu_p)
        sigmas.append(math.sqrt(np.dot(w,np.dot(Omega,w))))
    plt.plot(sigmas,mu_ps)
    plt.xlabel("varince/$\sigma$")
    plt.ylabel("Expected/$\mu_p$")
    sigmas=[]
    mu_ps1=np.linspace(mu_f,1.003,100)
    for mu_p in mu_ps1:
        w_p,w_f=getw(Omega,mu_f,mu_p,mu)
        sigmas.append(math.sqrt(np.dot(w_p,np.dot(Omega,w_p))))
    plt.plot(sigmas,mu_ps)
    plt.xlim(0,0.016)
    plt.show()

mu_f=1+0.62/9000   
Omega=getData()
mu_m=setting(Omega)
mu_ps=np.linspace(mu_m,1.003,100)
draw(mu_f,mu_ps,Omega,mu_m)


one=np.ones(20).T
lam=1/np.dot(np.dot(mu-mu_f*one,np.linalg.inv(Omega)),mu-mu_f*one)
w_p=lam*(np.dot(np.linalg.inv(Omega),mu-mu_f*one))

mu_p=1/np.dot(one,w_p)+mu_f
weights,_=getw(Omega,mu_f,mu_p,mu)
weights=weights.tolist()
plt.bar(range(len(weights)), weights ,tick_label=names)
plt.show()
weights1=[weight*50 for weight in weights]
leverage=sum((np.array(weights))[np.array(weights)>0])
money=dict(zip(names,weights1))
table = BeautifulTable()
table.column_headers = ["name", "money"]
for item in money:
	table.append_row([item,money[item]])
