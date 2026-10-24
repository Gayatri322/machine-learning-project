# make_report_figure.py - draws Fig. 1 of the report (interpolation vs extrapolation curves).
# Run after generate_data.py:  python make_report_figure.py   ->  fig_curves.png
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
plt.rcParams.update({"font.family":"serif","font.size":8,"axes.linewidth":.6})
def st(m): return TransformedTargetRegressor(regressor=m, transformer=StandardScaler())
def build(cfg):
    n,p=cfg
    return {"Linear": lambda: make_pipeline(StandardScaler(),LinearRegression()),
     "Poly": lambda: make_pipeline(StandardScaler(),PolynomialFeatures(p),LinearRegression()),
     "Tree": lambda: DecisionTreeRegressor(max_depth=p,random_state=0),
     "Forest": lambda: RandomForestRegressor(n_estimators=200,max_depth=p,random_state=0,n_jobs=-1),
     "SVR": lambda: st(make_pipeline(StandardScaler(),SVR(kernel="rbf",C=p,epsilon=0.01))),
     "NN": lambda: st(make_pipeline(StandardScaler(),MLPRegressor(hidden_layer_sizes=p,max_iter=3000,random_state=0)))}[n]()
def z0(wh,er):
    ee=(er+1)/2+(er-1)/2/np.sqrt(1+12/wh); return 120*np.pi/(np.sqrt(ee)*(wh+1.393+2/3*np.log(wh+1.444)))
X=["W_over_h","eps_r"]; y="Z0_ohm"
A=pd.read_csv("data/microstrip_z0_train.csv")
D=pd.concat([pd.read_csv(f"data/microstrip_z0_{s}.csv") for s in ("train","val","test")])
cfgs_i=[("Linear",None),("Poly",6),("Tree",16),("Forest",None),("SVR",100),("NN",(64,64,32))]
cfgs_e=[("Linear",None),("Poly",6),("Tree",16),("Forest",None),("SVR",10),("NN",(32,32))]
wh=np.linspace(1,10,300); er=6.0; g=pd.DataFrame({"W_over_h":wh,"eps_r":er}); act=z0(wh,er)
sty={"Linear":(":","#888888"),"Poly":("--","#C2410C"),"Tree":("-.","#7C3AED"),"Forest":("-.","#0F766E"),"SVR":("--","#1D4E89"),"NN":("-","#B45309")}
fig,ax=plt.subplots(1,2,figsize=(6.875,2.45),dpi=300)
for a,cfgs,tr,title in [(ax[0],cfgs_i,A,"(a) Interpolation: train on random 70%"),(ax[1],cfgs_e,D[(D.W_over_h<7)],"(b) Extrapolation: train on W/h $<$ 7")]:
    a.plot(wh,act,color="k",lw=1.8,label="Formula")
    for cfg in cfgs:
        m=build(cfg).fit(tr[X],tr[y]); ls,c=sty[cfg[0]]
        a.plot(wh,m.predict(g),ls=ls,color=c,lw=1.0,label=cfg[0] if cfg[0]!="Poly" else "Poly (6)")
    a.set_xlabel("W/h"); a.set_ylabel("$Z_0$ ($\\Omega$) at $\\epsilon_r$ = 6"); a.set_title(title,fontsize=8)
    a.grid(alpha=.25,lw=.4)
ax[1].axvspan(8,10,color="#FDE68A",alpha=.5,lw=0); ax[1].axvline(7,color="gray",lw=.6,ls=":")
ax[1].set_ylim(-5,70); ax[0].set_ylim(0,70)
ax[0].legend(fontsize=6.5,frameon=False,ncol=2,loc="upper right")
fig.tight_layout(); fig.savefig("fig_curves.png"); print("ok")
