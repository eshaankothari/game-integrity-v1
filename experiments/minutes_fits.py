import pandas as pd, numpy as np, warnings, sys, pathlib
warnings.filterwarnings("ignore")
sys.path.insert(0,"/Users/eshaankothari/Desktop/game-integrity-v1")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import db
OUT=pathlib.Path("/Users/eshaankothari/Desktop/game-integrity-v1/experiments/out")
q="""SELECT minutes, touches, passes, usage_pct, distance, contested_shots,
            deflections, loose_balls, box_outs, screen_assists
     FROM player_games WHERE minutes>0 AND points IS NOT NULL"""
with db.connect() as c: d=pd.read_sql(q,c)
for c_ in d.columns: d[c_]=pd.to_numeric(d[c_],errors="coerce")
E=["touches","passes","usage_pct","distance","contested_shots","deflections",
   "loose_balls","box_outs","screen_assists"]
fig,ax=plt.subplots(3,3,figsize=(15,12))
for a,c_ in zip(ax.ravel(),E):
    ok=d[[c_,"minutes"]].dropna()
    m=LinearRegression().fit(ok[["minutes"]].values, ok[c_].values)
    r2=m.score(ok[["minutes"]].values, ok[c_].values)
    a.scatter(ok.minutes, ok[c_], s=3, alpha=.05, color="#4C72B0", rasterized=True)
    xs=np.linspace(0, ok.minutes.max(), 100)
    a.plot(xs, m.coef_[0]*xs+m.intercept_, color="crimson", lw=2)
    a.axhline(0,color="grey",lw=.7); a.axvline(0,color="grey",lw=.7)
    a.plot([0],[m.intercept_],"o",color="black",ms=6,zorder=5)
    a.set_title(f"{c_}\nslope {m.coef_[0]:+.4f}   intercept {m.intercept_:+.3f}   "
                f"R2 {r2:.2f}", fontsize=10)
    a.set_xlabel("minutes"); a.set_ylabel(c_)
    a.set_xlim(0, ok.minutes.max()*1.02)
fig.suptitle("Each effort stat regressed on minutes -- fitted separately\n"
             "red = fitted line, black dot = intercept (predicted value at 0 minutes)",
             fontsize=12)
fig.tight_layout(rect=[0,0,1,0.955])
fig.savefig(OUT/"minutes_fits.png",dpi=130)
print(f"-> {OUT/'minutes_fits.png'}")
