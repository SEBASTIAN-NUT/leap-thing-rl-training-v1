#!/usr/bin/env python3
"""python3 plot_phase.py {p2b|p3} {vx|yaw} [--edit]"""
import argparse, json
import numpy as np
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument("phase", choices=["p2b","p3"])
parser.add_argument("axis",  choices=["vx","yaw"])
parser.add_argument("--edit", action="store_true")
args = parser.parse_args()
if args.edit:
    import matplotlib; matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import japanize_matplotlib
plt.rcParams.update({"xtick.labelsize":11,"ytick.labelsize":11})
PROJECT_DIR  = Path(__file__).resolve().parent.parent
EVAL_DIR     = PROJECT_DIR/"eval_results"/"archived"
OFFSETS_JSON = PROJECT_DIR/f"{args.phase}_{args.axis}_offsets.json"
OUT_PNG      = PROJECT_DIR/f"{args.phase}_{args.axis}.png"
CONFIGS = {
    ("p2b","vx"):  {"title":"Phase 2b: sigma_ang sweep - vx tracking",
     "conds":[{"label":"sigma_ang=0.25","json":"p2b_sigang_0p25.json","sigma":0.025,"color":"#2980b9"},
              {"label":"sigma_ang=1.0", "json":"p2b_sigang_1p0.json", "sigma":0.025,"color":"#e67e22"},
              {"label":"sigma_ang=2.0", "json":"p2b_sigang_2p0.json", "sigma":0.025,"color":"#27ae60"}]},
    ("p2b","yaw"): {"title":"Phase 2b: sigma_ang sweep - yaw tracking",
     "conds":[{"label":"sigma_ang=0.25","json":"p2b_sigang_0p25.json","sigma":0.25, "color":"#2980b9"},
              {"label":"sigma_ang=1.0", "json":"p2b_sigang_1p0.json", "sigma":1.0,  "color":"#e67e22"},
              {"label":"sigma_ang=2.0", "json":"p2b_sigang_2p0.json", "sigma":2.0,  "color":"#27ae60"}]},
    ("p3","vx"):   {"title":"Phase 3: alive_penalty sweep - vx tracking",
     "conds":[{"label":"w_alive=0.1","json":"p3_alive_0p1.json","sigma":0.025,"color":"#2980b9"},
              {"label":"w_alive=0.3","json":"p3_alive_0p3.json","sigma":0.025,"color":"#e67e22"},
              {"label":"w_alive=1.0","json":"p3_alive_1p0.json","sigma":0.025,"color":"#27ae60"}]},
    ("p3","yaw"):  {"title":"Phase 3: alive_penalty sweep - yaw tracking",
     "conds":[{"label":"w_alive=0.1","json":"p3_alive_0p1.json","sigma":0.25,"color":"#2980b9"},
              {"label":"w_alive=0.3","json":"p3_alive_0p3.json","sigma":0.25,"color":"#e67e22"},
              {"label":"w_alive=1.0","json":"p3_alive_1p0.json","sigma":0.25,"color":"#27ae60"}]},
}
AXIS_CFG = {
    "vx":  {"cmd":"cmd_vx","meas":"meas_vx",
            "flt":lambda p:abs(p.get("cmd_vy",0))<0.005 and abs(p.get("cmd_yaw",0))<0.005,
            "xlim":(-0.18,0.18),"ylim":(-0.10,0.18),"xr":(-0.17,0.17),
            "xl":"Command vx [m/s]","yl":"Measured vx [m/s]",
            "emax":0.30,"eu":"m/s","bg":"#fff9f9"},
    "yaw": {"cmd":"cmd_yaw","meas":"meas_yaw",
            "flt":lambda p:abs(p.get("cmd_vx",0))<0.005 and abs(p.get("cmd_vy",0))<0.005,
            "xlim":(-1.2,1.2),"ylim":(-1.2,1.2),"xr":(-1.1,1.1),
            "xl":"Command yaw [rad/s]","yl":"Measured yaw [rad/s]",
            "emax":2.5,"eu":"rad/s","bg":"#f9fff9"},
}
def load_pts(jf, axis):
    ac=AXIS_CFG[axis]
    stem=Path(jf).stem; full=EVAL_DIR/f"{stem}_full.json"
    src=full if full.exists() else EVAL_DIR/jf
    d=json.loads(src.read_text())
    return [(p[ac["cmd"]],p[ac["meas"]]) for p in d.get("phases",[]) if ac["flt"](p)]
cfg=CONFIGS[(args.phase,args.axis)]; ac=AXIS_CFG[args.axis]
offs=json.loads(OFFSETS_JSON.read_text()) if OFFSETS_JSON.exists() else {}
fig,axes=plt.subplots(2,3,figsize=(15,10))
fig.suptitle(cfg["title"],fontsize=18,fontweight="bold")
ann_map={}; leg_map={}
for col,cond in enumerate(cfg["conds"]):
    clr,lbl,sigma=cond["color"],cond["label"],cond["sigma"]
    rows=load_pts(cond["json"],args.axis)
    cmds=np.array([r[0] for r in rows]); meas=np.array([r[1] for r in rows])
    if args.axis=="yaw":
        collapse=np.array([abs(c)>0.05 and abs(m)>0.05 and c*m<0 for c,m in zip(cmds,meas)])
    else:
        collapse=np.zeros(len(cmds),dtype=bool)
    valid=~collapse
    ax=axes[0][col]
    ax.scatter(cmds[valid],meas[valid],color=clr,s=80,zorder=5,
               label=("正常" if args.axis=="yaw" else "data"))
    if collapse.any():
        ax.scatter(cmds[collapse],meas[collapse],color=clr,s=80,zorder=5,
                   marker="x",linewidths=2,label="崩壊")
    m_ols,b_ols=(np.polyfit(cmds[valid],meas[valid],1) if valid.sum()>=2 else (0.,0.))
    xf=np.linspace(*ac["xr"],300)
    ax.plot(xf,m_ols*xf+b_ols,color=clr,lw=2,label=f"OLS slope={m_ols:.3f}")
    ax.plot(list(ac["xr"]),list(ac["xr"]),"k--",lw=1,alpha=0.25,label="完全追従")
    ax.axhline(0,color="gray",lw=0.6,alpha=0.4); ax.axvline(0,color="gray",lw=0.6,alpha=0.4)
    ax.set_xlim(*ac["xlim"]); ax.set_ylim(*ac["ylim"])
    ax.set_xlabel(ac["xl"],fontsize=13); ax.set_ylabel(ac["yl"],fontsize=13)
    ax.set_title(f"{lbl}\nOLS slope={m_ols:.3f}  (理想=1.00)",fontweight="bold",fontsize=13)
    lp=offs.get("legend_pos",{}).get(str(col))
    if lp: leg=ax.legend(fontsize=10,bbox_to_anchor=(lp["x"],lp["y"]),loc="lower left",borderaxespad=0)
    else:  leg=ax.legend(fontsize=10,loc="upper left")
    leg_map[col]=(ax,leg); ax.grid(alpha=0.3)
    for i,(c,mv) in enumerate(zip(cmds,meas)):
        key=f"top_{col}_{i}"; off=offs.get(key,{"dx":5.,"dy":5.})
        ann_map[key]=ax.annotate(f"{mv:+.3f}",xy=(c,mv),xytext=(off["dx"],off["dy"]),
            textcoords="offset points",fontsize=10,color=clr,
            arrowprops=dict(arrowstyle="-",color=clr,lw=0.5,alpha=0.5))
    ax2=axes[1][col]; e_r=np.linspace(0,ac["emax"],600); r_c=np.exp(-e_r**2/sigma)
    ax2.plot(e_r,r_c,color=clr,lw=2.5); ax2.set_facecolor(ac["bg"])
    obs_e=float(np.sqrt(np.mean((cmds-meas)**2)))
    r_obs=float(np.exp(-obs_e**2/sigma)); grad=-2.*obs_e/sigma*r_obs
    ax2.scatter([obs_e],[r_obs],color="red",s=130,zorder=6)
    e_t=np.array([max(0,obs_e-.05*ac["emax"]),min(ac["emax"],obs_e+.05*ac["emax"])])
    ax2.plot(e_t,r_obs+grad*(e_t-obs_e),"r-",lw=1.5,alpha=0.8)
    key2=f"bot_{col}_obs"; off2=offs.get(key2,{"dx":12.,"dy":10.})
    ann_map[key2]=ax2.annotate(
        f"RMS誤差={obs_e:.3f} {ac['eu']}\n|dr/de|={abs(grad):.2f}",
        xy=(obs_e,r_obs),xytext=(off2["dx"],off2["dy"]),textcoords="offset points",
        fontsize=11,color="red",arrowprops=dict(arrowstyle="->",color="red",lw=1.2))
    ax2.set_xlim(0,ac["emax"]); ax2.set_ylim(-0.05,1.15)
    ax2.set_xlabel(f"速度誤差 e=|cmd-meas|[{ac['eu']}]",fontsize=13)
    ax2.set_ylabel("追従報酬  r(e)",fontsize=13)
    ax2.set_title(f"{lbl}\n|dr/de|={abs(grad):.2f}  at e={obs_e:.3f}",fontsize=13)
    ax2.grid(alpha=0.3)
    ax2.text(0.97,0.95,f"r(e)=exp(-e^2/{sigma})",
             transform=ax2.transAxes,ha="right",va="top",fontsize=11,color=clr)
plt.tight_layout(rect=[0,.01,1,.97])
if args.edit:
    _drags=[ann.draggable(True) for ann in ann_map.values()]
    for _,lr in leg_map.values(): lr.set_draggable(True)
    def on_close(event):
        data={k:{"dx":a.xyann[0],"dy":a.xyann[1]} for k,a in ann_map.items()}
        try:
            renderer=fig.canvas.get_renderer(); lpos={}
            for c,(ax_r,lr) in leg_map.items():
                lw=lr.get_window_extent(renderer); aw=ax_r.get_window_extent(renderer)
                lpos[str(c)]={"x":float((lw.x0-aw.x0)/aw.width),"y":float((lw.y0-aw.y0)/aw.height)}
            data["legend_pos"]=lpos
        except Exception: pass
        OFFSETS_JSON.write_text(json.dumps(data,indent=2)); print(f"保存:{OFFSETS_JSON}")
    fig.canvas.mpl_connect("close_event",on_close)
    print("GUI編集モード"); plt.show()
    fig.savefig(OUT_PNG,dpi=150,bbox_inches="tight"); print(f"PNG保存:{OUT_PNG}")
else:
    plt.savefig(OUT_PNG,dpi=150,bbox_inches="tight"); print(f"PNG保存:{OUT_PNG}")
