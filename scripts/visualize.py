"""Render saved physical trajectories without rerunning the planner."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trajectory', type=Path, required=True)
    p.add_argument('--scene', type=Path, required=True)
    p.add_argument('--output', type=Path, default=Path('trajectory.mp4'))
    p.add_argument('--title', default='Akule · recorded trajectory')
    p.add_argument('--duration', type=float, default=8.)
    args=p.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter
    from matplotlib.patches import Circle, Rectangle
    data=np.load(args.trajectory,allow_pickle=False)
    path=np.asarray(data['physical'])
    if path.ndim==4: path=path[0]
    if path.ndim!=3 or path.shape[-1] not in (2,4) or not np.isfinite(path).all(): raise ValueError('Expected finite [T,N,2 or 4] physical trajectories')
    scene=json.loads(args.scene.read_text());n=path.shape[1]
    if n!=len(scene['starts']):raise ValueError('Scene/trajectory population mismatch')
    colors=plt.cm.turbo(np.linspace(.06,.92,n))
    fig,ax=plt.subplots(figsize=(8,8),dpi=100);fig.patch.set_facecolor('#f6f9ff');ax.set_facecolor('#ffffff')
    bounds=np.asarray(scene.get('workspace',[[-1.,-1.],[1.,1.]]))
    lo=np.minimum(bounds[0],path[...,:2].min((0,1)));hi=np.maximum(bounds[1],path[...,:2].max((0,1)))
    ax.set(xlim=(lo[0]-.08,hi[0]+.08),ylim=(lo[1]-.08,hi[1]+.08),aspect='equal');ax.set_xticks([]);ax.set_yticks([])
    for spine in ax.spines.values():spine.set_color('#dfe5ef')
    for item in scene.get('obstacles',{}).get('items',[]):
        center=np.asarray(item['center'])
        shape=Circle(center,item['radius']) if 'radius' in item else Rectangle(center-np.array(item['size'])/2,*item['size'])
        shape.set(facecolor='#dbe4f1',edgecolor='#aebed5',linewidth=.6);ax.add_patch(shape)
    ax.scatter(np.array(scene['goals'])[:,0],np.array(scene['goals'])[:,1],marker='*',s=30,c=colors,alpha=.6,linewidths=0)
    points=ax.scatter(path[0,:,0],path[0,:,1],s=max(12,100-n*.25),c=colors,edgecolors='#153653',linewidths=.4,zorder=4)
    lines=[ax.plot([],[],color=colors[i],lw=.8,alpha=.4)[0] for i in range(n)]
    ax.set_title(args.title,fontsize=14,color='#1850a0',pad=20)
    stamp=ax.text(.02,.02,'',transform=ax.transAxes,color='#55708e',fontsize=9)
    count=int(args.duration*24)
    def update(frame):
        t=frame/(count-1)*(len(path)-1);a=int(t);b=min(a+1,len(path)-1);xy=(1-(t-a))*path[a,:,:2]+(t-a)*path[b,:,:2]
        points.set_offsets(xy)
        for i,line in enumerate(lines):line.set_data(path[:a+1,i,0],path[:a+1,i,1])
        stamp.set_text(f'{n} robots · recorded trajectory')
        return points,*lines,stamp
    args.output.parent.mkdir(parents=True,exist_ok=True)
    animation=FuncAnimation(fig,update,frames=count,interval=1000/24,blit=False)
    animation.save(args.output,writer=FFMpegWriter(fps=24,codec='libx264',extra_args=['-crf','25','-pix_fmt','yuv420p','-movflags','+faststart','-map_metadata','-1']))
    plt.close(fig)


if __name__=='__main__':main()
