"""Native Matplotlib sliders. Start: python3 doyle_slider.py
Headless verification: python3 doyle_slider.py --preview
"""
import sys
from pathlib import Path
import matplotlib
if '--preview' in sys.argv:
    matplotlib.use('Agg')
elif sys.platform == 'darwin':
    matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import numpy as np
import doyle_model as model

BASE = model.PAR.copy()
LABELS = {
 'Vtot':'Blood volume (L)', 'Rs0':'Baseline resistance', 'A':'O₂–resistance coefficient',
 'cl':'Left-heart pump coefficient', 'cr':'Right-heart pump coefficient',
 'cas':'Systemic arterial compliance', 'cvs':'Systemic venous compliance',
 'cap':'Pulmonary arterial compliance', 'cvp':'Pulmonary venous compliance',
 'Rp':'Pulmonary resistance', 'Oa':'Arterial O₂ content',
 'rho':'O₂ demand per watt', 'M0':'Baseline O₂ demand', 'VT':'Effective tissue O₂ volume',
 'qP':'Pressure penalty ×', 'qO':'O₂ penalty ×', 'qH':'Heart-rate penalty ×'}
KEYS = list(LABELS)

def run(values, high=False):
    """Use modified parameters consistently, then restore original dictionary."""
    saved = model.PAR.copy()
    try:
        model.PAR.update({k:values[k] for k in BASE})
        H0,W0,q = (120,100,(80,100000,35)) if high else (60,0,(30,100000,1))
        weights=tuple(v*values[k] for v,k in zip(q,['qP','qO','qH']))
        x0=model.equilibrium(H0,W0)
        if not (np.all(x0[:3]>0) and 0<x0[3]<model.PAR['Oa']):
            raise ValueError('Initial state is outside the model domain.')
        return model.simulate(H0,W0,weights)
    finally:
        model.PAR.clear();model.PAR.update(saved)

def create_window():
    fig = plt.figure(figsize=(14,10))
    fig.suptitle('Doyle model: explore the parameters',fontsize=17,x=.67,y=.97)
    fig.text(.47,.925,'Published equations and parameters · illustrative simulation',fontsize=10)
    fig.text(.025,.96,'1. Move a slider   2. Recalculate',fontsize=12,weight='bold')
    sliders={}
    for i,k in enumerate(KEYS):
        init=BASE[k] if k in BASE else 1.0
        low,high=(.5*init,1.5*init) if k in BASE else (.2,3.0)
        # This is exploratory bounds selection, not clinical reference ranges.
        if k=='Oa':low,high=.12,.24
        ax=fig.add_axes([.245,.885-i*.043,.165,.021])
        sliders[k]=Slider(ax,LABELS[k],low,high,valinit=init,valfmt='%.4g',color='#17848b')
        sliders[k].label.set_fontsize(9)
        sliders[k].valtext.set_fontsize(9)
    axs=[fig.add_axes([.55,.715-i*.19,.41,.14]) for i in range(4)]
    ylabel=['Workload (W)','Heart rate (beats/min)','Mean arterial pressure (mmHg)','Venous O₂ content\n(ml O₂/L blood)']
    original=[];current=[]
    for ax,label in zip(axs,ylabel):
        original.append(ax.plot([],[],color='#999999',ls='--',label='Published parameters')[0])
        current.append(ax.plot([],[],color='#147f89',label='Your parameters')[0])
        # Horizontal panel headings stay clear of slider labels and values.
        ax.set_title(label.replace('\n',' '),loc='left',fontsize=10,pad=8)
        ax.tick_params(labelsize=9)
        ax.grid(alpha=.2)
    axs[-1].set_xlabel('Time (seconds)');axs[1].legend(fontsize=8,loc='upper left')
    status=fig.text(.47,.045,'',fontsize=10,color='#24545a',wrap=True)
    fig.text(.025,.135,'Slider ranges are exploratory. Each simulation starts\nat its own equilibrium; parameters remain constant\nthroughout that simulation.',fontsize=9)
    radio=RadioButtons(fig.add_axes([.025,.03,.15,.085]),['0–50 W','100–150 W'])
    calculate=Button(fig.add_axes([.20,.055,.13,.045]),'Recalculate')
    reset=Button(fig.add_axes([.345,.055,.09,.045]),'Reset')
    def redraw(event=None):
        status.set_text('Calculating …');fig.canvas.draw_idle()
        values={k:s.val for k,s in sliders.items()}
        high=radio.value_selected=='100–150 W'
        try:
            result=run(values,high)
            reference=run({**BASE,'qP':1,'qO':1,'qH':1},high)
            for lines,data in [(original,reference),(current,result)]:
                t,x,w,H,K=data
                for line,y in zip(lines,[w,H,x[:,0],x[:,3]*1000]):line.set_data(t,y)
            for ax in axs:ax.relim();ax.autoscale_view()
            t,x,w,H,K=result
            status.set_text(f'Calculated. Initial pressure: {x[0,0]:.1f} mmHg · maximum HR: {H.max():.1f}/min')
        except (ValueError,RuntimeError,AssertionError,np.linalg.LinAlgError) as exc:
            status.set_text('Outside the model domain / controller cannot be solved.\nLast valid curves remain visible; reset parameters.')
            print('Calculation rejected:',type(exc).__name__,str(exc))
        fig.canvas.draw_idle()
    def pending(value):
        status.set_text('Parameters changed. Click Recalculate.');fig.canvas.draw_idle()
    def defaults(event=None):
        for s in sliders.values():s.reset()
        redraw()
    for s in sliders.values():s.on_changed(pending)
    radio.on_clicked(pending);calculate.on_clicked(redraw);reset.on_clicked(defaults)
    redraw()
    # Keep widgets alive for the entire GUI lifetime.
    fig._doyle_widgets=(sliders,radio,calculate,reset,redraw)
    return fig

if __name__=='__main__':
    fig=create_window()
    if '--preview' in sys.argv:
        fig.savefig(Path(__file__).resolve().parent/'slider_preview.png',dpi=120)
    else:
        plt.show()
