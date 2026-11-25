import argparse, zipfile
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, confusion_matrix, average_precision_score
from sklearn.utils.class_weight import compute_sample_weight
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def rules(y,cl,lcl,ucl):
    y=pd.Series(y,dtype=float).reset_index(drop=True); d=y.diff()
    return pd.DataFrame({'r1_low':y<lcl,'r1_high':y>ucl,'r2_low':(y<cl).rolling(9).sum().eq(9),'r2_high':(y>cl).rolling(9).sum().eq(9),'r3_low':(d<0).rolling(5).sum().eq(5),'r3_high':(d>0).rolling(5).sum().eq(5)})

def main(a):
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    lab=pd.read_csv(a.lab,sep=';'); p=pd.read_csv(a.process,sep=';'); assert len(lab.merge(p,on=['batch','code'],validate='one_to_one'))==len(lab)==len(p)
    codes=[17,23,13]; sensor=['tbl_speed','main_comp','pre_comp','tbl_fill','SREL','fom','stiffness','ejection','cyl_main','cyl_pre']; rows=[]
    with zipfile.ZipFile(a.zip) as z:
        for c in codes:
            d=pd.read_csv(z.open(f'Process/{c}.csv'),sep=';',usecols=['timestamp','batch','code','campaign']+sensor)
            d['ts']=pd.to_datetime(d.timestamp,format='%d%m%Y %H:%M',errors='coerce')
            for (b,code),g in d.groupby(['batch','code']):
                start,end=g.ts.min(),g.ts.max(); w=g[(g.ts>=start)&(g.ts<start+pd.Timedelta(minutes=30))]
                r=dict(batch=b,code=code,start_ts=start,end_ts=end,campaign=g.campaign.iloc[0],eligible_30m=end>=start+pd.Timedelta(minutes=30),early_n=len(w))
                for s in sensor: r['early_'+s+'_mean']=w[s].mean(); r['early_'+s+'_std']=w[s].std()
                rows.append(r)
    df=lab[lab.code.isin(codes)].merge(pd.DataFrame(rows),on=['batch','code'],how='left',validate='one_to_one'); df[['batch','code','start_ts','end_ts','campaign','eligible_30m','early_n']].to_csv(out/'log_coverage.csv',index=False)
    print('coverage',len(df),df.start_ts.notna().sum()); df=df.dropna(subset=['start_ts','end_ts','tbl_yield']).sort_values(['end_ts','batch']); cut=df.end_ts.iloc[int(len(df)*.6)]; print('cutoff',cut)
    df['phase']=np.select([df.end_ts<cut,df.start_ts>=cut],['train','test'],default='boundary'); print(df.groupby(['code','phase']).size()); df=df[df.phase!='boundary']; frames=[]; limits=[]
    for c,g in df.groupby('code'):
        g=g.sort_values(['end_ts','batch']).reset_index(drop=True); ref=g[g.phase=='train'].tbl_yield; cl=ref.mean(); mrbar=ref.diff().abs().mean(); sigma=mrbar/1.128; lo=cl-3*sigma; hi=cl+3*sigma; mrhi=3.267*mrbar
        s=rules(g.tbl_yield,cl,lo,hi); g=pd.concat([g,s],axis=1); g['mr']=g.tbl_yield.diff().abs(); g['mr_high']=g.mr>mrhi; g['lower_signal']=g[['r1_low','r2_low','r3_low']].any(axis=1); g['lower_trend']=g[['r2_low','r3_low']].any(axis=1); g['any_signal']=g[list(s.columns)+['mr_high']].any(axis=1)
        for i,r in g.iterrows():
            prior=g[(g.end_ts<=r.start_ts)&(g.index<i)].tbl_yield
            for k in range(1,9): g.loc[i,f'lag{k}']=prior.iloc[-k] if len(prior)>=k else np.nan
        t=g[g.phase=='test']; limits.append(dict(code=c,train_n=len(ref),test_n=len(t),cl=cl,lcl=lo,ucl=hi,mrbar=mrbar,mr_ucl=mrhi,**{x:int(t[x].sum()) for x in list(s.columns)+['mr_high','lower_signal','lower_trend','any_signal']})); frames.append(g)
        fig,axs=plt.subplots(2,1,figsize=(12,6),sharex=True,layout='constrained'); x=np.arange(1,len(g)+1)
        for ax,v,center,l,h,mask,title in [(axs[0],g.tbl_yield,cl,lo,hi,s.any(axis=1),'I chart'),(axs[1],g.mr,mrbar,0,mrhi,g.mr_high,'MR chart')]:
            ax.plot(x,v,'.-',lw=.8,ms=3); ax.scatter(x[mask],v[mask],s=22,c='#c43c35',label='SPC signal'); ax.axhline(center,color='gray'); ax.axhline(l,color='#c43c35',ls='--'); ax.axhline(h,color='#c43c35',ls='--'); ax.axvline(len(ref)+.5,color='black',ls=':'); ax.set_title(title); ax.set_ylabel('Yield (%)' if ax==axs[0] else 'Moving range (pp)'); ax.legend(loc='lower left',fontsize=8)
        axs[1].set_xlabel('Same-product batch sequence (completion order)'); fig.suptitle(f'Product {c} | historical limits | train / test'); fig.savefig(out/f'imr_code_{c}.png',dpi=150); plt.close(fig)
    g=pd.concat(frames,ignore_index=True); lim=pd.DataFrame(limits); lim.to_csv(out/'spc_summary.csv',index=False); print(lim.to_string(index=False)); g[['batch','code','start_ts','end_ts','phase','tbl_yield','mr']+list(s.columns)+['mr_high','lower_signal','lower_trend','any_signal']].to_csv(out/'spc_signals.csv',index=False)
    raw=['size','api_water','api_total_impurities','api_l_impurity','api_content','api_ps01','api_ps05','api_ps09','lactose_water','lactose_sieve0045','lactose_sieve015','lactose_sieve025','smcc_water','smcc_td','smcc_bd','smcc_ps01','smcc_ps05','smcc_ps09','starch_ph','starch_water']; lag=[f'lag{k}' for k in range(1,9)]; early=[x for x in g if x.startswith('early_') and x!='early_n']; g=g[g.eligible_30m.eq(True)].copy(); train=g[g.phase=='train']; test=g[g.phase=='test']; results=[]; preds=[]; direct=[]; lm=lim.set_index('code')
    for mode,cols in [('pre_batch',raw+lag),('early_30m',raw+lag+early)]:
        X=pd.get_dummies(g[['code']].astype(str),dtype=float).join(g[cols].apply(pd.to_numeric,errors='coerce'))
        for name,pred in [('ProductMedian',test.code.map(train.groupby('code').tbl_yield.median()).to_numpy()),('LastYield',test.lag1.fillna(test.code.map(train.groupby('code').tbl_yield.median())).to_numpy())]:
            results.append(dict(mode=mode,model=name,train_n=len(train),test_n=len(test),mae=mean_absolute_error(test.tbl_yield,pred),rmse=mean_squared_error(test.tbl_yield,pred)**.5))
        # Direct classification: positive classes are retrospective SPC labels, not GMP deviations.
        for target in ['lower_signal','lower_trend']:
            yt=train[target].astype(int); ya=test[target].astype(int)
            for name,model in [('RandomForestClassifier',RandomForestClassifier(n_estimators=200,min_samples_leaf=5,class_weight='balanced',random_state=42,n_jobs=-1)),('GradientBoostingClassifier',GradientBoostingClassifier(n_estimators=100,max_depth=2,min_samples_leaf=5,random_state=42))]:
                pipe=make_pipeline(SimpleImputer(strategy='median'),model)
                kwargs={'gradientboostingclassifier__sample_weight':compute_sample_weight('balanced',yt)} if name=='GradientBoostingClassifier' else {}
                pipe.fit(X.loc[train.index],yt,**kwargs); prob=pipe.predict_proba(X.loc[test.index])[:,1]; pr=prob>=.5
                tn,fp,fn,tp=confusion_matrix(ya,pr,labels=[0,1]).ravel(); direct.append(dict(mode=mode,model=name,target=target,train_positive=int(yt.sum()),test_positive=int(ya.sum()),pred_positive=tp+fp,precision=tp/max(1,tp+fp),recall=tp/max(1,tp+fn),f1=2*tp/max(1,2*tp+fp+fn),average_precision=average_precision_score(ya,prob),tn=tn,fp=fp,fn=fn,tp=tp))
        for name,model in [('DummyMedian',DummyRegressor(strategy='median')),('RandomForest',RandomForestRegressor(n_estimators=200,min_samples_leaf=5,random_state=42,n_jobs=-1)),('GradientBoosting',GradientBoostingRegressor(n_estimators=100,max_depth=2,min_samples_leaf=5,random_state=42))]:
            pipe=make_pipeline(SimpleImputer(strategy='median'),model); pipe.fit(X.loc[train.index],train.tbl_yield); pred=pipe.predict(X.loc[test.index]); results.append(dict(mode=mode,model=name,train_n=len(train),test_n=len(test),mae=mean_absolute_error(test.tbl_yield,pred),rmse=mean_squared_error(test.tbl_yield,pred)**.5))
            for (_,r),v in zip(test.iterrows(),pred):
                l=lm.loc[r.code]; sig=rules([r[f'lag{k}'] for k in range(8,0,-1)]+[v],l.cl,l.lcl,l.ucl).iloc[-1]; preds.append(dict(mode=mode,model=name,batch=r.batch,code=r.code,actual=r.tbl_yield,predicted=v,actual_lower_signal=bool(r.lower_signal),pred_lower_signal=bool(sig[['r1_low','r2_low','r3_low']].any()),actual_lower_trend=bool(r.lower_trend),pred_lower_trend=bool(sig[['r2_low','r3_low']].any()),actual_proxy_below95=r.tbl_yield<95,pred_proxy_below95=v<95))
    pred=pd.DataFrame(preds); pred.to_csv(out/'predictions.csv',index=False); metrics=[]
    for (mode,model),h in pred.groupby(['mode','model']):
        for target in ['lower_signal','lower_trend','proxy_below95']:
            tn,fp,fn,tp=confusion_matrix(h['actual_'+target],h['pred_'+target],labels=[False,True]).ravel(); metrics.append(dict(mode=mode,model=model,target=target,positive=tp+fn,pred_positive=tp+fp,precision=tp/max(1,tp+fp),recall=tp/max(1,tp+fn),f1=2*tp/max(1,2*tp+fp+fn),tn=tn,fp=fp,fn=fn,tp=tp))
    for name,data in [('ml_regression',results),('ml_alerts',metrics)]:
        d=pd.DataFrame(data); d.to_csv(out/f'{name}.csv',index=False); print(d.to_string(index=False))
    pd.DataFrame(direct).to_csv(out/'ml_direct_alerts.csv',index=False); print(pd.DataFrame(direct).to_string(index=False))
    (out/'run_info.txt').write_text(f'Cutoff: {cut}\nTrain/test ML: {len(train)}/{len(test)}\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--lab',required=True); p.add_argument('--process',required=True); p.add_argument('--zip',required=True); p.add_argument('--out',default='results'); main(p.parse_args())
