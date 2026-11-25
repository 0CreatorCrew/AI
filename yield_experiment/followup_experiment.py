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
    g=pd.concat(frames,ignore_index=True); lim=pd.DataFrame(limits); lim.to_csv(out/'spc_summary.csv',index=False); print(lim.to_string(index=False)); g[['batch','code','start_ts','end_ts','phase','tbl_yield','mr']+list(s.columns)+['mr_high','lower_signal','lower_trend','any_signal']].to_csv(out/'spc_signals.csv',index=False)
    raw=['size','api_water','api_total_impurities','api_l_impurity','api_content','api_ps01','api_ps05','api_ps09','lactose_water','lactose_sieve0045','lactose_sieve015','lactose_sieve025','smcc_water','smcc_td','smcc_bd','smcc_ps01','smcc_ps05','smcc_ps09','starch_ph','starch_water']; lag=[f'lag{k}' for k in range(1,9)]; early=[x for x in g if x.startswith('early_') and x!='early_n']; g=g[g.eligible_30m.eq(True)].copy(); train=g[g.phase=='train']; test=g[g.phase=='test']; results=[]; preds=[]; direct=[]; lm=lim.set_index('code')
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import precision_score, recall_score, fbeta_score, average_precision_score
    from itertools import combinations
    test=test.copy(); combos=[]
    for subset_n in range(1,4):
        for ss in combinations([1,2,3],subset_n):
            low=test[[f'r{i}_low' for i in ss]].any(axis=1); high=test[[f'r{i}_high' for i in ss]].any(axis=1)
            combos.append(dict(rules='+'.join(map(str,ss)),lower=int(low.sum()),upper=int(high.sum()),any=int((low|high).sum()),with_mr=int((low|high|test.mr_high).sum())))
    pd.DataFrame(combos).to_csv(out/'rule_comparison.csv',index=False)
    print('RULES',pd.DataFrame(combos).to_string(index=False))
    # chronological internal validation; limits estimated from earlier subset only
    cut2=train.sort_values('end_ts').end_ts.iloc[int(len(train)*.7)]
    inner=train[train.end_ts<cut2].copy(); val=train[train.start_ts>=cut2].copy()
    tmp=[]
    for c,h in train.groupby('code'):
        h=h.sort_values(['end_ts','batch']).copy(); ref=inner[inner.code==c].sort_values('end_ts').tbl_yield; cl=ref.mean(); sd=ref.diff().abs().mean()/1.128
        rr=rules(h.tbl_yield,cl,cl-3*sd,cl+3*sd); h['inner_label']=rr[['r1_low','r2_low','r3_low']].any(axis=1).to_numpy(); tmp.append(h[['inner_label']])
    labels=pd.concat(tmp).inner_label.astype(int); yi=labels.loc[inner.index]; yv=labels.loc[val.index]; yt=train.lower_signal.astype(int); ya=test.lower_signal.astype(int)
    print('INNER',cut2,len(inner),len(val),yi.sum(),yv.sum())
    imp=[]; tuning=[]
    for mode,cols in [('pre_batch',raw+lag),('early_30m',raw+lag+early)]:
        X=pd.get_dummies(g[['code']].astype(str),dtype=float).join(g[cols].apply(pd.to_numeric,errors='coerce'))
        for name in ['RF','GB']:
            def newclf():
                return make_pipeline(SimpleImputer(strategy='median'),RandomForestClassifier(n_estimators=200,min_samples_leaf=5,class_weight='balanced',random_state=42,n_jobs=1) if name=='RF' else GradientBoostingClassifier(n_estimators=100,max_depth=2,min_samples_leaf=5,random_state=42))
            def fitclf(pipe,idx,y):
                kw={'gradientboostingclassifier__sample_weight':compute_sample_weight('balanced',y)} if name=='GB' else {}; return pipe.fit(X.loc[idx],y,**kw)
            ci=fitclf(newclf(),inner.index,yi); pv=ci.predict_proba(X.loc[val.index])[:,1]; candidates=np.unique(np.r_[.5,pv]); scores=[fbeta_score(yv,pv>=th,beta=2,zero_division=0) for th in candidates]; th=float(candidates[np.argmax(scores)])
            clf=fitclf(newclf(),train.index,yt); pt=clf.predict_proba(X.loc[test.index])[:,1]
            for kind,t in [('default',.5),('val_F2',th)]:
                pred=pt>=t; tn,fp,fn,tp=confusion_matrix(ya,pred,labels=[0,1]).ravel(); tuning.append(dict(mode=mode,model=name,policy=kind,threshold=t,val_n=len(val),val_positive=int(yv.sum()),val_precision=precision_score(yv,pv>=t,zero_division=0),val_recall=recall_score(yv,pv>=t,zero_division=0),tp=tp,fp=fp,fn=fn,tn=tn,precision=precision_score(ya,pred,zero_division=0),recall=recall_score(ya,pred,zero_division=0),f2=fbeta_score(ya,pred,beta=2,zero_division=0),ap=average_precision_score(ya,pt)))
            if mode=='early_30m':
                reg=make_pipeline(SimpleImputer(strategy='median'),RandomForestRegressor(n_estimators=200,min_samples_leaf=5,random_state=42,n_jobs=1)) if name=='RF' else make_pipeline(SimpleImputer(strategy='median'),GradientBoostingRegressor(n_estimators=100,max_depth=2,min_samples_leaf=5,random_state=42))
                reg.fit(X.loc[train.index],train.tbl_yield)
                for typ,model,y,scoring in [('regression',reg,test.tbl_yield,'neg_mean_absolute_error'),('classification',clf,ya,'average_precision')]:
                    pi=permutation_importance(model,X.loc[test.index],y,scoring=scoring,n_repeats=5,random_state=42,n_jobs=1)
                    kept=model[0].get_feature_names_out(); mdi=dict(zip(kept,model[-1].feature_importances_))
                    for f,mean,std in zip(X.columns,pi.importances_mean,pi.importances_std): imp.append(dict(model=name,task=typ,feature=f,mdi=mdi.get(f,0),permutation_mean=mean,permutation_std=std))
    pd.DataFrame(tuning).to_csv(out/'threshold_comparison.csv',index=False); pd.DataFrame(imp).to_csv(out/'feature_importance.csv',index=False)
    print(pd.DataFrame(tuning).to_string(index=False)); print(pd.DataFrame(imp).query("model=='RF'").sort_values('permutation_mean',ascending=False).groupby('task').head(8).to_string(index=False))

    # Rule-aware prediction: turn each eligible Nelson rule into a yield boundary.
    # Internal validation uses only earlier historical control limits; outer limits
    # are rebuilt from the full training period, so the label definition can shift.
    import json, joblib, hashlib, sys, sklearn
    inner_limits={}
    for c,h in inner.groupby('code'):
        ref=h.sort_values(['end_ts','batch']).tbl_yield; cl=ref.mean(); sd=ref.diff().abs().mean()/1.128
        inner_limits[c]=(cl,cl-3*sd)
    outer_limits={c:(r.cl,r.lcl) for c,r in lm.iterrows()}
    def boundary(h,ls):
        out=[]
        for _,r in h.iterrows():
            cl,lo=ls[r.code]; prev=np.array([r[f'lag{k}'] for k in range(8,0,-1)],float); b=[lo]
            if np.isfinite(prev).all() and (prev<cl).all(): b.append(cl)
            if np.isfinite(prev[-5:]).all() and (np.diff(prev[-5:])<0).all(): b.append(prev[-1])
            out.append(max(b))
        return np.array(out)
    X=pd.get_dummies(g[['code']].astype(str),dtype=float).join(g[raw+lag+early].apply(pd.to_numeric,errors='coerce'))
    def newreg(): return make_pipeline(SimpleImputer(strategy='median'),RandomForestRegressor(n_estimators=200,min_samples_leaf=5,random_state=42,n_jobs=1))
    mi=newreg().fit(X.loc[inner.index],inner.tbl_yield); mv=mi.predict(X.loc[val.index]); bv=boundary(val,inner_limits); rv=bv-mv
    # Predeclared exploratory alert budget: no more than 15% of validation batches.
    candidates=np.unique(np.r_[0.,rv]); feasible=[float(t) for t in candidates if (rv>=t).mean()<=.15 and (rv>=t).any()]
    chosen=max(feasible,key=lambda t:(fbeta_score(yv,rv>=t,beta=2,zero_division=0),precision_score(yv,rv>=t,zero_division=0),t)) if feasible else float('inf')
    final=newreg().fit(X.loc[train.index],train.tbl_yield); pred=final.predict(X.loc[test.index]); bt=boundary(test,outer_limits); rt=bt-pred
    rr=[]
    for label,t in [('unadjusted',0.),('validation_F2_alert_budget_15pct',chosen)]:
        pr=rt>=t; tn,fp,fn,tp=confusion_matrix(ya,pr,labels=[0,1]).ravel(); rr.append(dict(policy=label,risk_margin_threshold=t,val_alerts=int((rv>=t).sum()),val_recall=recall_score(yv,rv>=t,zero_division=0),val_precision=precision_score(yv,rv>=t,zero_division=0),test_alerts=int(pr.sum()),tp=tp,fp=fp,fn=fn,tn=tn,precision=precision_score(ya,pr,zero_division=0),recall=recall_score(ya,pr,zero_division=0),f2=fbeta_score(ya,pr,beta=2,zero_division=0)))
    pd.DataFrame(rr).to_csv(out/'rule_aware_comparison.csv',index=False); print('RULE AWARE',pd.DataFrame(rr).to_string(index=False))
    # Exploratory split-conformal interval: keep the early model fixed; do NOT refit it after calibration.
    # Chronological/correlated data may violate exchangeability; report actual observed coverage.
    residual=np.abs(val.tbl_yield.to_numpy()-mv); n=len(residual); k=min(n,int(np.ceil((n+1)*.90))); q=float(np.sort(residual)[k-1]); fixed_pred=mi.predict(X.loc[test.index]); actual=test.tbl_yield.to_numpy(); covered=np.abs(actual-fixed_pred)<=q
    pd.DataFrame([dict(train_n=len(inner),calibration_n=n,test_n=len(test),nominal_coverage=.9,half_width=q,observed_coverage=float(covered.mean()),covered_n=int(covered.sum()),mae=mean_absolute_error(actual,fixed_pred))]).to_csv(out/'uncertainty_summary.csv',index=False)
    # Feature envelope per product is a simple screening proxy, NOT validated drift or a confidence probability.
    dr=[]
    for idx,r in test.iterrows():
        ref=X.loc[inner[inner.code==r.code].index,raw+lag+early]; z=X.loc[idx,raw+lag+early]; lo=ref.min(); hi=ref.max(); outside=z.notna()&lo.notna()&((z<lo)|(z>hi)); missing=z.isna(); dr.append((idx,int(outside.sum()),int(missing.sum())))
    drift=pd.DataFrame(dr,columns=['index','outside_count','missing_count']).set_index('index').loc[test.index]; intervals=pd.DataFrame({'batch':test.batch,'code':test.code,'actual':actual,'prediction':fixed_pred,'lower':fixed_pred-q,'upper':fixed_pred+q,'covered':covered,'outside_count':drift.outside_count,'missing_count':drift.missing_count}); intervals['review_required']=(intervals.outside_count>0)|(intervals.missing_count>0); intervals.to_csv(out/'uncertainty_review.csv',index=False)
    # Save static model and configuration locally; outputs are ignored by Git.
    model_dir=out/'models'; model_dir.mkdir(exist_ok=True); joblib.dump(mi,model_dir/'rf_early_fixed.joblib'); joblib.dump(final,model_dir/'rf_full_train.joblib')
    reloaded=joblib.load(model_dir/'rf_full_train.joblib'); repeat_error=float(np.max(np.abs(final.predict(X.loc[test.index])-pred))); reload_error=float(np.max(np.abs(reloaded.predict(X.loc[test.index])-pred))); assert repeat_error==0 and reload_error==0
    (out/'determinism_check.txt').write_text(f'repeated prediction max abs difference: {repeat_error}\nreloaded prediction max abs difference: {reload_error}\n')
    meta={'intended_use':'Research PoC, advisory yield estimate; no autonomous GMP decision','python':sys.version,'sklearn':sklearn.__version__,'random_state':42,'features':list(X.columns),'training_cutoff':str(cut2),'outer_cutoff':str(cut),'calibration_n':n,'interval_half_width':q,'input_range_screen':'per product training min/max; exploratory','sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in model_dir.glob('*.joblib')},'automatic_retraining':False}
    (model_dir/'metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    # SHAP local explanation on the full-training regression model, separate from interval model.
    try:
        import shap
        Xt=final[0].transform(X.loc[test.index]); features=final[0].get_feature_names_out(); explainer=shap.TreeExplainer(final[-1]); sv=explainer.shap_values(Xt); base=float(np.asarray(explainer.expected_value).reshape(-1)[0]); err=float(np.max(np.abs(base+sv.sum(axis=1)-pred)))
        pd.DataFrame({'feature':features,'mean_abs_shap':np.abs(sv).mean(axis=0),'mean_shap':sv.mean(axis=0)}).sort_values('mean_abs_shap',ascending=False).to_csv(out/'shap_global.csv',index=False)
        local=[]
        for j,(_,r) in enumerate(test.iterrows()):
            for f,value,contribution in zip(features,Xt[j],sv[j]): local.append(dict(batch=r.batch,code=r.code,actual=r.tbl_yield,prediction=pred[j],baseline=base,feature=f,input_value=value,shap=contribution))
        pd.DataFrame(local).to_csv(out/'shap_local.csv',index=False); (out/'shap_check.txt').write_text(f'SHAP version: {shap.__version__}\nmax absolute additivity error: {err}\n'); print('SHAP',err)
    except ImportError:
        print('SHAP not installed; run uv add shap==0.50.0 then rerun')
    print('UNCERTAINTY',q,covered.sum(),len(test),'review',int(intervals.review_required.sum()))

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--lab",required=True); p.add_argument("--process",required=True); p.add_argument("--zip",required=True); p.add_argument("--out",default="followup_results"); main(p.parse_args())
