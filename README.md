# CAPE
### Individual Treatment Effect Estimation
**Run CAPE**
```
python train_causal.py --loop 10 -m cape_cau -d NI --i_t 1
```
**Run a baseline model**
```
python train_causal_baselines.py --loop 10 -m cfrmmd -d NI --i_t 1
```

### Event Forecasting with Causal information
**Run CAPE**
```
python train_event_with_causal.py --loop 10 -m cape -d NI 
```
**Add noise to data**
```
python train_event_with_causal.py --loop 10 -m cape -d NI --train_noise 0.1
```
