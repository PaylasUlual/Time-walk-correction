# Time-Walk Correction in LGAD Sensors Using Machine Learning Regression

## Physics Background

A Low-Gain Avalanche Detector is a silicon sensor with a thin, highly doped multiplication layer under the collecting implant. Electrons and holes drifting into this layer are amplified by ionization. Due to the short drift distance and fast rising edge thin sensors can also give large, fast signals and these steep pulses are ideal for picosecond timing measurements. 

The arrival time is measured when the amplified pulse crosses a threshold instead of at the start of the pulse. A large pulse crosses the threshold early on its rising edge while for a smaller amplitude pulse this occurs later. The measured time depends on the amplitude of the pulse instead of the true arrival time. This is the time-walk effect. It is represented as;

$\Delta t = t_{raw} - t_{true} \propto \frac{1}{A}$

Since the amplitude fluctuates between events the time-walk effect causes a spread of the time measured and worsens the time resoultion $\sigma_t$. This is more significant for small pulses since the rising edge reaches the threshold more slowly and is more sensitive to noise. There is also other factors affecting the shape of the pulse such as particle position and the angle of incidence that could result in charge sharing between pixels and a tilted path respectively.

ALICE3 has a target of a single-hit resolution of 20 ps [1] and the current MadPix prototype [2] that this project is based on reaches about 75 ps. A hardware Constant Fraction Discriminator (CFD) is used as a correction to remove most of the amplitude dependence by triggering at a fixed fraction of the pulse height. However, this is a single variable method of correction and it does not take into account the position inside the pixel, charge sharing or incidence angle. 

The remanining time-walk depends on several correlated variables non-linearly. A gradient boosted tree model can learn the dependence on these variables and improve on the CFD correction. Therefore, the goal of this project is to implement this and investigate the further improvements this method can provide.

## Method

The dataset is generated with allpix-squared based on the MadPix sensor. The source simulated is 120 GeV protons from a flat rectangular beam tilted in the x-z direction by 0, 5, 10, 15, 20, 25°. 17000 events per angle is generated. A linear field model is used with 230 V bias and hole propagation. CSADigitizer is used with a rise time of 0.5 ns, threshold of 10 mV and 10 ps TDC bin. The details of this can be found in the file `detector.conf`. `convert_to_hdf5.py` keeps only one row per event which is the fired pixel with the most charge and removes events where no pixel fired. As a result there are 85899 events that can be used. 

The model uses 10 features; `pulse_integral`, `pixel_charge`, `inv_pulse_integral`, `cluster_charge`, `cluster_size`, `charge_per_pixel`, `col_frac`, `row_fraw`, `angle_deg` and `v_bias`. This is further explained in Notebook 3.

Initially a software CFD was used with a parametric fit $\Delta t = \frac{a}{A} + b$ where A= `pulse_integral` that is subtracted from the data. This fit was done both globally and per angle and the results for both cases were evaluated.

For the machine learning model XGBRegressor was used with up to 3000 trees, `max_depth` of 5, `learning_rate` of 0.05, `subsample` of 0.8, `colsample_bytree` of 0.8 and `min_child_weight` of 10. The data is split 70/15/15 between the training, validation and test sets. Features are standardised with the training set only. The model predicts $\Delta t$ which is subtracted from the measured value.

$\sigma_t$ is the $\sigma$ of a Gaussian fit to the residuals within $\pm 200 ps$ of their median. The RMS is not used as it is dominated by the non-Gaussian tail. The results are evaluated and compared based on the test set.

## Results

| Method | σ_t [ps] | Improvement vs raw |
|---|---|---|
| Raw (no correction) | 61.1 | – |
| CFD, global fit | 55.0 | 10.0 % |
| CFD, per-angle fit | 54.9 | 10.1 % |
| **XGBoost** | **32.8** | **46.2 %** |

The ML based correction improves the single-hit time resolution from $\sigma_t = 54.9 ps$ to $\sigma_t = 32.8 ps$. This is an improvement of 40.2% over the CFD method.

Furthermore, the timing resolution against incidence angles was evaluated for each angle and a SHAP feature importance chart was made. These can be found in Notebook 4 where the results are discussed.

## Reproduce

The whole project runs inside Docker (allpix-squared image + Python ML stack). From the project root:

```bash
#1.Build the image
docker build -t timewalk .

#2.Run the full pipeline: simulation -> HDF5 conversion -> notebooks 1-4
docker run --rm -v $(pwd):/project timewalk bash run.sh
```

`run.sh` runs these steps in order:

1. `simulation/generate_configs.py` and `simulation/run_simulation.sh`: allpix-squared, 6 angles × 17,000 events.
2. `src/convert_to_hdf5.py`: ROOT output → `data/hits.h5`.
3. Notebooks `1-visualization`, `2-cfd_baseline`, `3-ml_regression`, `4-evaluation_results`, executed in place. Plots go to `notebooks/plots/`.

For JupyterLab:

```bash
docker run -p 8888:8888 -v $(pwd):/project timewalk
```

## References
[1] ALICE Collaboration, "Letter of intent for ALICE 3: A next-generation heavy-ion experiment
   at the LHC", arXiv:2211.02491 [physics.ins-det] (2022). https://arxiv.org/abs/2211.02491

[2] U. Follo, "Development of CMOS LGAD sensors for the ALICE 3 Time of Flight detector",
   arXiv:2601.02823 [physics.ins-det] (2026). https://arxiv.org/abs/2601.02823
