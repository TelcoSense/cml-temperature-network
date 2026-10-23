"""Regression tests for review corrections; all observations here are synthetic."""
import contextlib
import importlib.util
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd

SCRIPTS=Path(__file__).resolve().parents[1]/"scripts"
sys.path.insert(0,str(SCRIPTS))
def load(name, filename):
    spec=importlib.util.spec_from_file_location(name,SCRIPTS/filename)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module

runner=load("review_runner","run_models.py")
baseline=load("review_baseline","add_per_endpoint_baseline.py")
cold=load("review_cold","fit_coldstarts.py")

class ReviewCorrections(unittest.TestCase):
    def test_endpoint_baseline_uses_full_training_fold_and_agrees_with_backfill(self):
        rows=[]
        for month in [1,2,3]:
            for endpoint,offset in [("a",20.0),("b",30.0)]:
                for i in range(30):
                    target=float(i%9+month)
                    rows.append(dict(time_utc=f"2025-{month:02d}-01 {i//6:02d}:{(i%6)*10:02d}:00+00:00",
                        month=f"2025-{month:02d}", cml_id=endpoint, roof_id="roof-"+endpoint,
                        technology="T",temp_unit=target+offset+(i%7)*0.3,t_ref=target))
        df=pd.DataFrame(rows)
        with tempfile.TemporaryDirectory() as directory:
            d=Path(directory); data=d/"data.csv"; metrics=d/"metrics.csv"; preds=d/"predictions.csv"
            df.to_csv(data,index=False)
            config=d/"test.ini"
            config.write_text(f"""[paths]
ml_dataset_csv = {data}
metrics_csv = {metrics}
predictions_csv = {preds}
[dataset]
target_col = t_ref
[validation]
protocols = last_month
min_train_samples = 1
min_test_samples = 1
[features]
include_clock_features = false
include_geometry_features = false
include_power_features = false
include_categorical_ids = true
[models]
max_train_samples_per_fold = 8
[experiments]
run_all_technologies = true
run_single_technology = false
""",encoding="utf-8")
            def run():
                with patch.object(sys,"argv",["run_models.py","--config",str(config)]), \
                     patch.object(runner,"build_models",return_value={"per_endpoint_offset":(runner.PerEndpointOffsetRegressor(),{})}), \
                     contextlib.redirect_stdout(io.StringIO()):
                    runner.main()
                return pd.read_csv(metrics),pd.read_csv(preds)
            m,p=run()
            self.assertEqual(int(m.iloc[0].n_fit),120)
            train=df[df.month!="2025-03"]; test=df[df.month=="2025-03"]
            expected=baseline.per_endpoint_offset_predict(train,test,"temp_unit","t_ref")
            np.testing.assert_allclose(p.y_pred,expected,rtol=0,atol=1e-12)
            # Changing only held-out reference labels must not change predictions.
            df.loc[df.month=="2025-03","t_ref"]+=1000
            df.to_csv(data,index=False)
            _,p_shifted=run()
            np.testing.assert_allclose(p_shifted.y_pred,p.y_pred,rtol=0,atol=1e-12)

    def test_warmup_missing_slots_do_not_compress_time(self):
        t=np.arange(31)*60.0
        readings=cold.warmup(t,49.0,24.0,420.0)
        full=cold.fit_series("full",pd.Series(readings),None)
        readings[[0,2,5,6,12,20]]=np.nan
        readings[10]=np.inf
        missing=cold.fit_series("gaps",pd.Series(readings),None)
        self.assertAlmostEqual(full["tau_min"],7.0,places=5)
        self.assertAlmostEqual(missing["tau_min"],7.0,places=5)
        self.assertEqual(missing["n_samples"],24)
        self.assertAlmostEqual(missing["t0_c"],24.0,places=5)

    def test_warmup_insufficient_readings_fail_clearly(self):
        with self.assertRaisesRegex(ValueError,"at least four finite"):
            cold.fit_series("short",pd.Series([20,np.nan,22,np.inf,23]),None)

if __name__=="__main__":
    unittest.main()
