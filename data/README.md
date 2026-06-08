# Data

Raw CML telemetry, exact link locations, IP addresses, physical site names,
station names, and real technology names are not distributed with this code.

To run the pipeline, prepare a private or anonymized dataset and point
`configs/config.ini` at it. The expected 10-minute analysis columns are:

`time_utc, cml_id, technology, temp_unit, t_ref, azimuth, altitude, sun, day, hour`

Optional columns such as `trsl`, `rsl`, and `tsl` are used automatically when present.
