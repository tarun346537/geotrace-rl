# Gravity survey inputs

Keep actual survey files and model weights outside version control. Files under this directory are ignored except this guide and the empty-directory marker.

Create a small seeded fixture in the same assignment schema used by the finite-grid pipeline:

```bash
python scripts/generate_sample_data.py --surveys 10 --output outputs/sample-traces.jsonl
```

This writes 40 traces across 10 synthetic surveys with train/validation/test membership assigned by survey.

For measured inputs, provide `x_m,gravity_microgal` CSV observations, a finite parameter-grid JSON file, survey ID, split and source release URL. Import with `python -m geotrace.data import-survey --help`. [The data contract](../docs/DATA.md) defines coordinates, sign, units and provenance fields.

All traces from one survey must stay in the same split. Group nearby surveys or shared geology together when those could leak correlated structure. Keep separate held-out receiver observations outside the model-visible task for policy benchmarks.
