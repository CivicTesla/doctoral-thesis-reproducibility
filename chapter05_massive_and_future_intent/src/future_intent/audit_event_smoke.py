import argparse,csv,glob,math
from pathlib import Path
def main():
 p=argparse.ArgumentParser(); p.add_argument('--runs-dir',type=Path,required=True); p.add_argument('--expected',type=int,required=True); a=p.parse_args()
 files=list(a.runs_dir.glob('*.csv')); assert len(files)==a.expected,(len(files),a.expected)
 for f in files:
  r=next(csv.DictReader(f.open())); assert r['completed']=='True' and r['invariants_passed']=='True'
  assert float(r['transmitted_bytes'])>0 and float(r['field_update_count'])>0
  assert math.isfinite(float(r['active_route_usable_source_age_p90']))
  assert 0<=float(r['drop_ratio'])<=1
 print(f'{len(files)}/{a.expected} smoke cases passed')
if __name__=='__main__': main()
