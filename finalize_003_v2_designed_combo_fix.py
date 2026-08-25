from pathlib import Path

src = Path('finalize_003_v2_designed_combo.py').read_text(encoding='utf-8')
# Use the measured near-horizontal core from CMU 02_08 instead of the wider windup/recovery window.
src = src.replace("R2,P2=segment(R08,P08,70,102,23)", "R2,P2=segment(R08,P08,76,97,20)")
# Evaluate the final horizontal core after the 5-frame transition blend, not the incoming blend itself.
src = src.replace("a2s=max(0,b1+1); a2e=min(len(rh)-1,b2+2); d2=rh[a2e]-rh[a2s]", "a2s=max(0,b1+4); a2e=min(len(rh)-1,b2-1); d2=rh[a2e]-rh[a2s]")
exec(compile(src, 'finalize_003_v2_designed_combo.py', 'exec'), {'__name__':'__main__'})
