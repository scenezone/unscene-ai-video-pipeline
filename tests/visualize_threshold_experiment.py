"""Create an offline interactive HTML report beside a threshold comparison JSON."""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Detection threshold comparison</title>
<style>
:root{font-family:system-ui,sans-serif;color:#18283c;background:#f3f6fa;font-size:16px}*{box-sizing:border-box}body{margin:0}main{max-width:1240px;margin:auto;padding:32px 24px 64px}h1{font-size:32px;margin:8px 0}h2{font-size:21px;margin:0 0 12px}p{line-height:1.55}.muted{color:#526278}.eyebrow{font-size:12px;letter-spacing:2px;font-weight:700;color:#345cb3}.cards{display:flex;gap:16px;flex-wrap:wrap;margin:24px 0}.card{background:white;border:1px solid #dce4ef;border-radius:12px;padding:18px;flex:1;min-width:150px}.card strong{display:block;font-size:28px}.card span{color:#526278;font-size:14px}section{background:white;border:1px solid #dce4ef;border-radius:14px;padding:24px;margin-top:20px}.note{border-left:4px solid #d59821;background:#fff8e8;padding:12px 16px;font-size:14px}.controls{display:flex;gap:16px;align-items:end;flex-wrap:wrap;margin:16px 0}label{display:flex;flex-direction:column;gap:6px;font-size:14px;font-weight:600}select,button{font:inherit;padding:9px 12px;border:1px solid #b9c6d7;border-radius:7px;background:white;color:#18283c}button{cursor:pointer}button:hover{background:#eaf1fc}a{color:#2456aa}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:11px 12px;text-align:right;border-bottom:1px solid #e3e9f1;white-space:nowrap}th:first-child,td:first-child{text-align:left}th{background:#f5f8fc}caption{text-align:left;color:#526278;padding:0 0 12px}.bars{margin-top:20px}.barrow{display:grid;grid-template-columns:65px 1fr 70px;gap:12px;align-items:center;margin:12px 0;font-size:14px}.track{height:26px;background:#eef2f7;border-radius:4px;overflow:hidden;display:flex}.clothing{background:#4263db}.objects{background:#0d9488}.legend{font-size:13px;display:flex;gap:20px}.dot{display:inline-block;width:12px;height:12px;margin-right:5px;border-radius:3px}.views{display:grid;grid-template-columns:1fr 1fr;gap:20px}.views figure{margin:0;min-width:0}.views img{width:100%;height:620px;object-fit:contain;background:#18212d;border-radius:8px}.views figcaption{font-size:14px;line-height:1.5;margin:8px 0}.slider{flex:1;min-width:240px}input[type=range]{width:100%}.small{font-size:13px}.framecounts{min-height:50px}details{margin-top:20px}summary{cursor:pointer}footer{margin-top:20px;color:#526278;font-size:13px}@media(max-width:700px){main{padding:20px 12px}section{padding:16px}.views{grid-template-columns:1fr}.views img{height:auto}h1{font-size:26px}}
</style></head><body><main>
<div class="eyebrow">GROUNDINGDINO · EXPERIMENT REVIEW</div><h1>Find the threshold trade-off</h1>
<p class="muted" id="subtitle"></p>
<div class="cards"><div class="card"><strong id="samples"></strong><span>sampled video frames</span></div><div class="card"><strong id="settings"></strong><span>box thresholds compared</span></div><div class="card"><strong id="labels"></strong><span>clothing + object labels</span></div><div class="card"><strong id="text"></strong><span>fixed text threshold</span></div></div>
<p class="note">These are model confidence scores, not accuracy measurements. More boxes may include false positives; fewer boxes may miss real items. Higher average confidence is expected when low scores are filtered out. There is no automatically proven “best” threshold.</p>
<section><h2>1. How many detections survive?</h2><p class="muted small">All thresholds use the same sampled frames and predictions. Counts are boxes, not unique products.</p>
<div class="legend"><span><i class="dot clothing"></i>Clothing</span><span><i class="dot objects"></i>Objects</span></div><div id="bars" class="bars"></div><div class="scroll"><table id="totals"></table></div></section>
<section><h2>2. Compare each prompted label</h2><div class="controls"><label>Measure<select id="metric"><option value="detections">Detection count</option><option value="coverage">Sampled frames with a detection (%)</option><option value="mean">Mean confidence of retained boxes</option></select></label></div>
<p class="muted small" id="metricHelp"></p><div class="scroll"><table id="heatmap"></table></div></section>
<section><h2>3. Inspect the same frame at two thresholds</h2><p class="muted small">Look for incorrect boxes and real items that disappear when you raise the threshold. Image annotations show all prompt categories.</p>
<div class="controls"><label>Left threshold<select id="left"></select></label><label>Right threshold<select id="right"></select></label><button id="previous" type="button">Previous frame</button><button id="next" type="button">Next frame</button></div>
<label class="slider">Sample position<input id="frame" type="range" min="0" step="1" value="0"></label><p id="position" class="muted"></p>
<div class="views"><figure><a id="leftLink" target="_blank" rel="noopener"><img id="leftImage" alt="Left threshold annotated frame"></a><figcaption id="leftCaption"></figcaption><div id="leftCounts" class="small framecounts"></div></figure><figure><a id="rightLink" target="_blank" rel="noopener"><img id="rightImage" alt="Right threshold annotated frame"></a><figcaption id="rightCaption"></figcaption><div id="rightCounts" class="small framecounts"></div></figure></div></section>
<section><h2>How to use this comparison</h2><p>Start by comparing 0.30 with 0.55. Move through the sampled frames and inspect the objects and clothing you care about. Lower the threshold if real items vanish; raise it if irrelevant boxes dominate. Frame coverage only says whether the model returned a box—it does not say that box is correct.</p><p><a href="summary.csv" download>Download compact summary CSV</a> · <a href="comparison.json">Open full detection JSON</a></p><details><summary>Experiment method and limitations</summary><p id="method"></p><ul id="limitations"></ul></details></section>
<footer>Offline report. Keep this HTML in its experiment folder so the saved frame images remain available.</footer>
</main><script id="data" type="application/json">__DATA__</script><script>
'use strict';
const data=JSON.parse(document.getElementById('data').textContent), $=id=>document.getElementById(id);
const esc=v=>String(v).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt=v=>Number(v).toFixed(2), settings=data.thresholds;
const sum=(t,cat)=>Object.values(t.summary.by_category[cat]||{}).reduce((s,x)=>s+x.detections,0);
const cells=(values,tag='td')=>values.map(v=>`<${tag}>${v}</${tag}>`).join('');
$('samples').textContent=data.completed_samples;$('settings').textContent=settings.length;$('labels').textContent=data.prompts.length;$('text').textContent=fmt(data.text_threshold);
$('subtitle').textContent=`${data.video_name} · ${data.sampling} · Run status: ${data.status}`;
$('method').textContent=data.method;$('limitations').innerHTML=data.limitations.map(x=>`<li>${esc(x)}</li>`).join('');
const maxTotal=Math.max(1,...settings.map(t=>t.summary.total_detections));
$('bars').innerHTML=settings.map(t=>`<div class="barrow"><strong>${fmt(t.box_threshold)}</strong><div class="track" aria-label="${sum(t,'clothing')} clothing boxes and ${sum(t,'objects')} object boxes"><div class="clothing" style="width:${100*sum(t,'clothing')/maxTotal}%"></div><div class="objects" style="width:${100*sum(t,'objects')/maxTotal}%"></div></div><span>${t.summary.total_detections}</span></div>`).join('');
$('totals').innerHTML='<thead><tr>'+cells(['Box threshold','Clothing boxes','Object boxes','Total boxes'],'th')+'</tr></thead><tbody>'+settings.map(t=>'<tr>'+cells([fmt(t.box_threshold),sum(t,'clothing'),sum(t,'objects'),t.summary.total_detections])+'</tr>').join('')+'</tbody>';
function heatmap(){
 const metric=$('metric').value;
 $('metricHelp').textContent=metric==='coverage'?'Percentage of sampled frames with at least one returned box. This is not recall: actual presence was not labeled.':metric==='mean'?'Average confidence among retained boxes. A dash means no detections. This is not accuracy.':'Number of retained boxes across all sampled frames. Multiple boxes can refer to the same item. Darker cells indicate larger values within each row.';
 const value=s=>metric==='coverage'?s.frame_coverage*100:metric==='mean'?s.mean_confidence:s.detections;
 $('heatmap').innerHTML='<thead><tr>'+cells(['Category / label',...settings.map(t=>fmt(t.box_threshold))],'th')+'</tr></thead><tbody>'+data.prompts.map(p=>{
  const values=settings.map(t=>value(t.summary.by_category[p.category][p.label]));
  const max=metric==='mean'?1:metric==='coverage'?100:Math.max(1,...values.filter(v=>v!==null));
  return '<tr><td>'+esc(p.category+' / '+p.label)+'</td>'+values.map(v=>`<td style="background:rgba(66,99,219,${v===null?0:0.04+0.22*v/max})">${v===null?'—':metric==='mean'?fmt(v):metric==='coverage'?v.toFixed(0)+'%':v}</td>`).join('')+'</tr>';
 }).join('')+'</tbody>';
}
for(const side of ['left','right']){
 $(''+side).innerHTML=settings.map((t,i)=>`<option value="${i}">${fmt(t.box_threshold)}</option>`).join('');
 const index=settings.findIndex(t=>Math.abs(t.box_threshold-(side==='left'?.3:.55))<1e-8);
 $(side).value=index>=0?String(index):side==='left'?'0':String(settings.length-1);
 $(side).addEventListener('change',frames);
}
const count=settings[0].frames.length;$('frame').max=count-1;
function frames(){
 const i=Number($('frame').value), base=settings[0].frames[i];
 $('position').textContent=`Sample ${i+1} of ${count} · ${base.timestamp.toFixed(2)} seconds · Video frame ${base.frame}`;
 for(const side of ['left','right']){
  const t=settings[Number($(side).value)], f=t.frames[i];
  const url=f.frame_path.split('/').map(encodeURIComponent).join('/');
  $(side+'Image').src=url;$(side+'Link').href=url;
  $(side+'Caption').textContent=`Box score > ${fmt(t.box_threshold)} · ${f.total} boxes · Click image to view full size`;
  $(side+'Counts').textContent=data.prompts.map(p=>`${p.category}/${p.label}: ${f.counts[p.category][p.label]}`).join(' | ');
 }
 $('previous').disabled=i===0;$('next').disabled=i===count-1;
}
$('metric').addEventListener('change',heatmap);$('frame').addEventListener('input',frames);
$('previous').addEventListener('click',()=>{$('frame').value=Math.max(0,Number($('frame').value)-1);frames()});
$('next').addEventListener('click',()=>{$('frame').value=Math.min(count-1,Number($('frame').value)+1);frames()});
heatmap();frames();
</script></body></html>'''


def build_report(source):
    report = json.loads(source.read_text(encoding="utf-8"))
    settings = report["thresholds"]
    if not settings or not settings[0]["frames"]:
        raise ValueError("The experiment has no completed frames to visualize")
    indices = [f["frame"] for f in settings[0]["frames"]]
    for setting in settings:
        if [f["frame"] for f in setting["frames"]] != indices:
            raise ValueError("Thresholds must contain matching frame samples")
        for frame in setting["frames"]:
            image = (source.parent / frame["frame_path"]).resolve()
            if not image.is_relative_to(source.parent.resolve()) or not image.is_file():
                raise ValueError(f"Missing or invalid frame path: {frame['frame_path']}")
            counts = {p["category"]: {} for p in report["prompts"]}
            for p in report["prompts"]:
                counts[p["category"]][p["label"]] = sum(
                    r["status"] == "detected" and r["category"] == p["category"]
                    and r["prompt_label"] == p["label"] for r in frame["results"])
            frame["counts"] = counts
            frame["total"] = sum(sum(labels.values()) for labels in counts.values())
            del frame["results"]
    report["video_name"] = Path(report["video"]).name
    payload = json.dumps(report, ensure_ascii=True).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    target = source.with_name("comparison.html")
    target.write_text(HTML.replace("__DATA__", payload), encoding="utf-8")
    with source.with_name("summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["box_threshold", "category", "label", "detections", "frames_with_detection",
                         "frames_without_detection", "frame_coverage", "mean_confidence"])
        for setting in settings:
            for category, labels in setting["summary"]["by_category"].items():
                for label, stats in labels.items():
                    # Keep editable prompt labels from becoming spreadsheet formulas.
                    safe_label = "'" + label if label.startswith(("=", "+", "-", "@")) else label
                    writer.writerow([setting["box_threshold"], category, safe_label,
                                     *[stats[k] for k in ("detections", "frames_with_detection",
                                                         "frames_without_detection", "frame_coverage", "mean_confidence")]])
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comparison", nargs="?", type=Path, help="Comparison JSON; defaults to the latest experiment")
    args = parser.parse_args()
    source = args.comparison
    if source is None:
        candidates = sorted((ROOT / "outputs/threshold_experiments").glob("*/comparison.json"))
        if not candidates:
            parser.error("No threshold experiment found")
        source = candidates[-1]
    print(build_report(source.resolve()))


if __name__ == "__main__":
    main()
