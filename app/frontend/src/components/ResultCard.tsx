import type { PredictResponse, Label, Probabilities } from "../lib/api";

const labels: Label[] = ["Urgency", "Scarcity", "Social Proof", "Normal"];
const colors: Record<Label, string> = { Urgency: "bg-coral", Scarcity: "bg-amber-400", "Social Proof": "bg-violet-500", Normal: "bg-mint" };

function ProbabilityBars({ probabilities }: { probabilities: Probabilities }) {
  return <div className="space-y-3">{labels.map((label) => <div key={label}><div className="mb-1 flex justify-between text-sm"><span>{label}</span><span className="font-semibold">{(probabilities[label] * 100).toFixed(1)}%</span></div><div className="h-2 overflow-hidden rounded-full bg-slate-100 dark:bg-slate-700"><div className={`h-full ${colors[label]}`} style={{ width: `${probabilities[label] * 100}%` }} /></div></div>)}</div>;
}

export default function ResultCard({ result }: { result: PredictResponse }) {
  return <div className="space-y-5">
    <div className="card flex flex-wrap items-center gap-6">
      <div className="relative grid h-28 w-28 place-items-center rounded-full" style={{ background: `conic-gradient(#75e6c2 ${result.confidence_percent}%, #e2e8f0 0)` }}><div className="grid h-20 w-20 place-items-center rounded-full bg-white text-center dark:bg-slate-800"><strong className="text-xl">{result.confidence_percent}%</strong><span className="text-[10px] text-slate-500">confidence</span></div></div>
      <div><p className="muted">Final label</p><h2 className="text-3xl font-bold">{result.label}</h2><p className="muted mt-1">Ensemble prediction #{result.prediction_id}</p></div>
    </div>
    <div className="card"><h3 className="mb-4 font-semibold">Per-model probabilities</h3>{Object.entries(result.per_model).map(([name, prediction]) => <div className="mb-5 last:mb-0" key={name}><p className="mb-2 text-sm font-medium">{name}</p><ProbabilityBars probabilities={prediction.probabilities} /></div>)}</div>
    <div className="card"><h3 className="font-semibold">Phase 1 word hints</h3><p className="muted mt-1">Legacy word hints are not validated model attributions. Use the Robustness evaluation section for measured model sensitivity.</p><div className="mt-4 flex flex-wrap gap-2">{result.top_contributing_words.length ? result.top_contributing_words.map((word) => <span className="rounded-full bg-coral/15 px-3 py-1 text-sm font-medium text-coral" key={word}>{word}</span>) : <span className="muted">No word hints available.</span>}</div></div>
  </div>;
}
