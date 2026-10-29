import { useState } from "react";
import { AlertCircle, LoaderCircle, Zap } from "lucide-react";
import { PageTitle } from "../App";
import { predict, type PredictResponse } from "../lib/api";
import ResultCard from "../components/ResultCard";
import ModelExplanation from "../components/ModelExplanation";

export default function Analyze() {
  const [text, setText] = useState("");
  const [result, setResult] = useState<PredictResponse>();
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const analyze = async () => { if (!text.trim()) return; setLoading(true); setError(""); try { setResult(await predict(text)); } catch (e) { setError(e instanceof Error ? e.message : "Unable to analyze text"); } finally { setLoading(false); } };
  return <><PageTitle icon={Zap} title="Analyze product copy">Spot urgency, scarcity, and social proof before they influence your customers.</PageTitle>
    <div className="grid gap-6 lg:grid-cols-[1fr_1fr]">
      <div className="card h-fit"><label className="mb-2 block font-semibold" htmlFor="copy">E-commerce text</label><textarea id="copy" value={text} onChange={(e) => setText(e.target.value)} placeholder="Paste a product headline, offer, or checkout message..." rows={10} className="w-full resize-none rounded-xl border border-slate-200 bg-slate-50 p-4 outline-none ring-mint focus:ring-2 dark:border-slate-600 dark:bg-slate-900" /><button disabled={loading || !text.trim()} onClick={analyze} className="mt-4 flex w-full items-center justify-center gap-2 rounded-xl bg-ink px-4 py-3 font-semibold text-white transition hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-50">{loading && <LoaderCircle className="animate-spin" size={18} />}{loading ? "Analyzing..." : "Analyze copy"}</button>{error && <p role="alert" className="mt-4 flex gap-2 text-sm text-red-600"><AlertCircle size={18} />{error}</p>}</div>
      {result ? <ResultCard result={result} /> : <div className="card flex min-h-[300px] flex-col items-center justify-center text-center"><ShieldIcon /><h2 className="mt-4 font-semibold">Your analysis appears here</h2><p className="muted mt-1 max-w-sm">Paste text on the left to see the ensemble decision and model explanations.</p></div>}
    </div><ModelExplanation text={text} /></>;
}
function ShieldIcon() { return <div className="grid h-16 w-16 place-items-center rounded-2xl bg-mint/20 text-3xl">🛡️</div>; }
