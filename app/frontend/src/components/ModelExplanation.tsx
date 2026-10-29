import { useState, type ReactNode } from "react";
import { explainText, type Explanation } from "../lib/api";

function highlightedText(result: Explanation): ReactNode[] {
  // API offsets count Unicode code points; JS slice normally counts UTF-16 units.
  const characters = Array.from(result.text);
  const output: ReactNode[] = [];
  let position = 0;
  const scale = Math.max(...result.tokens.map((token) => Math.abs(token.score)), 0.00001);
  for (const [index, token] of result.tokens.entries()) {
    output.push(characters.slice(position, token.start).join(""));
    const intensity = Math.abs(token.score) / scale;
    output.push(
      <mark key={index} title={`${token.text}: ${(token.score * 100).toFixed(2)} percentage points`}
        className="rounded px-0.5 text-inherit"
        style={{ backgroundColor: token.score === 0 ? "transparent" :
          `rgba(${token.score > 0 ? "34, 197, 94" : "244, 114, 182"}, ${0.12 + 0.45 * intensity})` }}>
        {characters.slice(token.start, token.end).join("")}
      </mark>,
    );
    position = token.end;
  }
  output.push(characters.slice(position).join(""));
  return output;
}

export default function ModelExplanation({ text }: { text: string }) {
  const [perturbed, setPerturbed] = useState("");
  const [result, setResult] = useState<Explanation>();
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const explain = async () => {
    setLoading(true); setError(""); setResult(undefined);
    try { setResult(await explainText(text, perturbed)); }
    catch (e) { setError(e instanceof Error ? e.message : "Explanation unavailable"); }
    finally { setLoading(false); }
  };
  return <section className="card mt-6" aria-labelledby="model-explanation-title">
    <h2 id="model-explanation-title" className="text-xl font-semibold">Model explanations</h2>
    <p className="muted mt-2">Explain the text entered above using the actual loaded trained model. Maximum 80 words; this can take several seconds. Text is not saved by this action.</p>
    <label htmlFor="perturbed-copy" className="mt-4 block text-sm font-medium">Optional perturbed text</label>
    <textarea id="perturbed-copy" value={perturbed} onChange={(e) => setPerturbed(e.target.value)} rows={2}
      className="mt-2 w-full rounded-xl border border-slate-200 bg-transparent p-3 dark:border-slate-600" />
    <button onClick={explain} disabled={loading || !text.trim()}
      className="mt-3 rounded-xl bg-ink px-4 py-2 text-white disabled:opacity-50">
      {loading ? "Explaining…" : "Explain loaded model"}
    </button>
    {error && <p role="alert" className="mt-3 text-red-600">{error}</p>}
    {result && <div className="mt-5 space-y-3" aria-live="polite">
      <p><strong>{result.label}</strong> · {(result.confidence * 100).toFixed(1)}% model confidence</p>
      <p className="muted break-all text-sm">{result.model.name}, seed {result.model.seed} · checkpoint SHA-256: {result.model.checkpoint_sha256}</p>
      <p className="whitespace-pre-wrap rounded-lg border border-slate-200 p-4 leading-8 dark:border-slate-600" aria-label="Attributed text">{highlightedText(result)}</p>
      <p className="text-sm">Green: supports the predicted class. Pink: opposes it. Hover for signed probability changes. Colour intensity is relative within this text.</p>
      <details><summary className="cursor-pointer text-sm font-medium">Word contribution values</summary>
        <table className="mt-2 text-sm"><thead><tr><th className="pr-4 text-left">Word</th><th>Probability change (percentage points)</th></tr></thead>
          <tbody>{result.tokens.map((token, index) => <tr key={index}><td className="pr-4">{token.text}</td><td>{(token.score * 100).toFixed(2)}</td></tr>)}</tbody></table>
      </details>
      <p className="muted text-sm">{result.description}</p>
      <p className="muted text-sm">{result.truncation_warning}</p>
      {result.feature_attributions && <details>
        <summary className="cursor-pointer font-medium">Exact TF-IDF feature contributions</summary>
        <p className="muted mt-2 text-sm">{result.feature_attributions.description}</p>
        <p className="mt-2 text-sm">Class: {result.feature_attributions.target_class} · Intercept: {result.feature_attributions.bias.toFixed(4)} · Logit: {result.feature_attributions.logit.toFixed(4)}</p>
        <table className="mt-2 text-sm"><thead><tr><th className="pr-4 text-left">Feature (top 20 by magnitude)</th><th>Logit contribution</th></tr></thead><tbody>
          {result.feature_attributions.features.slice(0, 20).map((feature) => <tr key={feature.feature}><td className="pr-4">{feature.feature}</td><td>{feature.score.toFixed(4)}</td></tr>)}
        </tbody></table>
        <p className="muted text-sm">The logit uses all features; the table displays at most 20.</p>
      </details>}
      {result.comparison && <div className="rounded-lg bg-slate-100 p-4 dark:bg-slate-800">
        <h3 className="font-semibold">Perturbed prediction</h3>
        <p className="whitespace-pre-wrap">{result.comparison.text}</p>
        <p>{result.comparison.label} · {(result.comparison.confidence * 100).toFixed(1)}% model confidence · {result.comparison.prediction_changed ? "Prediction changed" : "Prediction unchanged"}</p>
        <p>Original-class probability change: {(result.comparison.original_class_probability_delta * 100).toFixed(2)} percentage points.</p>
        <p className="muted text-sm">{result.comparison.note}</p>
      </div>}
    </div>}
  </section>;
}
