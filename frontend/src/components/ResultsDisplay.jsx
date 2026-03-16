import { api } from "../services/api";

export default function ResultsDisplay({ jobId, result }) {
  const comparison = result?.comparison_json || [];
  const summary = result?.summary_json || {};
  const metrics = result?.metrics_json || {};
  const detection = summary.detection || {};
  const training = summary.training || {};

  return (
    <div className="stack">
      <section className="card hero-card">
        <div>
          <p className="eyebrow">Winning model</p>
          <h2>{comparison[0]?.name || "Pending"}</h2>
          <p>
            {Object.entries(metrics)
              .map(([key, value]) => `${key}: ${value}`)
              .join(" | ")}
          </p>
          <p>
            Dataset type: {detection.dataset_type || result.data_type} | Problem type: {detection.problem_type || result.problem_type}
          </p>
        </div>
        <div className="download-grid">
          <a className="button secondary" href={api.getDownloadUrl(jobId, "model")}>
            Download model
          </a>
          <a className="button secondary" href={api.getDownloadUrl(jobId, "code")}>
            Download code
          </a>
          <a className="button secondary" href={api.getDownloadUrl(jobId, "report")}>
            Download report
          </a>
          <a className="button secondary" href={api.getDownloadUrl(jobId, "cleaned-data")}>
            Download cleaned data
          </a>
          <a className="button secondary" href={api.getDownloadUrl(jobId, "bundle")}>
            Download full bundle
          </a>
        </div>
      </section>

      <section className="card">
        <h3>Model ranking</h3>
        <div className="comparison-table">
          <table>
            <thead>
              <tr>
                <th>Rank</th>
                <th>Model</th>
                <th>Primary score</th>
                <th>CV mean</th>
                <th>CV std</th>
                <th>Train time</th>
              </tr>
            </thead>
            <tbody>
              {comparison.map((row) => (
                <tr key={row.rank}>
                  <td>{row.rank}</td>
                  <td>{row.name}</td>
                  <td>{row.primary_score}</td>
                  <td>{row.cv_score_mean}</td>
                  <td>{row.cv_score_std}</td>
                  <td>{row.training_time_seconds}s</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="card split-card">
        <div>
          <h3>Cleaning summary</h3>
          <pre>{JSON.stringify(summary.cleaning || {}, null, 2)}</pre>
        </div>
        <div>
          <h3>Dataset analysis</h3>
          <pre>{JSON.stringify(summary.analysis || {}, null, 2)}</pre>
        </div>
      </section>

      <section className="card split-card">
        <div>
          <h3>Training workflow</h3>
          <pre>{JSON.stringify(training, null, 2)}</pre>
        </div>
        <div>
          <h3>Error analysis</h3>
          <pre>{JSON.stringify(summary.error_analysis || {}, null, 2)}</pre>
        </div>
      </section>
    </div>
  );
}
