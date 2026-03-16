import { Link } from "react-router-dom";

import AgentConsole from "../components/AgentConsole";

export default function Dashboard({ jobs, availableModels, settings, onSettingsChange, providerCatalog }) {
  const latestJobs = jobs.slice(0, 5);

  return (
    <div className="stack">
      <section className="card hero-card">
        <div>
          <p className="eyebrow">Upload raw data, let the system take over</p>
          <h2>Autonomous pipeline for messy datasets</h2>
          <p>
            No login, no target-column questionnaire, and no cloud dependency by default. Upload tabular, text, or
            time-series CSV data and the app will infer the target, clean the data, train multiple models, compare
            them, and export everything.
          </p>
        </div>
        <div className="cta-row">
          <Link className="button" to="/upload">
            Upload dataset
          </Link>
          <Link className="button secondary" to="/models">
            Configure agent models
          </Link>
        </div>
      </section>

      <section className="card split-card">
        <div>
          <h3>Recent jobs</h3>
          <div className="job-list">
            {latestJobs.length === 0 ? <p>No jobs yet.</p> : null}
            {latestJobs.map((job) => (
              <Link className="job-row" key={job.id} to={`/results/${job.id}`}>
                <strong>{job.filename}</strong>
                <span>{job.status}</span>
                <span>{job.stage}</span>
              </Link>
            ))}
          </div>
        </div>
        <div>
          <h3>Built-in model presets</h3>
          <div className="preset-list">
            {(availableModels.builtin_models || []).map((item) => (
              <article className="preset-card" key={item.name}>
                <strong>{item.label}</strong>
                <p>{item.description || item.path}</p>
              </article>
            ))}
          </div>
        </div>
      </section>

      <AgentConsole
        currentJobId={latestJobs[0]?.id}
        settings={settings}
        onSettingsChange={onSettingsChange}
        providerCatalog={providerCatalog}
      />
    </div>
  );
}
