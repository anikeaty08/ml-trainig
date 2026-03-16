import { useState } from "react";
import { Link } from "react-router-dom";

import AgentConsole from "../components/AgentConsole";
import { api } from "../services/api";

export default function Dashboard({ jobs, availableModels, settings, onJobsRefresh, onSettingsChange, providerCatalog }) {
  const latestJobs = jobs.slice(0, 5);
  const [deleteError, setDeleteError] = useState("");
  const [deletingJobId, setDeletingJobId] = useState("");

  async function handleDelete(job) {
    const confirmed = window.confirm(`Delete ${job.filename} and all local artifacts for job ${job.id}?`);
    if (!confirmed) {
      return;
    }
    setDeletingJobId(job.id);
    setDeleteError("");
    try {
      await api.deleteJob(job.id);
      await onJobsRefresh?.();
    } catch (error) {
      setDeleteError(error.message);
    } finally {
      setDeletingJobId("");
    }
  }

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
          <Link className="button secondary" to="/terminal">
            Open chat terminal
          </Link>
        </div>
      </section>

      <section className="card split-card">
        <div>
          <h3>Recent jobs</h3>
          {deleteError ? <p className="error-text">{deleteError}</p> : null}
          <div className="job-list">
            {latestJobs.length === 0 ? <p>No jobs yet.</p> : null}
            {latestJobs.map((job) => (
              <article className="job-row" key={job.id}>
                <Link className="job-link" to={`/results/${job.id}`}>
                  <strong>{job.filename}</strong>
                  <span>{job.status}</span>
                  <span>{job.stage}</span>
                </Link>
                <button className="button secondary slim-button" disabled={deletingJobId === job.id} onClick={() => handleDelete(job)} type="button">
                  {deletingJobId === job.id ? "Deleting..." : "Delete"}
                </button>
              </article>
            ))}
          </div>
        </div>
        <div>
          <h3>Automatic training families</h3>
          <p className="subtle-copy">
            These are the ML models the pipeline may train on your dataset automatically. They are separate from the
            chat-agent model you configure in the terminal.
          </p>
          <div className="preset-list">
            {(availableModels.candidate_families || []).map((item) => (
              <article className="preset-card" key={item}>
                <strong>{item}</strong>
                <p>The pipeline decides when to use this family and compares it against the rest.</p>
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
